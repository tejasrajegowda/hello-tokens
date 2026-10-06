"""CUDA graphs: record the one-token step once, replay it for every token.

Profiling (lesson 11) showed that a generation step is overhead-bound: each step launches about 200
small GPU operations, and launching them from Python costs more time than the GPU spends running them.
A CUDA graph removes that cost. The GPU work of one step is recorded once; each later step is a single
"replay" call, which sends the whole recorded sequence to the GPU at once, with no Python in between.

A recording keeps everything exactly as it was: the same operations, the same shapes and the same memory
addresses. So the step must never change shape, and every number that changes between steps (which
token, which position) must live in a tensor whose memory is reused. GPT.decode_step is written that
way. Before each replay the new token and position are copied into the recorded input tensors, and
after it the logits are read from the recorded output tensor.

The recording is made once per model and attention kernel, made again whenever the weights move in memory
(a precision switch, quantization), and reused by every later story: its KV cache is emptied by moving
the pointer back to 0, never reallocated.

`torch.compile(mode="reduce-overhead")` would do the same automatically, but it needs Triton, which
this Windows setup doesn't have; recording by hand needs nothing extra and shows how it works.
"""

import weakref

import torch

from hello_tokens.model.gpt import GPT


class OneTokenStep:
    """The one-token step of `model` with its own KV cache, recorded as a CUDA graph on a GPU.

    On a CPU there are no CUDA graphs: the same fixed-shape step runs as ordinary code. The results are
    the same either way, which is what the CPU tests check.
    """

    def __init__(self, model: GPT, warmup: int = 3):
        # A weak reference: the step is stored per model (below), and a strong one would keep the model,
        # and with it the recording and its cache, alive for as long as the program runs.
        self.model = weakref.ref(model)
        self.cache = model.new_cache()
        device = self.cache.keys.device
        # The recorded inputs: copy the next token and its position in here before each replay.
        self.ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.graph = None
        if device.type != "cuda":
            return
        with torch.no_grad():
            # Run a few steps first, on a side stream, as PyTorch requires before recording: start-up
            # work (choosing kernels, allocating memory) must happen outside the recording. These
            # steps write junk at position 0, which is overwritten when the next prompt is read.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    model.decode_step(self.ids, self.position, self.cache)
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # Recorded, not run. `self.logits` is the recorded output tensor: each replay
                # overwrites it.
                self.logits = model.decode_step(self.ids, self.position, self.cache)

    def __call__(self, token: int) -> torch.Tensor:
        """Feed one token at the cache's current length; return the logits for the next one, (vocab,).

        The returned tensor is overwritten by the next call, so use it (or copy it) before then.
        """
        self.ids.fill_(token)
        self.position.fill_(self.cache.length)
        if self.graph is not None:
            self.graph.replay()
            logits = self.logits
        else:
            logits = self.model().decode_step(self.ids, self.position, self.cache)
        self.cache.advance(1)
        return logits[0]


# One recorded step per model and attention kernel (switching the kernel changes which operations run).
# Weak references let a model and its recordings be freed together.
_steps: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _weight_addresses(model: GPT) -> tuple[int, ...]:
    """Where every weight and buffer lives in memory. A recording reads the weights from these exact
    addresses, so a change of precision, a move to another device, or quantization (anything that puts
    the weights somewhere else) makes it stale: replaying it would read memory that may have been freed."""
    return tuple(t.data_ptr() for t in (*model.parameters(), *model.buffers()))


def one_token_step(model: GPT) -> OneTokenStep:
    """The recorded step for this model in its current state, made on first use and then reused."""
    per_model = _steps.setdefault(model, {})
    fused, addresses = model.blocks[0].attention.fused, _weight_addresses(model)
    if fused not in per_model or per_model[fused][0] != addresses:
        per_model[fused] = (addresses, OneTokenStep(model))  # replaces (and frees) a stale recording
    step = per_model[fused][1]
    step.cache.rollback(0)  # a new story: forget the last one
    return step
