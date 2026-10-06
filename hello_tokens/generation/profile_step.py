"""Where the time of one generation step goes.

A step is what `generate` does per new token: a forward pass, then sampling. It can run three ways:

- "full": the step re-reads a window of the last `context` tokens, as `generate` does without a cache.
- "cache": the first `context - 1` tokens are already in a KV cache; the step feeds one token at position
  `context - 1` with the ordinary cached forward pass.
- "graphs": the same one-token step, replayed from a recorded CUDA graph (generation/graphs.py). On a
  CPU there are no graphs, and the same fixed-shape step runs as ordinary code.

Two measurements:

- wall time: what the user waits, start to finish, with the GPU synchronized before the clock is read.
- GPU busy time: the time the GPU's kernels actually ran, from the PyTorch profiler.

CPU and GPU work overlap, so the two don't add up. The gap between them is time the GPU sat idle,
waiting for Python and for the CPU to launch the next kernel. If wall time barely changes between a
16-token and a 256-token step, the maths isn't the bottleneck: the overhead is.
"""

import statistics
import time
from dataclasses import dataclass

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from hello_tokens.generation.graphs import one_token_step
from hello_tokens.generation.sampling import sample_next
from hello_tokens.model.gpt import GPT

MODES = ("full", "cache", "graphs")


@dataclass(frozen=True)
class StepProfile:
    context: int  # tokens read by the step
    wall_ms: float  # median wall time of one step
    spread_ms: float  # interquartile range of the wall time: how noisy the measurement was
    gpu_busy_ms: float  # kernel time per step (0 on the CPU)
    kernels: float  # GPU kernels launched per step (0 on the CPU)

    @property
    def gpu_idle_fraction(self) -> float:
        return max(0.0, 1 - self.gpu_busy_ms / self.wall_ms)


@torch.no_grad()
def profile_steps(
    model: GPT, contexts: list[int], rounds: int = 200, warmup: int = 20, profiled: int = 20, seed: int = 0,
    mode: str = "full",
) -> list[StepProfile]:
    """Profile one generation step at each context length, run the way `mode` names.

    Every mode reads the same random windows, so at a given context the three modes compute the same
    logits. In "cache" and "graphs" the cache is rolled back to `context - 1` after each step, so every
    timed step runs at the same position. With "graphs" one recording serves every context: when the
    context changes, its cache is refilled with that context's tokens before the clock starts.

    With a CUDA graph, the profiler may report the replayed kernels one by one or not at all, depending
    on the CUDA profiling backend. The wall time is therefore the primary number in that mode, and the
    kernel count is reported as measured.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    if mode != "full" and min(contexts) < 2:
        raise ValueError(f"the {mode!r} step needs at least one token in the cache: contexts must be >= 2")
    model.eval()
    device = next(model.parameters()).device
    on_gpu = device.type == "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)
    windows = {c: torch.randint(0, model.config.vocab_size, (1, c), device=device, generator=generator)
               for c in contexts}

    def sample(logits: torch.Tensor) -> None:
        sample_next(logits, temperature=0.8, top_p=0.95, generator=generator)

    def prepare(c: int) -> None:
        """Untimed set-up before a step at context `c`; only "graphs" needs any."""

    if mode == "full":
        def step(c: int) -> None:
            sample(model(windows[c])[0, -1])
    elif mode == "cache":
        caches = {c: model.new_cache() for c in windows}
        for c, ids in windows.items():
            model(ids[:, :-1], caches[c])

        def step(c: int) -> None:
            logits = model(windows[c][:, -1:], caches[c])[0, -1]
            caches[c].rollback(c - 1)
            sample(logits)
    else:
        recorded = one_token_step(model)
        last_tokens = {c: int(ids[0, -1]) for c, ids in windows.items()}  # read once, not per step
        loaded = 0  # the context whose tokens are in the recorded step's cache (0: none yet)

        def prepare(c: int) -> None:
            nonlocal loaded
            if loaded != c:
                recorded.cache.rollback(0)
                model(windows[c][:, :-1], recorded.cache)
                loaded = c

        def step(c: int) -> None:
            logits = recorded(last_tokens[c])
            recorded.cache.rollback(c - 1)
            sample(logits)

    def sync() -> None:
        if on_gpu:
            torch.cuda.synchronize()

    for _ in range(warmup):
        for c in windows:
            prepare(c)
            step(c)
    sync()

    # Interleave the contexts round by round, so a slow patch (a background task, the CPU changing
    # speed) hits all of them alike instead of skewing one.
    times: dict[int, list[float]] = {c: [] for c in contexts}
    for _ in range(rounds):
        for c in windows:
            prepare(c)
            sync()  # set-up work must finish before the clock starts
            started = time.perf_counter()
            step(c)
            sync()
            times[c].append(time.perf_counter() - started)

    results = []
    for c in windows:
        busy_ms, kernels = 0.0, 0.0
        if on_gpu:
            prepare(c)
            sync()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(profiled):
                    step(c)
                sync()
            gpu_events = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
            busy_ms = sum(e.device_time for e in gpu_events) / profiled / 1000  # microseconds -> ms
            kernels = len(gpu_events) / profiled
        quartiles = statistics.quantiles(times[c], n=4)
        results.append(StepProfile(c, statistics.median(times[c]) * 1000,
                                   (quartiles[2] - quartiles[0]) * 1000, busy_ms, kernels))
    return results
