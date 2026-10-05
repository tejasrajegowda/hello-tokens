"""Sampling: turning the model's scores for the next token into a choice.

Three dials:
- temperature: how adventurous. Scores are divided by it before softmax. Below 1 sharpens the
  distribution toward the likeliest tokens; above 1 flattens it. 0 means "always take the top token".
- top_k: only the k likeliest tokens may be chosen.
- top_p (nucleus): only the smallest set of likeliest tokens whose probabilities add up to p.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from hello_tokens.generation.graphs import one_token_step
from hello_tokens.model.gpt import GPT


def filtered_probabilities(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """The distribution a token is actually drawn from, after the three dials.

    Speculative decoding compares two models' distributions, so it needs exactly this one, not the
    raw softmax. Temperature 0 puts all the probability on the top token.
    """
    if temperature == 0:
        return torch.nn.functional.one_hot(torch.argmax(logits), logits.numel()).float()
    logits = logits.float() / temperature
    if top_k is not None:
        kth_best = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))
    if top_p is not None:
        sorted_logits, order = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Drop a token if the tokens ranked above it already reach p. The top token always stays.
        drop = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
        logits = logits.masked_fill(drop.scatter(0, order, drop), float("-inf"))
    return torch.softmax(logits, dim=-1)


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Choose one token id from a 1-D tensor of scores over the vocabulary."""
    if temperature == 0:
        return int(torch.argmax(logits))  # greedy: no randomness at all
    probabilities = filtered_probabilities(logits, temperature, top_k, top_p)
    return int(torch.multinomial(probabilities, 1, generator=generator))


@dataclass(frozen=True)
class Step:
    """One written token, and (when asked for) what the model thought of it."""

    id: int
    seconds: float  # the forward pass and the sampling for this token, nothing else
    probability: float | None = None  # the model's own odds for this token (plain softmax)
    top: tuple[tuple[int, float], ...] = ()  # its five likeliest tokens, with their odds


@torch.no_grad()
def generate_stream(
    model: GPT,
    ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_id: int | None = None,
    generator: torch.Generator | None = None,
    details: bool = False,
    cache: bool = False,
    graphs: bool = False,
) -> Iterator[Step]:
    """Yield new tokens one at a time until stop_id (not yielded) or max_new_tokens.

    Without a cache, each step re-reads the whole text so far (the last `context` tokens) to predict
    one more token. With a cache, the prompt is read once and each step feeds only the newest token.
    When the cache is full, it starts again from the last three quarters of the context, read in one
    pass, then continues one token at a time. (A cache can't simply slide: each stored key was computed
    while the oldest tokens were still visible.) So past `context` tokens the cached text can differ
    from the uncached text, which always reads exactly the last `context` tokens.

    With graphs (which implies the cache), every one-token step is a replay of a recorded CUDA graph
    (see graphs.py); reading a prompt, or re-reading when the cache is full, still runs normally.

    With details, each step also reports the model's own probabilities: a plain softmax of its
    scores, before temperature or top-p, so they show how sure the model itself was. They are
    computed after the step's clock stops, so they never count towards its speed.
    """
    model.eval()
    device = next(model.parameters()).device
    context = model.config.context
    tokens = list(ids)
    step = one_token_step(model) if graphs else None
    kv = step.cache if step is not None else model.new_cache() if cache else None
    pending = tokens[-context:]  # with a cache: the tokens it hasn't read yet
    for _ in range(max_new_tokens):
        started = time.perf_counter()
        if kv is None:
            window = torch.tensor([tokens[-context:]], device=device)
            logits = model(window)[0, -1]  # scores for the token after the last one
        else:
            if kv.length + len(pending) > context:  # full: start again from the last 3/4
                kv.rollback(0)
                pending = tokens[-(context * 3 // 4) :]
            if step is not None and len(pending) == 1:
                logits = step(pending[0])
            else:
                logits = model(torch.tensor([pending], device=device), kv)[0, -1]
        next_id = sample_next(logits, temperature, top_k, top_p, generator)  # int(): waits for the GPU
        seconds = time.perf_counter() - started
        if next_id == stop_id:
            return
        tokens.append(next_id)
        pending = [next_id]
        if not details:
            yield Step(next_id, seconds)
            continue
        probabilities = torch.softmax(logits.float(), dim=-1)
        values, top_ids = torch.topk(probabilities, 5)
        top = tuple((i, p) for i, p in zip(top_ids.tolist(), values.tolist()) if p > 0)
        yield Step(next_id, seconds, probabilities[next_id].item(), top)


def generate(
    model: GPT,
    ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_id: int | None = None,
    seed: int | None = None,
    cache: bool = False,
    graphs: bool = False,
) -> list[int]:
    """Extend ids until stop_id or max_new_tokens. Returns only the new ids."""
    generator = None
    if seed is not None:
        device = next(model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
    steps = generate_stream(model, ids, max_new_tokens, temperature, top_k, top_p, stop_id, generator,
                            cache=cache, graphs=graphs)
    return [step.id for step in steps]
