"""Measure generation speed on a fixed workload.

Every variant (the KV cache, quantization, speculative decoding, ...) is measured here the same way,
so each one becomes a comparable row:

- The workload is fixed: greedy decoding (no randomness) and no stop token, so every run writes
  exactly the requested number of tokens. Normal writing stops when a story ends, which would let
  variants write different amounts.
- Three points, (prompt tokens, new tokens): a short prompt with a short and a long answer, and a
  long prompt. All stay within the 256-token context.
- Warm-up runs are discarded, the GPU is synchronized before the clock is read, and each number is
  the median of several runs.
- Every row names its full setup, so the effect of each change stays separate.
"""

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

from hello_tokens.generation.power import opt_out_of_power_throttling

POINTS = ((16, 64), (16, 224), (128, 128))

# A writer takes prompt ids and a number of new tokens, and returns exactly that many new ids.
Writer = Callable[[list[int], int], list[int]]


@dataclass(frozen=True)
class Setup:
    """What exactly was measured. Later lessons fill in the fields that are "none" today."""

    model: str  # checkpoint name, e.g. "v1"
    dtype: str  # "float32" or "bfloat16"
    cache: str = "none"
    graphs: str = "none"  # CUDA graphs or torch.compile
    quantization: str = "none"
    decoding: str = "normal"  # or "speculative"


@dataclass(frozen=True)
class Point:
    prompt_tokens: int
    new_tokens: int
    first_token_ms: float  # reading the prompt and writing one token
    per_token_ms: float  # each token after the first
    tokens_per_s: float  # new tokens divided by the total time
    spread_ms: float  # interquartile range of the total time
    identical_runs: bool  # every run wrote exactly the same tokens


@dataclass
class Result:
    row: str
    setup: Setup
    points: list[Point]
    model_mb: float
    peak_memory_mb: float  # 0 on the CPU
    perplexity: float | None = None
    machine: dict = field(default_factory=dict)
    measured_at: str = ""  # ISO date and time; rows are listed in this order


def _median_and_spread(seconds: list[float]) -> tuple[float, float]:
    quartiles = statistics.quantiles(seconds, n=4) if len(seconds) > 1 else [seconds[0]] * 3
    return statistics.median(seconds), quartiles[2] - quartiles[0]


def model_megabytes(model: torch.nn.Module) -> float:
    """Bytes of every weight and buffer, in MB (10^6 bytes)."""
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors) / 1e6


def measure_point(write: Writer, prompt: list[int], new_tokens: int, device: str,
                  repeats: int = 5, warmup: int = 1) -> Point:
    def timed(count: int) -> tuple[float, list[int]]:
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        out = write(prompt, count)
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - started, out

    for _ in range(warmup):
        timed(new_tokens)
    first = [timed(1)[0] for _ in range(repeats)]
    runs = [timed(new_tokens) for _ in range(repeats)]
    outputs = [out for _, out in runs]
    if any(len(out) != new_tokens for out in outputs):
        raise ValueError("the writer must return exactly the requested number of tokens")

    first_s, _ = _median_and_spread(first)
    total_s, spread_s = _median_and_spread([s for s, _ in runs])
    return Point(
        prompt_tokens=len(prompt),
        new_tokens=new_tokens,
        first_token_ms=first_s * 1000,
        # The total covers the first token too; the rest of the time is shared by the others.
        per_token_ms=(total_s - first_s) / (new_tokens - 1) * 1000,
        tokens_per_s=new_tokens / total_s,
        spread_ms=spread_s * 1000,
        identical_runs=all(out == outputs[0] for out in outputs),
    )


def machine_info() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "cpu": platform.processor()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def run_benchmark(row: str, setup: Setup, model: torch.nn.Module, write: Writer, device: str,
                  points=POINTS, repeats: int = 5, seed: int = 0) -> Result:
    """Measure every point. Prompts are fixed random tokens: with no stop token, speed doesn't
    depend on which tokens they are, only on how many."""
    opt_out_of_power_throttling()
    generator = torch.Generator().manual_seed(seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    measured = []
    for prompt_tokens, new_tokens in points:
        prompt = torch.randint(0, model.config.vocab_size, (prompt_tokens,), generator=generator).tolist()
        measured.append(measure_point(write, prompt, new_tokens, device, repeats))
    peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    return Result(row, setup, measured, model_megabytes(model), peak,
                  machine=machine_info(), measured_at=datetime.now().isoformat(timespec="seconds"))


def save_result(result: Result, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (result.row.replace(" ", "-") + ".json")
    path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return path


def load_results(directory: Path) -> list[dict]:
    """Every saved row, in the order it was measured (re-measuring a row moves it to the end)."""
    results = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("*.json")]
    return sorted(results, key=lambda r: r["measured_at"])


def format_table(results: list[dict]) -> str:
    """One line per row: tokens/s at each point, first-token and per-token time at the long answer."""
    names = [f"{p}+{n}" for p, n in POINTS]
    header = f"{'row':<20}" + "".join(f"{'tok/s ' + n:>14}" for n in names)
    header += f"{'first ms':>10}{'ms/token':>10}{'MB':>8}{'peak MB':>9}{'ppl':>7}"
    lines = [header]
    for r in results:
        points = {(p["prompt_tokens"], p["new_tokens"]): p for p in r["points"]}
        long = points.get(POINTS[1], r["points"][0])
        line = f"{r['row']:<20}" + "".join(
            f"{points[pt]['tokens_per_s']:>14.0f}" if pt in points else f"{'-':>14}" for pt in POINTS)
        ppl = f"{r['perplexity']:.3f}" if r.get("perplexity") else "-"
        line += (f"{long['first_token_ms']:>10.2f}{long['per_token_ms']:>10.2f}"
                 f"{r['model_mb']:>8.1f}{r['peak_memory_mb']:>9.0f}{ppl:>7}")
        lines.append(line)
    return "\n".join(lines)
