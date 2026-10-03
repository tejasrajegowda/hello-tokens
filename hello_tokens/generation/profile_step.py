"""Where the time of one generation step goes.

A step is what `generate` does per new token: one forward pass over the last `context` tokens, then
sampling. Two measurements:

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

from hello_tokens.generation.sampling import sample_next
from hello_tokens.model.gpt import GPT


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
    model: GPT, contexts: list[int], rounds: int = 200, warmup: int = 20, profiled: int = 20, seed: int = 0
) -> list[StepProfile]:
    """Profile one generation step at each context length."""
    model.eval()
    device = next(model.parameters()).device
    on_gpu = device.type == "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)
    windows = {c: torch.randint(0, model.config.vocab_size, (1, c), device=device, generator=generator)
               for c in contexts}

    def step(ids: torch.Tensor) -> None:
        sample_next(model(ids)[0, -1], temperature=0.8, top_p=0.95, generator=generator)

    def sync() -> None:
        if on_gpu:
            torch.cuda.synchronize()

    for _ in range(warmup):
        for ids in windows.values():
            step(ids)
    sync()

    # Interleave the contexts round by round, so a slow patch (a background task, the CPU changing
    # speed) hits all of them alike instead of skewing one.
    times: dict[int, list[float]] = {c: [] for c in contexts}
    for _ in range(rounds):
        for c, ids in windows.items():
            started = time.perf_counter()
            step(ids)
            sync()
            times[c].append(time.perf_counter() - started)

    results = []
    for c, ids in windows.items():
        busy_ms, kernels = 0.0, 0.0
        if on_gpu:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(profiled):
                    step(ids)
                sync()
            gpu_events = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
            busy_ms = sum(e.device_time for e in gpu_events) / profiled / 1000  # microseconds -> ms
            kernels = len(gpu_events) / profiled
        quartiles = statistics.quantiles(times[c], n=4)
        results.append(StepProfile(c, statistics.median(times[c]) * 1000,
                                   (quartiles[2] - quartiles[0]) * 1000, busy_ms, kernels))
    return results
