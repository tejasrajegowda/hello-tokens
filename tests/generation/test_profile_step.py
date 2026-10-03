import pytest
import torch

from hello_tokens.generation.profile_step import profile_steps
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT

TINY = ModelConfig(vocab_size=64, context=16, width=32, layers=2, heads=4)


def test_every_context_gets_a_sane_profile_on_the_cpu():
    results = profile_steps(GPT(TINY), contexts=[4, 16], rounds=10, warmup=2)
    assert [r.context for r in results] == [4, 16]
    for r in results:
        assert r.wall_ms > 0 and r.spread_ms >= 0
        assert r.gpu_busy_ms == 0 and r.kernels == 0  # no GPU, nothing to count


def test_the_gpu_is_profiled_when_there_is_one():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU on this machine")
    (r,) = profile_steps(GPT(TINY).to("cuda"), contexts=[16], rounds=10, warmup=2, profiled=5)
    assert r.kernels > 0
    assert 0 < r.gpu_busy_ms < r.wall_ms * 1.5  # profiling adds a little; it can't be wildly more
    assert 0 <= r.gpu_idle_fraction < 1
