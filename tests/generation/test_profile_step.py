import pytest
import torch

from hello_tokens.generation import profile_step
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


@pytest.mark.parametrize("mode", ["full", "cache", "graphs"])
def test_every_mode_gets_a_sane_profile_on_the_cpu(mode):
    results = profile_steps(GPT(TINY), contexts=[4, 16], rounds=10, warmup=2, mode=mode)
    assert [r.context for r in results] == [4, 16]
    for r in results:
        assert r.wall_ms > 0 and r.spread_ms >= 0
        assert r.gpu_busy_ms == 0 and r.kernels == 0


@pytest.mark.parametrize("mode", ["cache", "graphs"])
def test_the_one_token_modes_need_a_token_in_the_cache(mode):
    with pytest.raises(ValueError):
        profile_steps(GPT(TINY), contexts=[1, 16], rounds=2, warmup=0, mode=mode)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        profile_steps(GPT(TINY), contexts=[4], rounds=2, warmup=0, mode="compiled")


def recorded_logits(monkeypatch, mode, contexts):
    """Profile with sampling replaced by a recorder; return the logits of every step, in order."""
    seen = []
    monkeypatch.setattr(profile_step, "sample_next", lambda logits, **_: seen.append(logits.clone()))
    profile_steps(GPT(TINY), contexts=contexts, rounds=5, warmup=2, mode=mode)
    return seen


@pytest.mark.parametrize("mode", ["cache", "graphs"])
@pytest.mark.parametrize("contexts", [[9], [3, 16]])
def test_every_step_runs_at_the_same_position(monkeypatch, mode, contexts):
    # Without the rollback after each step the position would move on, so the logits would change (and
    # the cache would soon overflow). Two interleaved contexts also exercise the refill between them.
    torch.manual_seed(0)
    seen = recorded_logits(monkeypatch, mode, contexts)
    n = len(contexts)
    assert len(seen) == n * (2 + 5)  # warm-up and timed rounds, one step per context in each
    for i in range(n):
        for logits in seen[i::n]:
            assert torch.allclose(logits, seen[i], atol=1e-6)


@pytest.mark.parametrize("mode", ["cache", "graphs"])
def test_the_one_token_modes_compute_what_the_full_step_computes(monkeypatch, mode):
    torch.manual_seed(0)
    full = recorded_logits(monkeypatch, "full", contexts=[3, 16])
    torch.manual_seed(0)
    one_token = recorded_logits(monkeypatch, mode, contexts=[3, 16])
    for c in range(2):
        assert torch.allclose(one_token[c], full[c], atol=1e-5)
