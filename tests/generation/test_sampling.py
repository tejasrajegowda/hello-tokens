import math

import pytest
import torch

from hello_tokens.generation.sampling import generate, sample_next
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT

# Probabilities 0.5, 0.3, 0.15, 0.05 for tokens 0..3, written as scores (log-probabilities).
LOGITS = torch.tensor([math.log(p) for p in (0.5, 0.3, 0.15, 0.05)])
TINY = ModelConfig(vocab_size=32, context=8, width=16, layers=1, heads=2)


def draws(n=2000, **dials):
    g = torch.Generator().manual_seed(0)
    return [sample_next(LOGITS, generator=g, **dials) for _ in range(n)]


def test_temperature_zero_always_takes_the_top_token():
    assert set(draws(n=50, temperature=0)) == {0}


def test_plain_sampling_follows_the_probabilities():
    counts = torch.bincount(torch.tensor(draws()), minlength=4) / 2000
    assert torch.allclose(counts, torch.tensor([0.5, 0.3, 0.15, 0.05]), atol=0.03)


def test_low_temperature_sharpens_toward_the_top():
    assert draws(temperature=0.3).count(0) > draws(temperature=1.0).count(0)


def test_top_k_never_picks_outside_the_k_best():
    assert set(draws(top_k=2)) == {0, 1}


def test_top_p_keeps_the_smallest_set_reaching_p():
    # 0.5 alone is short of 0.75; 0.5 + 0.3 = 0.8 reaches it, so tokens 0 and 1 survive.
    assert set(draws(top_p=0.75)) == {0, 1}
    # Even a tiny p keeps the single best token.
    assert set(draws(n=50, top_p=0.01)) == {0}


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(TINY)


def test_generation_respects_the_length_limit(model):
    assert len(generate(model, [1, 2, 3], max_new_tokens=5, temperature=1.0, seed=0)) == 5


def test_generation_stops_at_the_stop_token(model):
    first = generate(model, [1, 2, 3], max_new_tokens=1, temperature=0)[0]
    assert generate(model, [1, 2, 3], max_new_tokens=5, temperature=0, stop_id=first) == []


def test_text_longer_than_the_context_is_handled(model):
    # 20 tokens of prompt with a context of 8: only the last 8 are read each step.
    assert len(generate(model, list(range(20)), max_new_tokens=3, temperature=1.0, seed=0)) == 3


def test_the_same_seed_writes_the_same_text(model):
    a = generate(model, [1, 2], max_new_tokens=10, temperature=1.0, seed=42)
    b = generate(model, [1, 2], max_new_tokens=10, temperature=1.0, seed=42)
    assert a == b
