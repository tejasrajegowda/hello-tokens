import numpy as np
import pytest
import torch

from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT
from hello_tokens.training.batches import get_batch
from hello_tokens.training.optimize import learning_rate_at, make_optimizer, next_token_loss, train_step

TINY = ModelConfig(vocab_size=64, context=16, width=32, layers=2, heads=4)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def test_targets_are_the_inputs_shifted_by_one():
    tokens = np.arange(100, dtype=np.uint16)  # 0, 1, 2, ... so every window is easy to read
    inputs, targets = get_batch(tokens, batch_size=3, context=8, generator=np.random.default_rng(0))
    assert inputs.shape == targets.shape == (3, 8)
    assert torch.equal(targets, inputs + 1)  # each target is the next token


def test_the_schedule_warms_up_then_decays_to_the_floor():
    rates = [learning_rate_at(s, peak=1.0, warmup=10, total=100) for s in range(101)]
    assert rates[0] == pytest.approx(0.1)  # 1/10 of the way into warm-up
    assert rates[9] == pytest.approx(1.0)  # warm-up complete: the peak
    assert all(a >= b for a, b in zip(rates[9:], rates[10:]))  # never rises after the peak
    assert rates[100] == pytest.approx(0.1)  # settles at 10% of the peak


def test_weight_decay_skips_biases_and_norms():
    optimizer = make_optimizer(GPT(TINY), learning_rate=1e-3)
    decayed, plain = optimizer.param_groups
    assert all(p.dim() >= 2 for p in decayed["params"]) and decayed["weight_decay"] > 0
    assert all(p.dim() < 2 for p in plain["params"]) and plain["weight_decay"] == 0


def test_the_model_can_memorise_one_batch():
    # The proof that learning works end to end: on one fixed batch, repeated, the loss must
    # fall from about ln(64) = 4.16 to almost nothing. If it can't do this, it can't learn at all.
    model = GPT(TINY)
    optimizer = make_optimizer(model, learning_rate=3e-3)
    tokens = np.random.default_rng(0).integers(0, TINY.vocab_size, 1000).astype(np.uint16)
    inputs, targets = get_batch(tokens, 4, TINY.context, np.random.default_rng(1))

    first = next_token_loss(model(inputs), targets).item()
    for _ in range(150):
        last = train_step(model, optimizer, inputs, targets, learning_rate=3e-3)
    assert first > 3.5
    assert last < 0.1
