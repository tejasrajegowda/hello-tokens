import copy

import pytest
import torch

from hello_tokens.generation.sampling import generate
from hello_tokens.generation.speculative import speculative_stream
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT

TARGETS = {
    "v1": ModelConfig(vocab_size=50, context=32, width=24, layers=2, heads=4),
    "v2": ModelConfig(vocab_size=50, context=32, width=24, layers=2, heads=4, norm="rmsnorm", position="rope",
                      feed_forward="swiglu", kv_heads=2),
}
DRAFT = ModelConfig(vocab_size=50, context=32, width=16, layers=1, heads=2)


def models(target_name, vocab=50):
    torch.manual_seed(0)
    target = GPT(ModelConfig(**{**TARGETS[target_name].__dict__, "vocab_size": vocab})).eval()
    torch.manual_seed(1)  # a different, worse model: it will guess wrong often
    draft = GPT(ModelConfig(**{**DRAFT.__dict__, "vocab_size": vocab})).eval()
    return target, draft


def speculative(target, draft, prompt, n, **kwargs):
    return [step.id for step in speculative_stream(target, draft, prompt, n, **kwargs)]


@pytest.mark.parametrize("name", TARGETS)
@pytest.mark.parametrize("k", [1, 4])
def test_greedy_speculative_writing_is_exactly_greedy_writing(name, k):
    target, draft = models(name)
    prompt = [3, 1, 4, 1, 5]
    stats = {}
    with torch.no_grad():
        expected = generate(target, prompt, 25, temperature=0)  # 5 + 25 stays within the context
        assert speculative(target, draft, prompt, 25, temperature=0, k=k, stats=stats) == expected
    assert stats["drafted"] == k * stats["rounds"] and 0 <= stats["accepted"] <= stats["drafted"]


def test_a_draft_identical_to_the_target_is_always_accepted():
    target, _ = models("v2")
    stats = {}
    out = speculative(target, copy.deepcopy(target), [1, 2], 20, temperature=1.0, k=4, stats=stats,
                      generator=torch.Generator().manual_seed(0))
    assert len(out) == 20 and stats["accepted"] == stats["drafted"]  # p/q = 1: nothing rejected


def test_it_writes_past_the_context_and_respects_the_limits():
    target, draft = models("v2")
    assert len(speculative(target, draft, [1, 2], 80, temperature=0)) == 80  # 2 + 80 > 32
    first = speculative(target, draft, [1, 2], 1, temperature=0)[0]
    assert speculative(target, draft, [1, 2], 10, temperature=0, stop_id=first) == []


def test_sampled_tokens_follow_the_target_s_distribution():
    # The guarantee: whatever the draft guesses, each token is distributed exactly as the target's.
    # Check the first two tokens' frequencies against the target's exact probabilities (vocab 6).
    vocab, samples = 6, 4000
    target, draft = models("v2", vocab)
    prompt = [1, 2]
    with torch.no_grad():
        first = torch.softmax(target(torch.tensor([prompt]))[0, -1], -1)
        second = sum(first[t] * torch.softmax(target(torch.tensor([prompt + [t]]))[0, -1], -1)
                     for t in range(vocab))
    g = torch.Generator().manual_seed(0)
    draws = torch.tensor([speculative(target, draft, prompt, 2, temperature=1.0, k=3, generator=g)
                          for _ in range(samples)])
    for position, expected in enumerate((first, second)):
        observed = torch.bincount(draws[:, position], minlength=vocab) / samples
        assert torch.allclose(observed, expected, atol=0.03), (position, observed, expected)
