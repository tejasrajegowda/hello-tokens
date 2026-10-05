import copy

import pytest
import torch

from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT

SHAPE = dict(vocab_size=50, context=32, width=24, layers=2, heads=6)
SHAPES = {
    "v1": ModelConfig(**SHAPE),
    "v2": ModelConfig(**SHAPE, norm="rmsnorm", position="rope", feed_forward="swiglu", kv_heads=2),
}


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def pair(name):
    """Two copies of one model: ours, and the same weights with PyTorch's fused attention."""
    ours = GPT(SHAPES[name]).eval()
    return ours, copy.deepcopy(ours).use_fused_attention()


@pytest.mark.parametrize("name", SHAPES)
def test_fused_attention_matches_ours_on_a_whole_prompt(name):
    ours, fused = pair(name)
    ids = torch.randint(0, 50, (2, 20))
    with torch.no_grad():
        assert torch.allclose(fused(ids), ours(ids), atol=1e-5)


@pytest.mark.parametrize("name", SHAPES)
def test_fused_attention_matches_ours_one_token_at_a_time_with_a_cache(name):
    ours, fused = pair(name)
    ids = torch.randint(0, 50, (1, 20))
    a, b = ours.new_cache(), fused.new_cache()
    with torch.no_grad():
        ours(ids[:, :5], a), fused(ids[:, :5], b)
        for n in range(5, 20):  # one query against all stored keys: no mask
            assert torch.allclose(fused(ids[:, n : n + 1], b), ours(ids[:, n : n + 1], a), atol=1e-5)


@pytest.mark.parametrize("name", SHAPES)
def test_fused_attention_matches_ours_on_a_block_of_new_tokens(name):
    # The verify step of speculative decoding: 4 new tokens on top of 10 stored ones. This is where
    # PyTorch's built-in causal mask would be wrong (it lines up with the first key).
    ours, fused = pair(name)
    ids = torch.randint(0, 50, (1, 14))
    a, b = ours.new_cache(), fused.new_cache()
    with torch.no_grad():
        ours(ids[:, :10], a), fused(ids[:, :10], b)
        block_ours, block_fused = ours(ids[:, 10:], a), fused(ids[:, 10:], b)
        assert torch.allclose(block_fused, block_ours, atol=1e-5)
        # And both equal reading all 14 tokens from scratch.
        assert torch.allclose(block_ours, ours(ids)[:, 10:], atol=1e-5)


def test_fused_attention_is_causal():
    _, fused = pair("v2")
    ids = torch.randint(0, 50, (1, 12))
    changed = ids.clone()
    changed[0, 5] = (ids[0, 5] + 1) % 50
    with torch.no_grad():
        assert torch.allclose(fused(ids)[0, :5], fused(changed)[0, :5])


def test_the_switch_goes_both_ways():
    ours, fused = pair("v1")
    assert all(b.attention.fused for b in fused.blocks)
    assert not any(b.attention.fused for b in fused.use_fused_attention(False).blocks)
