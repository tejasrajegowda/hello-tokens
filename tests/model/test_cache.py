import pytest
import torch

from hello_tokens.generation.sampling import generate
from hello_tokens.model.cache import KVCache
from hello_tokens.model.config import PRESETS, ModelConfig
from hello_tokens.model.gpt import GPT

SHAPE = dict(vocab_size=50, context=32, width=24, layers=2, heads=4)
SHAPES = {
    "v1": ModelConfig(**SHAPE),
    "v2": ModelConfig(**SHAPE, norm="rmsnorm", position="rope", feed_forward="swiglu", kv_heads=2),
}


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def model_for(name):
    return GPT(SHAPES[name]).eval()


@pytest.mark.parametrize("name", SHAPES)
def test_one_token_at_a_time_with_a_cache_matches_reading_everything_again(name):
    model = model_for(name)
    ids = torch.randint(0, 50, (1, 30))
    cache = model.new_cache()
    with torch.no_grad():
        model(ids[:, :5], cache)  # read the prompt once
        for n in range(5, 30):
            cached = model(ids[:, n : n + 1], cache)[0, -1]  # feed one token
            full = model(ids[:, : n + 1])[0, -1]  # read all n + 1 tokens from scratch
            assert torch.allclose(cached, full, atol=1e-5), f"position {n}"
    assert cache.length == 30


@pytest.mark.parametrize("name", SHAPES)
def test_reading_a_whole_prompt_into_the_cache_gives_the_usual_outputs(name):
    model = model_for(name)
    ids = torch.randint(0, 50, (2, 20))
    with torch.no_grad():
        assert torch.allclose(model(ids, model.new_cache(batch=2)), model(ids), atol=1e-5)


@pytest.mark.parametrize("name", SHAPES)
def test_rolling_back_forgets_exactly_the_dropped_tokens(name):
    model = model_for(name)
    ids, other = torch.randint(0, 50, (1, 10)), torch.randint(0, 50, (1, 3))
    with torch.no_grad():
        cache = model.new_cache()
        model(ids, cache)
        cache.rollback(6)  # as if tokens 6-9 were drafted and rejected
        after_rollback = model(other, cache)
        fresh = model.new_cache()
        model(ids[:, :6], fresh)
        assert torch.allclose(after_rollback, model(other, fresh), atol=1e-6)


def reference_with_the_cache_s_window(model, ids, steps):
    """Greedy writing without a cache, but reading the same window the cached version reads."""
    context, keep = model.config.context, model.config.context * 3 // 4
    tokens, start = list(ids), max(0, len(ids) - context)
    for _ in range(steps):
        if len(tokens) - start > context:
            start = len(tokens) - keep  # the cache was full: it re-read the last 3/4
        logits = model(torch.tensor([tokens[start:]]))[0, -1]
        tokens.append(int(torch.argmax(logits)))
    return tokens[len(ids) :]


@pytest.mark.parametrize("name", SHAPES)
def test_greedy_writing_with_a_cache(name):
    model = model_for(name)
    prompt = torch.randint(0, 50, (5,)).tolist()
    with torch.no_grad():
        # Within the context: exactly the same tokens as writing without a cache.
        assert generate(model, prompt, 20, temperature=0, cache=True) == generate(model, prompt, 20, temperature=0)
        # Past the context (5 + 70 > 32): the cache restarts from the last 3/4, and the tokens match
        # an uncached reference that reads the same windows.
        assert generate(model, prompt, 70, temperature=0, cache=True) == reference_with_the_cache_s_window(
            model, prompt, 70)


def test_a_prompt_longer_than_the_context_still_works():
    model = model_for("v2")
    prompt = torch.randint(0, 50, (40,)).tolist()  # 40 > 32: only the last 32 are read
    with torch.no_grad():
        assert generate(model, prompt, 5, temperature=0, cache=True) == generate(model, prompt, 5, temperature=0)


def test_gqa_makes_the_cache_three_times_smaller():
    v1 = KVCache(PRESETS["v1"], dtype=torch.bfloat16)
    v2 = KVCache(PRESETS["v2"], dtype=torch.bfloat16)
    # keys + values, 8 layers, (6 or 2) heads x 64 numbers, 2 bytes each, per position
    assert v1.bytes() // 256 == 2 * 8 * 6 * 64 * 2 == 12_288
    assert v2.bytes() // 256 == 2 * 8 * 2 * 64 * 2 == 4_096


def test_the_cache_refuses_to_overflow_or_roll_forward():
    model = model_for("v1")
    cache = model.new_cache()
    with torch.no_grad():
        model(torch.randint(0, 50, (1, 30)), cache)
        with pytest.raises(ValueError):
            model(torch.randint(0, 50, (1, 3)), cache)  # 30 + 3 > 32
    with pytest.raises(ValueError):
        cache.rollback(31)
