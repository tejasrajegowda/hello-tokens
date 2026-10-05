import pytest
import torch

from hello_tokens.generation.graphs import OneTokenStep, one_token_step
from hello_tokens.generation.sampling import generate
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT
from hello_tokens.quantization.weights import quantize_model

SHAPE = dict(vocab_size=50, context=32, width=24, layers=2, heads=4)
SHAPES = {
    "v1": ModelConfig(**SHAPE),
    "v2": ModelConfig(**SHAPE, norm="rmsnorm", position="rope", feed_forward="swiglu", kv_heads=2),
}
CASES = [(name, fused) for name in SHAPES for fused in (False, True)]


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def model_for(name, fused=False):
    return GPT(SHAPES[name]).eval().use_fused_attention(fused)


@pytest.mark.parametrize("name,fused", CASES)
def test_the_fixed_shape_step_equals_the_cached_step_at_every_position(name, fused):
    model = model_for(name, fused)
    ids = torch.randint(0, 50, (1, 32))
    normal, fixed = model.new_cache(), model.new_cache()
    with torch.no_grad():
        model(ids[:, :4], normal)
        model(ids[:, :4], fixed)
        for n in range(4, 32):  # up to the last slot of the context
            expected = model(ids[:, n : n + 1], normal)[0, -1]
            got = model.decode_step(ids[:, n : n + 1], torch.tensor([n]), fixed)[0]
            fixed.advance(1)
            assert torch.allclose(got, expected, atol=1e-5), f"position {n}"
    # Both caches now hold the same keys and values.
    assert torch.allclose(normal.keys, fixed.keys, atol=1e-6)
    assert torch.allclose(normal.values, fixed.values, atol=1e-6)


def test_the_fixed_shape_step_leaves_the_pointer_to_the_caller():
    model = model_for("v2")
    cache = model.new_cache()
    with torch.no_grad():
        model.decode_step(torch.tensor([[3]]), torch.tensor([0]), cache)
    assert cache.length == 0


def test_stale_entries_after_a_rollback_are_ignored():
    # Speculative decoding and the 256 restart leave old keys beyond the pointer; the mask must hide them.
    model = model_for("v2")
    ids, other = torch.randint(0, 50, (1, 12)), torch.randint(0, 50, (1, 1))
    with torch.no_grad():
        used = model.new_cache()
        model(ids, used)
        used.rollback(5)
        got = model.decode_step(other, torch.tensor([5]), used)[0]
        fresh = model.new_cache()
        model(ids[:, :5], fresh)
        assert torch.allclose(got, model(other, fresh)[0, -1], atol=1e-5)


@pytest.mark.parametrize("name,fused", CASES)
def test_writing_with_the_recorded_step_gives_the_same_text(name, fused):
    model = model_for(name, fused)
    prompt = torch.randint(0, 50, (5,)).tolist()
    # 80 tokens with a context of 32: the cache fills and restarts from the last 3/4 twice.
    cached = generate(model, prompt, 80, temperature=0, cache=True)
    graphed = generate(model, prompt, 80, temperature=0, graphs=True)
    assert graphed == cached
    sampled = generate(model, prompt, 40, temperature=1.0, top_p=0.9, seed=7, cache=True)
    assert generate(model, prompt, 40, temperature=1.0, top_p=0.9, seed=7, graphs=True) == sampled


def test_a_one_token_prompt_starts_with_the_recorded_step():
    model = model_for("v1")
    assert generate(model, [7], 10, temperature=0, graphs=True) == generate(model, [7], 10, temperature=0, cache=True)


def test_quantized_weights_work_in_the_fixed_shape_step():
    model = GPT(ModelConfig(vocab_size=50, context=32, width=64, layers=2, heads=4)).eval()
    quantize_model(model, 8)
    prompt = torch.randint(0, 50, (5,)).tolist()
    assert generate(model, prompt, 30, temperature=0, graphs=True) == generate(model, prompt, 30, temperature=0, cache=True)


def test_the_step_is_made_once_and_reused_until_the_attention_kernel_changes():
    model = model_for("v1")
    first = one_token_step(model)
    first(3)
    again = one_token_step(model)
    assert again is first and again.cache.length == 0  # reused, and emptied for the new story
    model.use_fused_attention(True)
    assert one_token_step(model) is not first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("name,fused", CASES)
def test_a_replayed_graph_equals_running_the_step_normally(name, fused):
    torch.backends.cuda.matmul.allow_tf32 = False  # exact comparisons in fp32 (W13)
    model = model_for(name, fused).cuda()
    step = OneTokenStep(model)
    assert step.graph is not None
    ids = torch.randint(0, 50, (20,)).tolist()
    plain = model.new_cache()
    with torch.no_grad():
        model(torch.tensor([ids[:4]], device="cuda"), step.cache)
        model(torch.tensor([ids[:4]], device="cuda"), plain)
        for token in ids[4:]:
            got = step(token).clone()
            expected = model(torch.tensor([[token]], device="cuda"), plain)[0, -1]
            assert torch.allclose(got, expected, atol=1e-5)
    prompt = ids[:5]
    assert generate(model, prompt, 80, temperature=0, graphs=True) == generate(model, prompt, 80, temperature=0, cache=True)
