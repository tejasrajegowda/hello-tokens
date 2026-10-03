import pytest
import torch

from hello_tokens.model.attention import CausalSelfAttention
from hello_tokens.model.block import FeedForward, SwiGLU, swiglu_hidden
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT

V2 = ModelConfig(norm="rmsnorm", position="rope", feed_forward="swiglu", kv_heads=2)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def weights(module):
    return sum(p.numel() for name, p in module.named_parameters() if name.endswith("weight"))


def test_swiglu_holds_exactly_as_many_weights_as_the_gelu_network():
    config = ModelConfig()
    assert swiglu_hidden(384) == 1024
    assert weights(SwiGLU(config)) == weights(FeedForward(config)) == 1_179_648


def test_swiglu_keeps_the_shape():
    config = ModelConfig(vocab_size=50, context=16, width=24, layers=1, heads=4, feed_forward="swiglu")
    x = torch.randn(2, 10, 24)
    assert SwiGLU(config)(x).shape == x.shape


def test_the_full_v2_shape_has_the_planned_parameter_count():
    # v1's 15,867,648, minus GQA's narrower key/value layers (1,576,960), RoPE's position table
    # (98,304) and RMSNorm's biases (6,528), plus SwiGLU's extra bias (4,096).
    assert GPT(V2).parameter_count() == 15_867_648 - 1_576_960 - 98_304 - 6_528 + 4_096 == 14_189_952


def test_gqa_is_ordinary_attention_with_each_key_value_head_shared_by_its_group():
    # Build a GQA layer (4 query heads, 2 key/value heads), then an ordinary multi-head layer whose
    # key/value weights are the GQA ones copied once per query head in the group. Same outputs.
    gqa_config = ModelConfig(vocab_size=50, context=16, width=32, layers=1, heads=4, kv_heads=2)
    mha_config = ModelConfig(vocab_size=50, context=16, width=32, layers=1, heads=4)
    gqa, mha = CausalSelfAttention(gqa_config), CausalSelfAttention(mha_config)

    width, head_width, group = 32, 8, 2
    q_w, k_w, v_w = gqa.qkv.weight.split([width, 2 * head_width, 2 * head_width])
    q_b, k_b, v_b = gqa.qkv.bias.split([width, 2 * head_width, 2 * head_width])

    def per_query_head(t):  # (2 heads * 8, ...) -> (4 heads * 8, ...), kv head 0 twice, then kv head 1 twice
        return t.reshape(2, head_width, *t.shape[1:]).repeat_interleave(group, dim=0).reshape(-1, *t.shape[1:])

    with torch.no_grad():
        mha.qkv.weight.copy_(torch.cat([q_w, per_query_head(k_w), per_query_head(v_w)]))
        mha.qkv.bias.copy_(torch.cat([q_b, per_query_head(k_b), per_query_head(v_b)]))
        mha.out.load_state_dict(gqa.out.state_dict())
    x = torch.randn(2, 10, width)
    assert torch.allclose(gqa(x), mha(x), atol=1e-6)


def test_kv_heads_equal_to_heads_is_exactly_v1_attention():
    config = ModelConfig(vocab_size=50, context=16, width=24, layers=1, heads=4)
    explicit = ModelConfig(vocab_size=50, context=16, width=24, layers=1, heads=4, kv_heads=4)
    v1, same = CausalSelfAttention(config), CausalSelfAttention(explicit)
    same.load_state_dict(v1.state_dict())  # identical shapes: the layer is still width * 3 wide
    x = torch.randn(1, 8, 24)
    assert torch.equal(v1(x), same(x))


def test_the_v2_model_is_causal():
    config = ModelConfig(vocab_size=50, context=16, width=24, layers=2, heads=6, norm="rmsnorm",
                         position="rope", feed_forward="swiglu", kv_heads=2)
    model = GPT(config)
    ids = torch.randint(0, 50, (1, 12))
    changed = ids.clone()
    changed[0, 5] = (ids[0, 5] + 1) % 50
    assert torch.allclose(model(ids)[0, :5], model(changed)[0, :5])


def test_each_preset_adds_one_change_to_the_one_before():
    from dataclasses import asdict

    from hello_tokens.model.config import PRESETS

    names = list(PRESETS)
    assert names == ["v1", "rmsnorm", "rope", "swiglu", "v2"]
    assert PRESETS["v1"] == ModelConfig() and PRESETS["v2"] == V2
    for before, after in zip(names, names[1:]):
        a, b = asdict(PRESETS[before]), asdict(PRESETS[after])
        assert sum(a[k] != b[k] for k in a) == 1  # exactly one field differs


def test_bad_switch_values_are_rejected():
    with pytest.raises(ValueError):
        ModelConfig(feed_forward="relu")
    with pytest.raises(ValueError):
        ModelConfig(heads=6, kv_heads=4)  # 6 query heads can't be split into 4 equal groups
