import copy

import pytest
import torch
from torch import nn

from hello_tokens.benchmark.harness import model_megabytes
from hello_tokens.model.config import PRESETS, ModelConfig
from hello_tokens.model.gpt import GPT
from hello_tokens.quantization.weights import (
    QuantizedLinear, dequantize_int4, dequantize_int8, pack_int4, quantize_int4, quantize_int8, quantize_model,
    unpack_int4,
)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def test_int8_is_off_by_at_most_half_a_step_per_weight():
    w = torch.randn(48, 128)
    values, scale = quantize_int8(w)
    assert values.dtype == torch.int8 and values.abs().max() == 127  # the largest weight uses the top level
    assert ((dequantize_int8(values, scale) - w).abs() <= scale / 2 + 1e-7).all()


def test_int4_packs_two_values_per_byte_and_unpacks_exactly():
    values = torch.randint(-7, 8, (5, 128), dtype=torch.int8)
    packed = pack_int4(values)
    assert packed.dtype == torch.uint8 and packed.shape == (5, 64)
    assert torch.equal(unpack_int4(packed), values)
    # Worked by hand: 3 and -2 become 11 and 6, so the byte is 6 * 16 + 11 = 107.
    assert pack_int4(torch.tensor([3, -2], dtype=torch.int8)).item() == 107


def test_int4_is_off_by_at_most_half_a_step_within_each_group():
    w = torch.randn(16, 256)
    packed, scale = quantize_int4(w)
    assert scale.shape == (16, 4, 1)  # 256 / 64 groups per row
    error = (dequantize_int4(packed, scale) - w).reshape(16, 4, 64).abs()
    assert (error <= scale / 2 + 1e-7).all()


def test_zero_stays_exactly_zero():
    w = torch.zeros(4, 64)
    w[0, 0] = 1.0
    assert torch.equal(dequantize_int8(*quantize_int8(w))[1:], w[1:])
    assert torch.equal(dequantize_int4(*quantize_int4(w))[1:], w[1:])


@pytest.mark.parametrize("bits", [8, 4])
def test_a_quantized_layer_computes_with_its_restored_weights(bits):
    linear = nn.Linear(128, 32)
    q = QuantizedLinear(linear, bits)
    x = torch.randn(3, 128)
    assert torch.allclose(q(x), x @ q.weight().T + linear.bias, atol=1e-6)


V2_SMALL = ModelConfig(vocab_size=50, context=32, width=64, layers=2, heads=4, norm="rmsnorm", position="rope",
                       feed_forward="swiglu", kv_heads=2)


@pytest.mark.parametrize("bits, closeness", [(8, 0.999), (4, 0.95)])
def test_a_quantized_model_still_gives_nearly_the_same_scores(bits, closeness):
    model = GPT(V2_SMALL).eval()
    quantized = quantize_model(copy.deepcopy(model), bits)
    ids = torch.randint(0, 50, (2, 20))
    with torch.no_grad():
        a, b = model(ids).flatten(), quantized(ids).flatten()
    assert torch.cosine_similarity(a, b, dim=0) > closeness


def test_every_layer_inside_the_blocks_is_quantized_and_the_word_table_is_not():
    model = quantize_model(GPT(V2_SMALL), 4)
    linears = [m for m in model.blocks.modules() if isinstance(m, nn.Linear)]
    assert linears == []
    assert model.output.weight is model.embedding.token.weight  # still shared, still full precision


def test_the_v2_model_shrinks_as_planned():
    # bf16 v2: 14,189,952 parameters x 2 bytes = 28.38 MB, plus RoPE's rotation tables (8 layers x 2 x
    # 256 x 32 x 2 bytes = 0.26 MB) = 28.64 MB. The layers hold 12,582,912 weights (25.17 MB in bf16).
    full = GPT(PRESETS["v2"]).to(torch.bfloat16)
    layers = 12_582_912
    size = model_megabytes(full)
    assert size == pytest.approx(28.642, abs=0.001)
    # int8: one byte per weight, plus one bf16 scale per row: ~0.56x.
    assert model_megabytes(quantize_model(copy.deepcopy(full), 8)) / size == pytest.approx(0.56, abs=0.01)
    # int4: half a byte per weight, plus one bf16 scale per 64 weights: exactly this many bytes.
    int4 = size - layers * 2 / 1e6 + layers / 2 / 1e6 + layers / 64 * 2 / 1e6
    assert model_megabytes(quantize_model(copy.deepcopy(full), 4)) == pytest.approx(int4, abs=1e-6)
    assert int4 / size == pytest.approx(0.355, abs=0.005)


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        quantize_model(GPT(V2_SMALL), 2)
    with pytest.raises(ValueError):
        quantize_int4(torch.randn(4, 100))  # 100 is not a multiple of 64
