from pathlib import Path

import pytest
import torch

from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT
from hello_tokens.model.norm import RMSNorm
from hello_tokens.model.rope import RotaryPositions

GOLDEN = Path(__file__).parent / "fixtures" / "v1_golden.torch"
MODERN = ModelConfig(vocab_size=50, context=16, width=24, layers=2, heads=4, norm="rmsnorm", position="rope")


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def test_a_v1_model_still_loads_and_gives_identical_outputs():
    # Weights and outputs recorded with the v1 code (tests/model/fixtures/make_v1_golden.py).
    golden = torch.load(GOLDEN)
    model = GPT(ModelConfig(**golden["model_config"])).eval()
    model.load_state_dict(golden["model"])  # strict: any renamed or extra saved tensor fails here
    with torch.no_grad():
        assert torch.allclose(model(golden["ids"]), golden["logits"], atol=1e-6, rtol=0)


def test_rmsnorm_by_hand():
    # [3, 4]: mean of squares = (9 + 16) / 2 = 12.5, root = 3.5355, so [3, 4] / 3.5355 = [0.8485, 1.1314].
    out = RMSNorm(2, eps=0.0)(torch.tensor([3.0, 4.0]))
    assert torch.allclose(out, torch.tensor([0.8485, 1.1314]), atol=1e-4)


def test_rope_leaves_position_zero_unchanged_and_never_changes_a_vector_s_length():
    rope = RotaryPositions(head_width=8, context=16)
    x = torch.randn(16, 8)
    rotated = rope(x)
    assert torch.allclose(rotated[0], x[0])  # angle 0: no rotation
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)  # rotations keep length


def test_rope_scores_depend_only_on_the_distance_between_positions():
    rope = RotaryPositions(head_width=8, context=16)
    q, k = torch.randn(8), torch.randn(8)
    rq, rk = rope(q.expand(16, 8)), rope(k.expand(16, 8))  # the same q and k at every position
    # Query at 3, key at 1 (distance 2) scores the same as query at 10, key at 8 (distance 2) ...
    assert torch.allclose(rq[3] @ rk[1], rq[10] @ rk[8], atol=1e-5)
    # ... but differently from distance 5.
    assert not torch.allclose(rq[3] @ rk[1], rq[6] @ rk[1], atol=1e-3)


def test_rope_with_an_offset_matches_the_same_positions_in_a_longer_sequence():
    rope = RotaryPositions(head_width=8, context=16)
    x = torch.randn(16, 8)
    assert torch.allclose(rope(x[5:9], offset=5), rope(x)[5:9])  # what the KV cache will rely on


def test_the_modern_model_is_causal_and_has_no_position_table():
    model = GPT(MODERN)
    ids = torch.randint(0, MODERN.vocab_size, (1, 12))
    changed = ids.clone()
    changed[0, 5] = (ids[0, 5] + 1) % MODERN.vocab_size
    assert torch.allclose(model(ids)[0, :5], model(changed)[0, :5])
    assert model.embedding.position is None
    assert not any("rope" in name for name in model.state_dict())  # rotation tables aren't saved


def test_rmsnorm_and_rope_remove_exactly_the_position_table_and_the_norm_biases():
    v1 = GPT(ModelConfig()).parameter_count()
    modern = GPT(ModelConfig(norm="rmsnorm", position="rope")).parameter_count()
    position_table = 256 * 384
    norm_biases = (2 * 8 + 1) * 384  # two norms per block plus the final one
    assert v1 - modern == position_table + norm_biases == 104_832


def test_unknown_switch_values_are_rejected():
    with pytest.raises(ValueError):
        ModelConfig(norm="batchnorm")
    with pytest.raises(ValueError):
        ModelConfig(position="sinusoidal")
