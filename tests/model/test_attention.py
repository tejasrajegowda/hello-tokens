import pytest
import torch

from hello_tokens.model.attention import CausalSelfAttention, causal_attention
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.embedding import Embedding

SMALL = ModelConfig(vocab_size=50, context=16, width=24, layers=1, heads=4)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def test_weights_are_zero_on_the_future_and_sum_to_one():
    q, k, v = (torch.randn(5, 8) for _ in range(3))
    _, weights = causal_attention(q, k, v)
    assert torch.all(torch.triu(weights, diagonal=1) == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5))


def test_equal_scores_give_the_average_of_the_past():
    # Worked by hand: if every query matches every key equally, position i averages values 0..i.
    q = k = torch.zeros(4, 8)
    v = torch.arange(4.0).unsqueeze(1).expand(4, 8)  # value i is all i's
    output, _ = causal_attention(q, k, v)
    assert torch.allclose(output[:, 0], torch.tensor([0.0, 0.5, 1.0, 1.5]))


def test_embedding_shape():
    ids = torch.randint(0, SMALL.vocab_size, (2, 10))
    assert Embedding(SMALL)(ids).shape == (2, 10, SMALL.width)


def test_too_long_a_sequence_is_rejected():
    with pytest.raises(ValueError):
        Embedding(SMALL)(torch.zeros(1, SMALL.context + 1, dtype=torch.long))


def test_attention_keeps_the_shape():
    x = torch.randn(2, 10, SMALL.width)
    assert CausalSelfAttention(SMALL)(x).shape == x.shape


def test_causality_changing_a_later_token_never_changes_an_earlier_output():
    embed, attend = Embedding(SMALL), CausalSelfAttention(SMALL)
    ids = torch.randint(0, SMALL.vocab_size, (1, 12))
    changed = ids.clone()
    changed[0, 7] = (ids[0, 7] + 1) % SMALL.vocab_size  # change token 7 only

    before, after = attend(embed(ids)), attend(embed(changed))
    assert torch.allclose(before[0, :7], after[0, :7])  # positions 0-6 can't see token 7
    assert not torch.allclose(before[0, 7:], after[0, 7:])  # positions 7+ can, and do change


def test_width_must_split_evenly_between_heads():
    with pytest.raises(ValueError):
        ModelConfig(width=10, heads=4)
