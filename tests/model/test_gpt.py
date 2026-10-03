import math

import pytest
import torch
import torch.nn.functional as F

from hello_tokens.model.block import Block
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT

SMALL = ModelConfig(vocab_size=50, context=16, width=24, layers=2, heads=4)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def test_the_full_size_model_has_the_planned_parameter_count():
    # Worked out by hand, per block (width w = 384):
    #   attention   qkv w*3w + 3w   +   out w*w + w
    #   feed-fwd    up  w*4w + 4w   +   down 4w*w + w
    #   two norms   2 * 2w
    w, config = 384, ModelConfig()
    block = (w * 3 * w + 3 * w) + (w * w + w) + (w * 4 * w + 4 * w) + (4 * w * w + w) + 2 * 2 * w
    embeddings = config.vocab_size * w + config.context * w  # tokens + positions
    final_norm = 2 * w
    expected = config.layers * block + embeddings + final_norm  # output layer is tied: no extra
    assert expected == 15_867_648
    assert GPT(config).parameter_count() == expected


def test_output_layer_shares_the_token_table():
    model = GPT(SMALL)
    assert model.output.weight is model.embedding.token.weight


def test_shapes_end_to_end():
    ids = torch.randint(0, SMALL.vocab_size, (3, 10))
    assert GPT(SMALL)(ids).shape == (3, 10, SMALL.vocab_size)


def test_a_block_keeps_the_shape():
    x = torch.randn(2, 10, SMALL.width)
    assert Block(SMALL)(x).shape == x.shape


def test_the_whole_model_is_causal():
    model = GPT(SMALL)
    ids = torch.randint(0, SMALL.vocab_size, (1, 12))
    changed = ids.clone()
    changed[0, 5] = (ids[0, 5] + 1) % SMALL.vocab_size
    assert torch.allclose(model(ids)[0, :5], model(changed)[0, :5])


def test_an_untrained_model_guesses_uniformly():
    # Before training, the model should spread its bets evenly over the vocabulary, so its loss is
    # about ln(vocab_size): the loss of picking uniformly at random. Far above means a bad start.
    config = ModelConfig(vocab_size=4096, context=64, width=64, layers=2, heads=4)
    ids = torch.randint(0, config.vocab_size, (4, 64))
    # Targets the model cannot see. (Using ids themselves would score better than chance: with
    # tied weights an untrained model already leans towards repeating the token it was given.)
    targets = torch.randint(0, config.vocab_size, (4, 64))
    logits = GPT(config)(ids)
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
    assert abs(loss.item() - math.log(4096)) < 0.1  # ln(4096) = 8.318
