import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hello_tokens.evaluation.perplexity import perplexity
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT

TINY = ModelConfig(vocab_size=64, context=16, width=32, layers=2, heads=4)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(TINY)


def random_tokens(count, seed=0):
    return np.random.default_rng(seed).integers(0, TINY.vocab_size, count).astype(np.uint16)


def test_every_token_after_the_first_is_scored_exactly_once(model):
    # 1,000 tokens: 62 full windows of 16 predictions, then a short window of 7.
    assert perplexity(model, random_tokens(1000)).tokens == 999


def test_a_model_that_knows_nothing_scores_the_vocabulary_size(model):
    # All weights zero: every logit is 0, so every token gets probability 1/64.
    for p in model.parameters():
        p.data.zero_()
    score = perplexity(model, random_tokens(500))
    assert score.loss == pytest.approx(math.log(TINY.vocab_size))
    assert score.perplexity == pytest.approx(TINY.vocab_size)


def test_the_batch_size_does_not_change_the_answer(model):
    tokens = random_tokens(1000)
    one = perplexity(model, tokens, batch_size=1)
    many = perplexity(model, tokens, batch_size=7)
    assert one.tokens == many.tokens
    assert one.loss == pytest.approx(many.loss, rel=1e-5)


def test_it_matches_scoring_each_window_by_hand(model):
    tokens = random_tokens(40)  # windows start at 0, 16, 32; the last holds 7 predictions
    model.eval()
    losses = []
    for start in (0, 16, 32):
        window = torch.from_numpy(tokens[start : start + 17].astype(np.int64))[None]
        logits = model(window[:, :-1])
        losses.append(F.cross_entropy(logits[0], window[0, 1:], reduction="none"))
    expected = torch.cat(losses).mean().item()
    assert perplexity(model, tokens).loss == pytest.approx(expected, rel=1e-5)
