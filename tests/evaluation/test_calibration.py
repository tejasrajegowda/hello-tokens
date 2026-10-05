import numpy as np
import pytest
import torch

from hello_tokens.evaluation.calibration import CalibrationBins, save_reliability_diagram
from hello_tokens.evaluation.perplexity import perplexity
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT


def test_a_perfectly_calibrated_predictor_scores_near_zero():
    # Confidence c, right with probability exactly c: the definition of calibrated.
    g = torch.Generator().manual_seed(0)
    confidence = torch.rand(200_000, generator=g)
    correct = torch.rand(200_000, generator=g) < confidence
    bins = CalibrationBins()
    bins.add(confidence, correct)
    assert bins.result().ece < 0.005


def test_an_overconfident_predictor_by_hand():
    # Always 90% sure, right 6 times in 10: one bin, gap 0.9 - 0.6 = 0.3.
    bins = CalibrationBins()
    bins.add(torch.full((10,), 0.9), torch.tensor([True] * 6 + [False] * 4))
    result = bins.result()
    assert result.ece == pytest.approx(0.3) and result.accuracy == pytest.approx(0.6)
    assert [b.count for b in result.bins if b.count] == [10]


def test_two_bins_by_hand():
    # Bin [0.2, 0.267): 4 predictions at 0.25, 1 right -> gap 0. Bin [0.933, 1]: 6 at 1.0, 3 right -> gap 0.5.
    # ECE = 4/10 * 0 + 6/10 * 0.5 = 0.3. (A confidence of exactly 1 belongs in the last bin.)
    bins = CalibrationBins()
    bins.add(torch.tensor([0.25] * 4 + [1.0] * 6), torch.tensor([1, 0, 0, 0, 1, 1, 1, 0, 0, 0]).bool())
    assert bins.result().ece == pytest.approx(0.3)


def test_it_rides_along_with_perplexity_and_counts_every_prediction(tmp_path):
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=20, context=16, width=16, layers=1, heads=2)).eval()
    tokens = np.random.default_rng(0).integers(0, 20, 300).astype(np.uint16)
    bins = CalibrationBins()
    score = perplexity(model, tokens, calibration=bins)
    result = bins.result()
    assert result.predictions == score.tokens == 299
    assert sum(b.count for b in result.bins) == 299
    # Top-1 accuracy, counted the slow way over the same windows.
    right = 0
    with torch.no_grad():
        for start in range(0, 299, 16):
            window = torch.from_numpy(tokens[start : start + 17].astype(np.int64))[None]
            right += (model(window[:, :-1]).argmax(-1) == window[:, 1:]).sum().item()
    assert result.accuracy == pytest.approx(right / 299)
    save_reliability_diagram(result, tmp_path / "reliability.png", "tiny model")
    assert (tmp_path / "reliability.png").stat().st_size > 0


def test_nothing_added_is_an_error():
    with pytest.raises(ValueError):
        CalibrationBins().result()
