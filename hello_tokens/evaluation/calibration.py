"""Calibration: does the model know when it is likely to be right?

For every next-token prediction, take the model's confidence (the probability of its top choice) and
whether that top choice was the true next token. A well-calibrated model is right 70% of the time
when it is 70% confident.

Expected calibration error (ECE): sort the predictions into equal-width confidence bins, and take the
average gap between each bin's confidence and its accuracy, weighted by how many predictions fall in it:
    ECE = sum over bins of (count / total) * |accuracy - confidence|
0 is perfect; a model that says 90% and is right 60% of the time has ECE 0.3.

Only sums are kept per bin, so millions of predictions need no extra memory.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

BINS = 15


@dataclass(frozen=True)
class Bin:
    lower: float
    upper: float
    count: int
    confidence: float  # average confidence of the predictions in the bin
    accuracy: float  # share of them whose top choice was right


@dataclass(frozen=True)
class Calibration:
    ece: float
    accuracy: float  # top-1 accuracy over every prediction
    predictions: int
    bins: list[Bin]


class CalibrationBins:
    """Collects confidence and correctness, batch by batch."""

    def __init__(self, bins: int = BINS):
        self.bins = bins
        self.count = torch.zeros(bins, dtype=torch.float64)
        self.confidence = torch.zeros(bins, dtype=torch.float64)
        self.correct = torch.zeros(bins, dtype=torch.float64)

    def add(self, confidence: torch.Tensor, correct: torch.Tensor) -> None:
        """confidence: probabilities in [0, 1]; correct: matching booleans (any shape)."""
        confidence = confidence.detach().flatten().double().cpu()
        correct = correct.detach().flatten().double().cpu()
        # Bin b covers [b/bins, (b+1)/bins); a confidence of exactly 1 goes in the last bin.
        index = (confidence * self.bins).long().clamp(max=self.bins - 1)
        self.count += torch.bincount(index, minlength=self.bins)
        self.confidence += torch.bincount(index, weights=confidence, minlength=self.bins)
        self.correct += torch.bincount(index, weights=correct, minlength=self.bins)

    def add_logits(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probabilities = torch.softmax(logits.float(), dim=-1)
        confidence, choice = probabilities.max(dim=-1)
        self.add(confidence, choice == targets)

    def result(self) -> Calibration:
        total = self.count.sum().item()
        if total == 0:
            raise ValueError("no predictions were added")
        bins, ece = [], 0.0
        for b in range(self.bins):
            n = self.count[b].item()
            confidence = self.confidence[b].item() / n if n else 0.0
            accuracy = self.correct[b].item() / n if n else 0.0
            ece += n / total * abs(accuracy - confidence)
            bins.append(Bin(b / self.bins, (b + 1) / self.bins, int(n), confidence, accuracy))
        return Calibration(ece, self.correct.sum().item() / total, int(total), bins)


def save_reliability_diagram(result: Calibration, path: Path, title: str) -> None:
    """Accuracy against confidence per bin. A calibrated model's bars touch the diagonal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    used = [b for b in result.bins if b.count]
    width = 1 / len(result.bins)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    top.bar([b.lower for b in used], [b.accuracy for b in used], width=width, align="edge",
            edgecolor="white", label="accuracy")
    top.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    top.set_ylabel("accuracy")
    top.set_title(f"{title}\nECE {result.ece:.4f} · top-1 accuracy {result.accuracy:.1%}")
    top.legend(loc="upper left")
    bottom.bar([b.lower for b in used], [b.count / result.predictions for b in used], width=width,
               align="edge", edgecolor="white", color="grey")
    bottom.set_xlabel("confidence (probability of the top choice)")
    bottom.set_ylabel("share")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
