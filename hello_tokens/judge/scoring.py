"""Scoring the options of a question, and the confidence switch.

An option's score is how likely the model finds it as the continuation of the context: the sum of the
log-probabilities of its tokens (and, as an alternative, their average, which doesn't favour short
options). All options are scored in one batched pass, which is the point of judge mode: one model call
instead of writing an answer token by token.

The switch: the softmax over the options' scores gives a confidence for the model's choice. Above a
threshold the judge answers; below it, it abstains (and the caller can fall back to writing). The
threshold is chosen on one half of the questions and the result reported on the other half.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hello_tokens.evaluation.calibration import CalibrationBins
from hello_tokens.judge.questions import Question
from hello_tokens.model.gpt import GPT
from hello_tokens.tokenizer.tokenizer import Tokenizer

METHODS = ("summed", "mean")


@dataclass(frozen=True)
class Judgement:
    summed: list[float]  # total log-probability of each option
    mean: list[float]  # average log-probability per token of each option

    def probabilities(self, method: str, temperature: float = 1.0) -> torch.Tensor:
        """How much the model favours each option: a softmax over the scores, divided by the temperature."""
        return torch.softmax(torch.tensor(getattr(self, method), dtype=torch.float64) / temperature, dim=0)


@torch.no_grad()
def score_options(model: GPT, tokenizer: Tokenizer, question: Question) -> Judgement:
    model.eval()
    device = next(model.parameters()).device
    options = [tokenizer.encode(" " + option) for option in question.options]
    # The model reads context + option, minus the last token, so the context may use what's left of
    # the window after the longest option. A long context loses its beginning, never its end.
    room = model.config.context + 1 - max(len(o) for o in options)
    if room < 1:
        raise ValueError("an option is longer than the model's context")
    context = (tokenizer.encode(question.context) or [tokenizer.end_of_text_id])[-room:]
    rows = [context + option for option in options]
    width = max(len(r) for r in rows)
    # Pad on the right: a token never sees what comes after it, so padding can't change any score.
    batch = torch.tensor([r + [0] * (width - len(r)) for r in rows], device=device)
    log_probs = F.log_softmax(model(batch[:, :-1]).float(), dim=-1)
    summed, mean = [], []
    for i, option in enumerate(options):
        # The option's tokens sit at positions len(context) ... len(context) + len(option) - 1, and
        # each is predicted from the position just before it.
        positions = torch.arange(len(context) - 1, len(context) - 1 + len(option), device=device)
        scores = log_probs[i, positions, torch.tensor(option, device=device)]
        summed.append(scores.sum().item())
        mean.append(scores.mean().item())
    return Judgement(summed, mean)


@dataclass(frozen=True)
class Outcome:
    threshold: float
    coverage: float  # share of questions answered
    accuracy: float  # share of answered questions answered right


def decisions(judgements: list[Judgement], answers: list[int], method: str,
              temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Each question's confidence (the chosen option's probability) and whether the choice was right."""
    confidence, correct = [], []
    for judgement, answer in zip(judgements, answers):
        p, choice = judgement.probabilities(method, temperature).max(dim=0)
        confidence.append(float(p))
        correct.append(int(choice) == answer)
    return torch.tensor(confidence, dtype=torch.float64), torch.tensor(correct)


def outcome(judgements: list[Judgement], answers: list[int], method: str, threshold: float,
            temperature: float = 1.0) -> Outcome:
    confidence, correct = decisions(judgements, answers, method, temperature)
    answered = confidence >= threshold
    count = int(answered.sum())
    accuracy = correct[answered].double().mean().item() if count else 1.0
    return Outcome(threshold, count / len(answers), accuracy)


def choose_threshold(judgements: list[Judgement], answers: list[int], method: str, target: float,
                     temperature: float = 1.0) -> float:
    """The lowest confidence threshold at which the answered questions reach the target accuracy:
    answer as much as possible while staying that accurate. 1.0 if no threshold gets there."""
    confidence, correct = decisions(judgements, answers, method, temperature)
    order = torch.argsort(confidence, descending=True)
    # Answering the i most confident questions: their running accuracy.
    running = torch.cumsum(correct[order].double(), 0) / torch.arange(1, len(order) + 1)
    sorted_confidence = confidence[order]
    for i in range(len(order) - 1, -1, -1):  # from answering everything down to answering one
        # Ties share a threshold: only cut where the next confidence is strictly lower.
        if (i == len(order) - 1 or sorted_confidence[i + 1] < sorted_confidence[i]) and running[i] >= target:
            return float(sorted_confidence[i])
    return 1.0


def option_calibration(judgements: list[Judgement], answers: list[int], method: str, temperature: float = 1.0,
                       bins: int = 10):
    """ECE of the judge's own confidence (not the next-token confidence of lesson 21)."""
    confidence, correct = decisions(judgements, answers, method, temperature)
    collector = CalibrationBins(bins)
    collector.add(confidence, correct)
    return collector.result()


def fit_temperature(judgements: list[Judgement], answers: list[int], method: str) -> float:
    """Temperature scaling (Guo et al., 2017): the one number T that makes the true answers most likely.

    Summing a whole sentence's log-probabilities makes the gaps between options large, so the softmax
    is far too sure of itself. Dividing every score by T > 1 softens it. T is chosen by minimising the
    negative log-likelihood of the true answers, on a grid from 0.25 to 100. Dividing by a positive
    number never changes which option scores highest, so accuracy is untouched: only the confidence moves.
    """
    scores = torch.tensor([getattr(j, method) for j in judgements], dtype=torch.float64)
    truth = torch.tensor(answers)
    grid = torch.logspace(-0.6, 2, 400, dtype=torch.float64)  # 0.25 ... 100
    losses = [torch.nn.functional.cross_entropy(scores / t, truth).item() for t in grid]
    return float(grid[int(torch.tensor(losses).argmin())])