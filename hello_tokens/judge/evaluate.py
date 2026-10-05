"""Running the judge on the question set, with an honest split.

The first half of the questions chooses the settings (summed or mean scoring, the temperature that
calibrates the confidence, and the confidence threshold); the second half, never used for choosing, is
what the results report. Choosing and reporting on the same questions would flatter the result.
"""

import time

from hello_tokens.judge.questions import Question
from hello_tokens.judge.scoring import (
    METHODS, choose_threshold, fit_temperature, option_calibration, outcome, score_options,
)
from hello_tokens.model.gpt import GPT
from hello_tokens.tokenizer.tokenizer import Tokenizer

REPORT_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def run_judge(model: GPT, tokenizer: Tokenizer, questions: list[Question], target: float = 0.9,
              progress=None) -> dict:
    started = time.perf_counter()
    judgements = []
    for i, question in enumerate(questions):
        judgements.append(score_options(model, tokenizer, question))
        if progress and (i + 1) % 200 == 0:
            progress(i + 1)
    seconds = time.perf_counter() - started
    answers = [q.answer for q in questions]
    half = len(questions) // 2
    choose_j, choose_a = judgements[:half], answers[:half]
    report_j, report_a = judgements[half:], answers[half:]

    # Settings come from the first half only.
    method = max(METHODS, key=lambda m: outcome(choose_j, choose_a, m, 0.0).accuracy)
    temperature = fit_temperature(choose_j, choose_a, method)
    threshold = choose_threshold(choose_j, choose_a, method, target, temperature)

    switched = outcome(report_j, report_a, method, threshold, temperature)
    return {
        "questions": len(questions),
        "ms_per_question": seconds / len(questions) * 1000,
        "method": method,
        "accuracy_by_method": {m: outcome(report_j, report_a, m, 0.0).accuracy for m in METHODS},
        "temperature": temperature,
        "option_ece_raw": option_calibration(report_j, report_a, method).ece,
        "option_ece": option_calibration(report_j, report_a, method, temperature).ece,
        "target_accuracy": target,
        "threshold": threshold,
        "answered": switched.coverage,
        "accuracy_when_answering": switched.accuracy,
        "coverage_curve": [vars(outcome(report_j, report_a, method, t, temperature)) for t in REPORT_THRESHOLDS],
    }
