"""Judge confidence scores and temperature scaling. Owner: Sai Chetan.

`JudgeAssessment` stores one confidence for the chosen label, not a distribution
over labels. Brier and ECE treat that confidence as P(the predicted label equals
the gold label). Temperature scaling rescales the same confidence before policy
rules R2 and R5. None of this calls an LLM or reads a final action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sekf.contracts import JudgeLabel


@dataclass(frozen=True)
class CalibrationReport:
    """Brier score and expected calibration error for one set of predictions."""

    n: int
    brier: float
    ece: float


def _check_pairs(correct: list[bool], confidences: list[float]) -> None:
    if len(correct) != len(confidences):
        raise ValueError("correct and confidences must have the same length")


def _check_confidence(confidence: float) -> None:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number between 0 and 1")
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0 and 1")


def confidence_brier_score(correct: list[bool], confidences: list[float]) -> float:
    """Mean squared error of reported confidence against label correctness.

    An empty input returns 0. A confidence of 1 on a wrong label scores 1.
    """

    _check_pairs(correct, confidences)
    if not correct:
        return 0.0
    total = 0.0
    for ok, confidence in zip(correct, confidences, strict=True):
        _check_confidence(confidence)
        outcome = 1.0 if ok else 0.0
        total += (float(confidence) - outcome) ** 2
    return total / len(correct)


def expected_calibration_error(
    correct: list[bool], confidences: list[float], *, n_bins: int = 10
) -> float:
    """Weighted absolute gap between bin accuracy and mean confidence.

    Bins are equal-width on [0, 1]. Confidence 1.0 falls in the last bin.
    An empty input returns 0.
    """

    _check_pairs(correct, confidences)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if not correct:
        return 0.0
    buckets: list[list[tuple[bool, float]]] = [[] for _ in range(n_bins)]
    for ok, confidence in zip(correct, confidences, strict=True):
        _check_confidence(confidence)
        value = float(confidence)
        index = min(n_bins - 1, int(value * n_bins))
        buckets[index].append((ok, value))
    total = len(correct)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        accuracy = sum(1 for ok, _ in bucket if ok) / len(bucket)
        mean_confidence = sum(value for _, value in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(accuracy - mean_confidence)
    return error


def label_confidence_calibration(
    gold: list[JudgeLabel],
    predicted: list[JudgeLabel],
    confidences: list[float],
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    """Calibrate a parallel list of gold labels, predictions, and confidences."""

    if len(gold) != len(predicted) or len(gold) != len(confidences):
        raise ValueError("gold, predicted, and confidences must have the same length")
    correct = [left is right for left, right in zip(gold, predicted, strict=True)]
    return CalibrationReport(
        n=len(gold),
        brier=confidence_brier_score(correct, confidences),
        ece=expected_calibration_error(correct, confidences, n_bins=n_bins),
    )


def _clip(probability: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, probability))


def apply_temperature(confidence: float, temperature: float) -> float:
    """Return the confidence after dividing its logit by `temperature`."""

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    _check_confidence(confidence)
    clipped = _clip(float(confidence))
    logit = math.log(clipped / (1.0 - clipped))
    scaled = logit / temperature
    return 1.0 / (1.0 + math.exp(-scaled))


def _nll(confidences: list[float], correct: list[bool], temperature: float) -> float:
    total = 0.0
    for confidence, ok in zip(confidences, correct, strict=True):
        probability = _clip(apply_temperature(confidence, temperature))
        total += -(math.log(probability) if ok else math.log(1.0 - probability))
    return total / len(confidences)


def fit_temperature(confidences: list[float], correct: list[bool]) -> float:
    """Grid-search a temperature in [0.05, 5]. Empty input returns 1."""

    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not confidences:
        return 1.0
    best_t = 1.0
    best_loss = _nll(confidences, correct, best_t)
    temperature = 0.05
    while temperature <= 5.0 + 1e-9:
        loss = _nll(confidences, correct, temperature)
        if loss < best_loss:
            best_loss = loss
            best_t = temperature
        temperature += 0.05
    return round(best_t, 4)
