"""Confidence calibration for Judge labels (SPEC.md section 14). Owner: Sai Chetan.

`JudgeAssessment` stores one confidence for the chosen label, not a distribution
over labels. These scores treat that confidence as P(the predicted label equals
the gold label). They do not call an LLM and they do not read a final action.
"""

from __future__ import annotations

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
