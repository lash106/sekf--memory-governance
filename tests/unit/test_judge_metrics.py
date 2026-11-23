import pytest

from sekf.contracts import JudgeLabel
from sekf.judge import (
    confidence_brier_score,
    evaluate_judge_labels,
    expected_calibration_error,
    judge_label_accuracy,
    label_confidence_calibration,
)


def test_accuracy_and_contradiction_f1() -> None:
    gold = [
        JudgeLabel.SUPPORTED,
        JudgeLabel.REFUTED,
        JudgeLabel.REFUTED,
        JudgeLabel.INSUFFICIENT_EVIDENCE,
    ]
    pred = [
        JudgeLabel.SUPPORTED,
        JudgeLabel.REFUTED,
        JudgeLabel.SUPPORTED,  # miss a contradiction
        JudgeLabel.INSUFFICIENT_EVIDENCE,
    ]
    assert judge_label_accuracy(gold, pred) == 0.75
    report = evaluate_judge_labels(gold, pred)
    assert report.n == 4
    assert report.accuracy == 0.75
    # TP=1, FN=1, FP=0 => precision=1, recall=0.5, f1=2/3
    assert abs(report.contradiction_f1 - (2.0 / 3.0)) < 1e-9


def test_brier_and_ece_for_confident_and_calibrated_predictions() -> None:
    assert confidence_brier_score([], []) == 0.0
    assert expected_calibration_error([], []) == 0.0
    assert confidence_brier_score([True, True], [1.0, 1.0]) == 0.0
    assert expected_calibration_error([True, True], [1.0, 1.0]) == 0.0
    assert confidence_brier_score([False], [1.0]) == 1.0
    half_correct = [True, False] * 5
    half_confidence = [1.0] * 10
    assert expected_calibration_error(half_correct, half_confidence) == pytest.approx(0.5)
    calibrated = [True, False]
    assert expected_calibration_error(calibrated, [0.5, 0.5]) == pytest.approx(0.0)


def test_label_calibration_uses_exact_label_match() -> None:
    gold = [JudgeLabel.SUPPORTED, JudgeLabel.REFUTED]
    predicted = [JudgeLabel.SUPPORTED, JudgeLabel.SUPPORTED]
    report = label_confidence_calibration(gold, predicted, [1.0, 0.0])
    assert report.n == 2
    assert report.brier == pytest.approx(0.0)
    assert report.ece == pytest.approx(0.0)


def test_calibration_rejects_mismatched_lengths_and_bad_confidence() -> None:
    with pytest.raises(ValueError, match="same length"):
        confidence_brier_score([True], [])
    with pytest.raises(ValueError, match="between 0 and 1"):
        confidence_brier_score([True], [1.2])
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error([True], [0.5], n_bins=0)
