"""Judge-label metrics (SPEC.md section 14). Owner: Sai Chetan.

Deterministic aggregation for experiment harnesses. No LLM calls.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sekf.contracts import JudgeLabel


@dataclass(frozen=True)
class LabelCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0


@dataclass(frozen=True)
class BinaryScores:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class JudgeMetricsReport:
    """Summary of Judge label accuracy and contradiction (REFUTED) F1."""

    n: int
    accuracy: float
    contradiction_f1: float
    contradiction: BinaryScores
    per_label_support: dict[str, int]
    confusion: dict[str, dict[str, int]]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def binary_scores(counts: LabelCounts) -> BinaryScores:
    precision = _safe_div(counts.true_positive, counts.true_positive + counts.false_positive)
    recall = _safe_div(counts.true_positive, counts.true_positive + counts.false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    support = counts.true_positive + counts.false_negative
    return BinaryScores(precision=precision, recall=recall, f1=f1, support=support)


def judge_label_accuracy(gold: list[JudgeLabel], predicted: list[JudgeLabel]) -> float:
    """Overall exact-match accuracy on Judge labels."""

    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    if not gold:
        return 0.0
    correct = sum(1 for g, p in zip(gold, predicted, strict=True) if g is p)
    return correct / len(gold)


def contradiction_detection_scores(
    gold: list[JudgeLabel], predicted: list[JudgeLabel]
) -> tuple[BinaryScores, LabelCounts]:
    """Treat gold/pred REFUTED as the positive class (contradiction detection F1)."""

    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    tp = fp = fn = tn = 0
    for g, p in zip(gold, predicted, strict=True):
        g_pos = g is JudgeLabel.REFUTED
        p_pos = p is JudgeLabel.REFUTED
        if g_pos and p_pos:
            tp += 1
        elif not g_pos and p_pos:
            fp += 1
        elif g_pos and not p_pos:
            fn += 1
        else:
            tn += 1
    counts = LabelCounts(tp, fp, fn, tn)
    return binary_scores(counts), counts


def confusion_matrix(
    gold: list[JudgeLabel], predicted: list[JudgeLabel]
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {label.value: {} for label in JudgeLabel}
    for g, p in zip(gold, predicted, strict=True):
        row = matrix[g.value]
        row[p.value] = row.get(p.value, 0) + 1
    return matrix


def evaluate_judge_labels(
    gold: list[JudgeLabel], predicted: list[JudgeLabel]
) -> JudgeMetricsReport:
    """Compute accuracy, contradiction F1, and a confusion matrix."""

    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    contradiction, _ = contradiction_detection_scores(gold, predicted)
    support = Counter(label.value for label in gold)
    return JudgeMetricsReport(
        n=len(gold),
        accuracy=judge_label_accuracy(gold, predicted),
        contradiction_f1=contradiction.f1,
        contradiction=contradiction,
        per_label_support=dict(support),
        confusion=confusion_matrix(gold, predicted),
    )
