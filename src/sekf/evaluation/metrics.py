"""SPEC.md section 14 metrics for the offline Sprint-3 harness. Deterministic. No LLM.

Contradiction F1 uses the Judge label REFUTED, matching the rest of this repository.
False-promotion rate is separate: it counts ACTIVE promotions of refuted or conflicting cases.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sekf.contracts import FinalAction


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def binary_f1(gold: Sequence[str], predicted: Sequence[str], positive: str) -> float:
    """F1 for one label. Empty input returns 0."""

    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    tp = fp = fn = 0
    for left, right in zip(gold, predicted, strict=True):
        if right == positive and left == positive:
            tp += 1
        elif right == positive:
            fp += 1
        elif left == positive:
            fn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall)


def macro_f1(gold: Sequence[str], predicted: Sequence[str]) -> float:
    """Unweighted mean F1 over labels that appear in gold or predicted."""

    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    labels = sorted(set(gold) | set(predicted))
    if not labels:
        return 0.0
    return sum(binary_f1(gold, predicted, label) for label in labels) / len(labels)


def promotion_scores(gold: Sequence[str], predicted: Sequence[str]) -> tuple[float, float]:
    """Precision and recall for final action PROMOTE."""

    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    tp = fp = fn = 0
    positive = FinalAction.PROMOTE.value
    for left, right in zip(gold, predicted, strict=True):
        if right == positive and left == positive:
            tp += 1
        elif right == positive:
            fp += 1
        elif left == positive:
            fn += 1
    return _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)


def revision_accuracy(gold: Sequence[str], predicted: Sequence[str]) -> float | None:
    """Fraction of gold REVISE cases predicted REVISE. None when no gold REVISE exists."""

    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    paired = [
        (left, right)
        for left, right in zip(gold, predicted, strict=True)
        if left == FinalAction.REVISE.value
    ]
    if not paired:
        return None
    return sum(left == right for left, right in paired) / len(paired)


def unnecessary_quarantine_rate(gold: Sequence[str], predicted: Sequence[str]) -> float:
    """Predicted QUARANTINE on a case whose gold action is not QUARANTINE, over all cases."""

    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    if not gold:
        return 0.0
    hits = sum(
        right == FinalAction.QUARANTINE.value and left != FinalAction.QUARANTINE.value
        for left, right in zip(gold, predicted, strict=True)
    )
    return hits / len(gold)


def false_promotion_rate(
    gold_actions: Sequence[str],
    predicted_actions: Sequence[str],
    *,
    gold_judge: Sequence[str],
    gold_conflict: Sequence[str],
) -> tuple[float, int, int]:
    """Promotions among refuted or conflicting cases. Returns rate, hits, and denominator."""

    rows = [gold_actions, predicted_actions, gold_judge, gold_conflict]
    if len({len(row) for row in rows}) != 1:
        raise ValueError("false-promotion inputs must have the same length")
    risky = [
        index
        for index, (label, conflict) in enumerate(zip(gold_judge, gold_conflict, strict=True))
        if label == "REFUTED" or conflict not in {"NONE", ""}
    ]
    if not risky:
        return 0.0, 0, 0
    hits = sum(1 for index in risky if predicted_actions[index] == FinalAction.PROMOTE.value)
    return hits / len(risky), hits, len(risky)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. `pct` is in 0..100."""

    if not values:
        return 0.0
    if pct < 0 or pct > 100:
        raise ValueError("pct must be between 0 and 100")
    ordered = sorted(values)
    rank = math.ceil(pct / 100 * len(ordered)) - 1
    return float(ordered[min(len(ordered) - 1, max(0, rank))])


def paired_bootstrap_interval(
    gold: Sequence[str],
    predicted: Sequence[str],
    score: Callable[[Sequence[str], Sequence[str]], float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> Interval:
    """95% paired bootstrap interval. The same index is drawn for gold and predicted."""

    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    point = score(gold, predicted)
    if not gold:
        return Interval(point, point, point)
    rng = random.Random(seed)
    stats: list[float] = []
    n = len(gold)
    for _ in range(n_resamples):
        indexes = [rng.randrange(n) for _ in range(n)]
        stats.append(score([gold[i] for i in indexes], [predicted[i] for i in indexes]))
    stats.sort()
    low = stats[int(0.025 * (n_resamples - 1))]
    high = stats[int(0.975 * (n_resamples - 1))]
    return Interval(point, low, high)


def paired_difference_interval(
    gold: Sequence[str],
    predicted_a: Sequence[str],
    predicted_b: Sequence[str],
    score: Callable[[Sequence[str], Sequence[str]], float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> Interval:
    """95% paired bootstrap interval for `score(a) - score(b)` on the same cases.

    Each resample draws one set of case indexes and scores both systems on it, so the
    interval reflects the difference between systems, not each system's own spread. An
    interval that excludes 0 means `a` and `b` differ on this data.
    """

    if not len(gold) == len(predicted_a) == len(predicted_b):
        raise ValueError("gold and both predictions must have the same length")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    point = score(gold, predicted_a) - score(gold, predicted_b)
    if not gold:
        return Interval(point, point, point)
    rng = random.Random(seed)
    stats: list[float] = []
    n = len(gold)
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        stats.append(
            score(g, [predicted_a[i] for i in idx]) - score(g, [predicted_b[i] for i in idx])
        )
    stats.sort()
    return Interval(
        point, stats[int(0.025 * (n_resamples - 1))], stats[int(0.975 * (n_resamples - 1))]
    )
