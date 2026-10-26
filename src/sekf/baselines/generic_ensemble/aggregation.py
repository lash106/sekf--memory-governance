"""Majority-vote aggregation for generic_ensemble (SPEC.md section 10.2)."""

from __future__ import annotations

from collections import Counter

from sekf.contracts import JudgeLabel

# Tie -> INSUFFICIENT_EVIDENCE (documented SPEC rule).
TIE_LABEL = JudgeLabel.INSUFFICIENT_EVIDENCE


def majority_judge_label(labels: list[JudgeLabel]) -> JudgeLabel:
    """Return the majority label; ties (including empty) become INSUFFICIENT_EVIDENCE."""

    if not labels:
        return TIE_LABEL
    counts = Counter(labels)
    best = max(counts.values())
    winners = [label for label, count in counts.items() if count == best]
    if len(winners) != 1:
        return TIE_LABEL
    return winners[0]
