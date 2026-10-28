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


def quorum_size(n_members: int) -> int:
    """How many members must agree before a label can authorize an action.

    This is a majority of the full panel, not of whoever happened to return.
    A three-member ensemble therefore needs two agreeing votes.
    """

    if n_members < 1:
        raise ValueError("n_members must be >= 1")
    return n_members // 2 + 1


def label_with_quorum(labels: list[JudgeLabel], n_members: int) -> JudgeLabel | None:
    """Return the label shared by at least `quorum_size` members, if one exists."""

    need = quorum_size(n_members)
    counts = Counter(labels)
    winners = [label for label, count in counts.items() if count >= need]
    if len(winners) == 1:
        return winners[0]
    return None
