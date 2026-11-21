"""Historian metrics: temporal-reasoning accuracy and provenance completeness.

Owner: Mahalakshmi. SPEC.md section 14 lists both as secondary metrics, and section 18
(Sprint 2) assigns them to the Historian owner. They also feed the Sprint 3 error analysis
for the provenance / time / conflict category.

Plain Python, no contract imports, so the functions work on per-case JSON rows from any
runner (``DecideResponse.model_dump(mode="json")`` and the case file). Every score keeps
its raw counts next to the rate, as section 14 asks.

Definitions used here:

- Temporal-reasoning accuracy: the share of labeled cases where the Historian's
  ``temporal_relation`` equals the gold ``temporal_relation``. ``conflict_type`` accuracy
  and the joint score (both fields right) are reported next to it, because the gold
  Historian labels in section 13.1 are the pair.
- A missing prediction (the Historian failed or was not run) counts as wrong, not skipped,
  so a system cannot look better by failing on hard cases.
- Provenance completeness: for decisions that put a claim into long-term knowledge
  (``PROMOTE``, ``REVISE``, ``ATTACH_EVIDENCE``), the share whose audit trail can be traced
  back to a source, a time, the Historian's provenance assessment, cited evidence, the
  superseded claim (for revisions), and the rules that fired.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Actions that add to or change long-term knowledge. Provenance is only scored for these:
# a rejected or quarantined claim never becomes knowledge, so it needs no trace back.
KNOWLEDGE_ACTIONS = frozenset({"PROMOTE", "REVISE", "ATTACH_EVIDENCE"})

PROVENANCE_CHECKS = (
    "source_identified",
    "time_recorded",
    "historian_linked",
    "provenance_assessed",
    "evidence_cited",
    "history_linked",
    "rules_recorded",
)


@dataclass(frozen=True)
class Score:
    """A rate with its raw counts. ``value`` is ``None`` when ``n`` is 0."""

    correct: int
    n: int

    @property
    def value(self) -> float | None:
        return self.correct / self.n if self.n else None


def _label(value: Any) -> str | None:
    """Normalize a label to an upper-case string; empty or missing becomes ``None``."""

    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip().upper()
    return text or None


def _check_lengths(*sequences: Sequence[Any]) -> None:
    lengths = {len(seq) for seq in sequences}
    if len(lengths) > 1:
        raise ValueError(f"sequences must have the same length, got {sorted(lengths)}")


def label_accuracy(gold: Sequence[Any], predicted: Sequence[Any]) -> Score:
    """Accuracy over cases that have a gold label. A missing prediction counts as wrong."""

    _check_lengths(gold, predicted)
    correct = n = 0
    for gold_label, predicted_label in zip(gold, predicted, strict=True):
        expected = _label(gold_label)
        if expected is None:
            continue
        n += 1
        correct += int(_label(predicted_label) == expected)
    return Score(correct, n)


def temporal_relation_accuracy(gold: Sequence[Any], predicted: Sequence[Any]) -> Score:
    """Temporal-reasoning accuracy: Historian ``temporal_relation`` versus gold."""

    return label_accuracy(gold, predicted)


def conflict_type_accuracy(gold: Sequence[Any], predicted: Sequence[Any]) -> Score:
    """Historian ``conflict_type`` versus gold."""

    return label_accuracy(gold, predicted)


def joint_historian_accuracy(
    gold_relation: Sequence[Any],
    predicted_relation: Sequence[Any],
    gold_conflict: Sequence[Any],
    predicted_conflict: Sequence[Any],
) -> Score:
    """Share of cases where both ``temporal_relation`` and ``conflict_type`` are right.

    Only cases with both gold labels are counted.
    """

    _check_lengths(gold_relation, predicted_relation, gold_conflict, predicted_conflict)
    correct = n = 0
    for g_rel, p_rel, g_con, p_con in zip(
        gold_relation, predicted_relation, gold_conflict, predicted_conflict, strict=True
    ):
        if _label(g_rel) is None or _label(g_con) is None:
            continue
        n += 1
        correct += int(_label(p_rel) == _label(g_rel) and _label(p_con) == _label(g_con))
    return Score(correct, n)


def confusion_counts(gold: Sequence[Any], predicted: Sequence[Any]) -> Counter[tuple[str, str]]:
    """Counts of ``(gold, predicted)`` pairs that disagree, for error analysis.

    A missing prediction shows up as ``"MISSING"``. Cases without a gold label are skipped.
    """

    _check_lengths(gold, predicted)
    counts: Counter[tuple[str, str]] = Counter()
    for gold_label, predicted_label in zip(gold, predicted, strict=True):
        expected = _label(gold_label)
        if expected is None:
            continue
        got = _label(predicted_label) or "MISSING"
        if got != expected:
            counts[(expected, got)] += 1
    return counts


def accuracy_by_category(
    categories: Sequence[str], gold: Sequence[Any], predicted: Sequence[Any]
) -> dict[str, Score]:
    """Accuracy per dataset category (SPEC.md section 13.3), sorted by category name."""

    _check_lengths(categories, gold, predicted)
    grouped: dict[str, tuple[list[Any], list[Any]]] = {}
    for category, gold_label, predicted_label in zip(categories, gold, predicted, strict=True):
        golds, preds = grouped.setdefault(str(category), ([], []))
        golds.append(gold_label)
        preds.append(predicted_label)
    return {
        category: label_accuracy(golds, preds)
        for category, (golds, preds) in sorted(grouped.items())
    }


def _ids(values: Iterable[Any] | None) -> set[str]:
    return {str(value) for value in values or [] if value is not None}


def provenance_checks(
    response: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, bool | None]:
    """Check one decision's provenance trail.

    ``response`` is a ``DecideResponse`` as JSON (``model_dump(mode="json")``) and
    ``candidate`` is the candidate claim as JSON. Each check is ``True``, ``False``, or
    ``None`` when it does not apply (for example ``history_linked`` for a new claim).
    """

    decision = response.get("decision") or {}
    historian = response.get("historian") or {}
    judge = response.get("judge") or {}
    action = _label(decision.get("final_action"))

    checks: dict[str, bool | None] = {
        "source_identified": bool(str(candidate.get("source_id") or "").strip()),
        "time_recorded": bool(candidate.get("observed_at")),
        "historian_linked": bool(historian)
        and decision.get("historian_assessment_id") is not None
        and str(decision.get("historian_assessment_id")) == str(historian.get("assessment_id")),
        "provenance_assessed": bool(historian)
        and _label(historian.get("provenance_quality")) not in (None, "UNKNOWN"),
        "evidence_cited": bool(_ids(judge.get("supporting_evidence_ids"))),
        "history_linked": None,
        "rules_recorded": bool(decision.get("rule_ids_fired")),
    }
    if action == "REVISE":
        superseded = historian.get("supersedes_claim_id")
        checks["history_linked"] = superseded is not None and str(superseded) in _ids(
            decision.get("previous_claim_ids")
        )
    return checks


@dataclass(frozen=True)
class ProvenanceSummary:
    """Provenance completeness over the decisions that wrote long-term knowledge."""

    fully_complete: Score
    excluded: int
    per_check: dict[str, Score] = field(default_factory=dict)


def provenance_completeness(
    items: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> ProvenanceSummary:
    """Score ``(response, candidate)`` pairs.

    Only decisions whose final action is in ``KNOWLEDGE_ACTIONS`` are scored; the rest are
    counted in ``excluded``. A decision is fully complete when every applicable check passes.
    """

    complete = scored = excluded = 0
    passed: Counter[str] = Counter()
    applicable: Counter[str] = Counter()
    for response, candidate in items:
        action = _label((response.get("decision") or {}).get("final_action"))
        if action not in KNOWLEDGE_ACTIONS:
            excluded += 1
            continue
        scored += 1
        checks = provenance_checks(response, candidate)
        all_ok = True
        for name in PROVENANCE_CHECKS:
            result = checks[name]
            if result is None:
                continue
            applicable[name] += 1
            passed[name] += int(result)
            all_ok = all_ok and result
        complete += int(all_ok)
    per_check = {name: Score(passed[name], applicable[name]) for name in PROVENANCE_CHECKS}
    return ProvenanceSummary(Score(complete, scored), excluded, per_check)
