"""Expected-versus-actual memory-delta comparison. Owner: Krishna.

The gold label of a case lists what should change in governed memory (`expected_memory_delta`).
After a system commits its decision to a store, `MemoryDelta` says what did change. This module
puts the two in the same shape, compares them bucket by bucket, and summarizes a run.

A system that picks the wrong action gets a wrong delta, but the delta is stricter than action
accuracy: two systems that both say REVISE can still differ in WHICH claim they supersede.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from sekf.contracts import (
    ClaimStatus,
    DecideMode,
    DecideRequest,
    DecideResponse,
    MemoryDelta,
)
from sekf.memory import MemoryRepository
from sekf.memory.fixtures import FIXTURE_NOW, MemoryCase, build_seeded_store

BUCKETS = (
    "superseded_claim_ids",
    "new_active_claim_ids",
    "quarantined_claim_ids",
    "rejected_claim_ids",
    "attached_evidence_ids",
)

_STATUS_BUCKET = {
    ClaimStatus.SUPERSEDED: "superseded_claim_ids",
    ClaimStatus.ACTIVE: "new_active_claim_ids",
    ClaimStatus.QUARANTINED: "quarantined_claim_ids",
    ClaimStatus.REJECTED: "rejected_claim_ids",
}


@dataclass(frozen=True)
class DeltaSet:
    """Ids touched per bucket, as sets so order never matters."""

    buckets: dict[str, frozenset[str]] = field(default_factory=dict)
    # Claims left as CANDIDATE for a human. Reported, never compared: the annotation guide
    # treats ESCALATE_FOR_REVIEW as "governed memory unchanged".
    review_claim_ids: frozenset[str] = frozenset()

    def get(self, bucket: str) -> frozenset[str]:
        return self.buckets.get(bucket, frozenset())

    @classmethod
    def from_gold(cls, expected: dict[str, list[str]]) -> DeltaSet:
        unknown = set(expected) - set(BUCKETS)
        if unknown:
            raise ValueError(f"unknown delta buckets: {sorted(unknown)}")
        return cls({name: frozenset(expected.get(name, [])) for name in BUCKETS})

    @classmethod
    def from_memory_delta(cls, delta: MemoryDelta | None) -> DeltaSet:
        """What a commit changed. A missing delta (dry run, no write) is the empty delta."""

        grouped: dict[str, set[str]] = {name: set() for name in BUCKETS}
        review: set[str] = set()
        if delta is not None:
            for change in delta.status_changes:
                if change.new_status is ClaimStatus.CANDIDATE:
                    review.add(str(change.claim_id))
                else:
                    grouped[_STATUS_BUCKET[change.new_status]].add(str(change.claim_id))
            grouped["attached_evidence_ids"].update(str(e) for e in delta.attached_evidence_ids)
        return cls({k: frozenset(v) for k, v in grouped.items()}, frozenset(review))


@dataclass(frozen=True)
class BucketDiff:
    bucket: str
    expected: frozenset[str]
    actual: frozenset[str]

    @property
    def missing(self) -> frozenset[str]:
        return self.expected - self.actual

    @property
    def unexpected(self) -> frozenset[str]:
        return self.actual - self.expected

    @property
    def matches(self) -> bool:
        return self.expected == self.actual


@dataclass(frozen=True)
class DeltaComparison:
    diffs: tuple[BucketDiff, ...]
    review_claim_ids: frozenset[str] = frozenset()

    @property
    def exact_match(self) -> bool:
        return all(diff.matches for diff in self.diffs)

    @property
    def mismatched_buckets(self) -> list[str]:
        return [diff.bucket for diff in self.diffs if not diff.matches]


def compare_deltas(expected: DeltaSet, actual: DeltaSet) -> DeltaComparison:
    return DeltaComparison(
        diffs=tuple(BucketDiff(name, expected.get(name), actual.get(name)) for name in BUCKETS),
        review_claim_ids=actual.review_claim_ids,
    )


def summarize_comparisons(comparisons: Sequence[DeltaComparison]) -> dict[str, object]:
    """Exact-match rate plus per-bucket precision and recall over a run.

    Precision and recall are None for a bucket nothing expected or produced anywhere.
    """

    n = len(comparisons)
    per_bucket: dict[str, dict[str, float | None]] = {}
    for name in BUCKETS:
        tp = fp = fn = 0
        for comparison in comparisons:
            diff = next(d for d in comparison.diffs if d.bucket == name)
            tp += len(diff.expected & diff.actual)
            fp += len(diff.unexpected)
            fn += len(diff.missing)
        per_bucket[name] = {
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    exact = sum(1 for c in comparisons if c.exact_match)
    return {
        "cases": n,
        "exact_match": exact,
        "exact_match_rate": exact / n if n else None,
        "per_bucket": per_bucket,
    }


class Decider(Protocol):
    async def decide(self, request: DecideRequest) -> DecideResponse: ...


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    response: DecideResponse
    comparison: DeltaComparison
    action_correct: bool


async def evaluate_case(
    make_system: Callable[[MemoryRepository, Callable[[], datetime]], Decider],
    case: MemoryCase,
    *,
    clock: Callable[[], datetime] = lambda: FIXTURE_NOW,
) -> CaseOutcome:
    """Run one system on one case against a fresh seeded store and compare memory deltas.

    The system is built per case so no state leaks between cases. `make_system` receives the
    store and the clock to use.
    """

    store = build_seeded_store(case, clock=clock)
    try:
        system = make_system(store, clock)
        response = await system.decide(
            DecideRequest(candidate=case.candidate, evidence=case.evidence, mode=DecideMode.COMMIT)
        )
        comparison = compare_deltas(
            DeltaSet.from_gold(case.expected_delta),
            DeltaSet.from_memory_delta(response.memory_delta),
        )
        return CaseOutcome(
            case_id=case.case_id,
            response=response,
            comparison=comparison,
            action_correct=response.decision.final_action is case.gold_action,
        )
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()


def action_accuracy(outcomes: Iterable[CaseOutcome]) -> float | None:
    items = list(outcomes)
    return sum(o.action_correct for o in items) / len(items) if items else None


__all__ = [
    "BUCKETS",
    "BucketDiff",
    "CaseOutcome",
    "DeltaComparison",
    "DeltaSet",
    "Decider",
    "action_accuracy",
    "compare_deltas",
    "evaluate_case",
    "summarize_comparisons",
]
