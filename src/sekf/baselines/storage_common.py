"""Shared machinery for the storage and retrieval baselines. Owner: Krishna.

`direct_storage`, `latest_write_wins` and `static_rag` make NO LLM calls. Each one picks a
`FinalAction` with a fixed rule, then (in `commit` mode) writes it through the same
`MemoryRepository.apply_decision` as the governed system. That keeps the comparison fair:
the only thing that differs between systems is who decides the action.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar
from uuid import UUID

from sekf.baselines.decision_links import action_claim_links
from sekf.contracts import (
    ClaimStatus,
    CommitBundle,
    DecideMode,
    DecideRequest,
    DecideResponse,
    DecisionRecord,
    ExistingClaim,
    FinalAction,
    MemoryDelta,
    Thresholds,
    compute_decision_id,
    compute_request_hash,
)
from sekf.memory import MemoryRepository


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class StorageChoice:
    """What a storage baseline decided, and why."""

    action: FinalAction
    rule_id: str
    explanation: str
    target_claim_id: UUID | None = None


class StorageBaseline(ABC):
    """A deterministic baseline that decides with a fixed rule and can commit the result."""

    baseline_name: ClassVar[str]
    policy_version: ClassVar[str]

    def __init__(
        self,
        memory: MemoryRepository,
        *,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._memory = memory
        self._clock = clock

    @abstractmethod
    def choose(self, request: DecideRequest, related: list[ExistingClaim]) -> StorageChoice:
        """Pick the action from the request and the related claims. Must be pure."""

    async def decide(self, request: DecideRequest) -> DecideResponse:
        """Decide; in `commit` mode also write the decision to memory atomically."""

        started = time.perf_counter()
        thresholds = request.thresholds or Thresholds()
        digest = compute_request_hash(
            policy_version=self.policy_version,
            candidate=request.candidate,
            evidence=request.evidence,
            thresholds=thresholds,
            run_config=request.run_config,
            idempotency_key=request.idempotency_key,
        )
        decision_id = compute_decision_id(self.policy_version, digest)
        commit = request.mode is DecideMode.COMMIT
        if commit:
            stored = self._memory.get_bundle(decision_id)
            if stored is not None:  # a retry costs nothing and returns what was committed
                return self._replay(stored, started)

        related = self._memory.search_related_claims(request.candidate)
        choice = self.choose(request, related)
        previous, resulting = action_claim_links(
            choice.action,
            candidate_id=request.candidate.claim_id,
            related=related,
            target_claim_id=choice.target_claim_id,
        )
        decision = DecisionRecord(
            decision_id=decision_id,
            candidate_claim_id=request.candidate.claim_id,
            final_action=choice.action,
            rule_action=choice.action,
            policy_version=self.policy_version,
            request_hash=digest,
            thresholds=thresholds,
            rule_ids_fired=[choice.rule_id],
            previous_claim_ids=previous,
            resulting_claim_id=resulting,
            explanation=choice.explanation,
            created_at=self._clock(),
        )
        delta: MemoryDelta | None = None
        if commit:
            delta = self._memory.apply_decision(
                CommitBundle(
                    candidate=request.candidate,
                    evidence=request.evidence,
                    decision=decision,
                    memory_before=related,
                    idempotency_key=request.idempotency_key,
                    run_config=request.run_config,
                )
            )
        return DecideResponse(
            mode=request.mode,
            decision=decision,
            memory_delta=delta,
            wall_latency_ms=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _replay(stored: CommitBundle, started: float) -> DecideResponse:
        return DecideResponse(
            mode=DecideMode.COMMIT,
            decision=stored.decision,
            memory_delta=MemoryDelta(
                decision_id=stored.decision.decision_id,
                applied=False,
                replayed=True,
                note="replay: returned the originally committed decision",
            ),
            wall_latency_ms=int((time.perf_counter() - started) * 1000),
        )


def active_related(related: list[ExistingClaim]) -> list[ExistingClaim]:
    """ACTIVE claims only: superseded, quarantined and rejected claims are not 'current'."""

    return [claim for claim in related if claim.status is ClaimStatus.ACTIVE]
