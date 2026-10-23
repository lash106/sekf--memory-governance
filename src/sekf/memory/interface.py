"""Memory module interface (SPEC.md section 10.1). Owner: Krishna."""

from typing import Protocol
from uuid import UUID

from sekf.contracts import (
    AuditTrace,
    CandidateClaim,
    CommitBundle,
    ExistingClaim,
    MemoryDelta,
)


class MemoryRepository(Protocol):
    """Typed repository, not a separate service.

    Reads are side-effect free. There is exactly one write: `apply_decision`, which persists
    the whole `CommitBundle` (episode, candidate, evidence, role outputs, decision, and the
    resulting state changes) in ONE transaction, all or nothing, idempotent by
    `decision.decision_id`. An inconsistent bundle raises `CommitRejected` and writes nothing.
    Previous claim versions are never deleted. See docs/decisions/0001.
    """

    def search_related_claims(self, claim: CandidateClaim) -> list[ExistingClaim]: ...

    def get_claim(self, claim_id: UUID) -> ExistingClaim | None: ...

    def get_claim_history(self, claim_id: UUID) -> list[ExistingClaim]: ...

    def get_bundle(self, decision_id: UUID) -> CommitBundle | None:
        """Return the stored bundle for a committed decision (used to answer replays)."""
        ...

    def apply_decision(self, bundle: CommitBundle) -> MemoryDelta: ...

    def get_audit_trace(self, candidate_claim_id: UUID) -> AuditTrace: ...
