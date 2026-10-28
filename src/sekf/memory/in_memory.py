"""Throwaway in-memory repository so the skeleton runs end to end.

Krishna replaces this with the SQLAlchemy implementation in Sprint 1. It deliberately does
NOT perform claim state transitions: `apply_decision` only records the bundle, so the
skeleton milestone ("no memory mutation") holds. It does model the commit contract: the
bundle is validated first, then written all-or-nothing, idempotently by `decision_id`.
"""

from uuid import UUID

from sekf.contracts import (
    AuditEntry,
    AuditTrace,
    CandidateClaim,
    CommitBundle,
    EpisodeRecord,
    EvidenceRecord,
    ExistingClaim,
    MemoryDelta,
    check_bundle_consistency,
)
from sekf.errors import CommitRejected


class InMemoryRepository:
    def __init__(self, claims: list[ExistingClaim] | None = None) -> None:
        self._claims: dict[UUID, ExistingClaim] = {
            c.claim_id: c.model_copy(deep=True) for c in claims or []
        }
        self._episodes: dict[UUID, EpisodeRecord] = {}
        self._candidates: dict[UUID, CandidateClaim] = {}
        self._evidence: dict[UUID, EvidenceRecord] = {}
        self._bundles: dict[UUID, CommitBundle] = {}
        self._deltas: dict[UUID, MemoryDelta] = {}

    def seed_claim(self, claim: ExistingClaim) -> None:
        """Test helper: place an existing claim into memory (stored as a copy)."""
        self._claims[claim.claim_id] = claim.model_copy(deep=True)

    def search_related_claims(self, claim: CandidateClaim) -> list[ExistingClaim]:
        return [
            c.model_copy(deep=True)  # callers must not be able to edit stored memory
            for c in self._claims.values()
            if c.subject_key == claim.subject_key
            and c.predicate_key == claim.predicate_key
            and c.claim_id != claim.claim_id
        ]

    def get_claim(self, claim_id: UUID) -> ExistingClaim | None:
        claim = self._claims.get(claim_id)
        return claim.model_copy(deep=True) if claim else None

    def get_claim_history(self, claim_id: UUID) -> list[ExistingClaim]:
        history: list[ExistingClaim] = []
        visited: set[UUID] = set()
        current = self._claims.get(claim_id)
        while current is not None and current.claim_id not in visited:
            visited.add(current.claim_id)
            history.append(current.model_copy(deep=True))
            current = (
                self._claims.get(current.supersedes_claim_id)
                if current.supersedes_claim_id
                else None
            )
        return history

    def get_bundle(self, decision_id: UUID) -> CommitBundle | None:
        bundle = self._bundles.get(decision_id)
        return bundle.model_copy(deep=True) if bundle else None  # callers cannot edit history

    def apply_decision(self, bundle: CommitBundle) -> MemoryDelta:
        decision = bundle.decision
        try:
            check_bundle_consistency(bundle)
        except ValueError as exc:
            raise CommitRejected(str(exc)) from exc

        stored = self._bundles.get(decision.decision_id)
        if stored is not None:
            # A replay needs identical stable inputs; the first committed decision wins and
            # role outputs may legitimately differ (LLM nondeterminism).
            if stored.decision.request_hash != decision.request_hash:
                raise CommitRejected("decision_id already used for different inputs")
            return MemoryDelta(
                decision_id=decision.decision_id,
                applied=False,
                replayed=True,
                note="replay: already recorded",
            )

        # Optimistic concurrency, including "phantoms": the complete related-claim set must be
        # exactly what the decision reasoned over. A claim that was changed, removed, OR newly
        # added since the snapshot was read makes the decision stale.
        current = {c.claim_id: c for c in self.search_related_claims(bundle.candidate)}
        snapshot = {c.claim_id: c for c in bundle.memory_before}
        if current != snapshot:
            changed = sorted(
                str(cid)
                for cid in current.keys() | snapshot.keys()
                if current.get(cid) != snapshot.get(cid)
            )
            raise CommitRejected(f"memory changed since the decision was made: {changed}")

        # All checks passed: write everything.
        delta = MemoryDelta(
            decision_id=decision.decision_id,
            applied=False,
            note="skeleton repository: bundle recorded, no claim state transition",
        )
        stored_bundle = bundle.model_copy(deep=True)
        if stored_bundle.episode:
            self._episodes[stored_bundle.episode.episode_id] = stored_bundle.episode
        self._candidates[stored_bundle.candidate.claim_id] = stored_bundle.candidate
        for ev in stored_bundle.evidence:
            self._evidence[ev.evidence_id] = ev
        self._bundles[decision.decision_id] = stored_bundle
        self._deltas[decision.decision_id] = delta.model_copy(deep=True)
        return delta

    def get_audit_trace(self, candidate_claim_id: UUID) -> AuditTrace:
        return AuditTrace(
            candidate_claim_id=candidate_claim_id,
            entries=[
                AuditEntry(
                    bundle=b.model_copy(deep=True),
                    memory_delta=self._deltas[did].model_copy(deep=True),
                )
                for did, b in self._bundles.items()
                if b.candidate.claim_id == candidate_claim_id
            ],
        )

    def decision_count(self) -> int:
        """Test helper."""
        return len(self._bundles)

    def candidate_count(self) -> int:
        """Test helper."""
        return len(self._candidates)

    def evidence_count(self) -> int:
        """Test helper."""
        return len(self._evidence)
