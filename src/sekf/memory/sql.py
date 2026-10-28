"""Persistent SQLAlchemy memory store (SPEC.md section 10.1, ADR 0001, ADR 0004). Owner: Krishna.

`SqlMemoryRepository.apply_decision` is the only write. One bundle is one database
transaction: the idempotency check, the stale-snapshot check, the claim state transitions,
and the audit record all commit together or not at all. Nothing is ever deleted.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self
from uuid import UUID

from sqlalchemy import Connection, Engine, create_engine, event, func, make_url, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from sekf.contracts import (
    AuditEntry,
    AuditTrace,
    CandidateClaim,
    ClaimStatus,
    CommitBundle,
    ConflictType,
    EvidenceRecord,
    ExistingClaim,
    FinalAction,
    JudgeLabel,
    MemoryDelta,
    RelationshipType,
    StatusChange,
    TemporalRelation,
    check_bundle_consistency,
)
from sekf.errors import CommitRejected
from sekf.memory.migrate import run_migrations
from sekf.memory.schema import (
    CandidateRow,
    ClaimEvidenceRow,
    ClaimKeyRow,
    ClaimRelationshipRow,
    ClaimRow,
    DecisionRow,
    EpisodeRow,
    EvidenceRow,
    ReviewQueueRow,
)

_WRITE_OPTION = "sekf_write"
DEFAULT_DB_URL = "sqlite:///./data/sekf.sqlite3"

# Status each action gives the candidate's own claim; None means "no claim row".
_CANDIDATE_STATUS: dict[FinalAction, ClaimStatus | None] = {
    FinalAction.PROMOTE: ClaimStatus.ACTIVE,
    FinalAction.REVISE: ClaimStatus.ACTIVE,
    FinalAction.QUARANTINE: ClaimStatus.QUARANTINED,
    FinalAction.REJECT: ClaimStatus.REJECTED,
    FinalAction.ESCALATE_FOR_REVIEW: ClaimStatus.CANDIDATE,
    FinalAction.ATTACH_EVIDENCE: None,
    FinalAction.KEEP_EPISODIC: None,
}
# A candidate's existing claim may only be re-decided while it is not yet governed knowledge.
_REOPENABLE = {ClaimStatus.CANDIDATE, ClaimStatus.QUARANTINED}
_CONFLICTS = {
    ConflictType.VALUE_CONFLICT,
    ConflictType.SOURCE_CONFLICT,
    ConflictType.TEMPORAL_UPDATE,
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _is_memory_sqlite(url: str) -> bool:
    return url.startswith("sqlite") and (
        url.rstrip("/") in {"sqlite:", "sqlite:/"} or ":memory:" in url
    )


def create_memory_engine(url: str) -> Engine:
    """Create an engine with the settings the memory store relies on.

    SQLite: foreign keys on, and write transactions start with `BEGIN IMMEDIATE` so concurrent
    writers are serialized (reads stay deferred). In-memory SQLite shares one connection.

    Example:
        >>> engine = create_memory_engine("sqlite://")
    """
    if not url.startswith("sqlite"):
        return create_engine(url)
    if _is_memory_sqlite(url):
        engine = create_engine(url, poolclass=StaticPool, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn: Any, _record: Any) -> None:
        dbapi_conn.isolation_level = None  # let SQLAlchemy emit BEGIN itself
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    @event.listens_for(engine, "begin")
    def _on_begin(conn: Connection) -> None:
        if conn.get_execution_options().get(_WRITE_OPTION):
            conn.exec_driver_sql("BEGIN IMMEDIATE")
        else:
            conn.exec_driver_sql("BEGIN")

    return engine


@dataclass
class _DeltaBuilder:
    created: list[UUID] = field(default_factory=list)
    changes: list[StatusChange] = field(default_factory=list)
    attached: list[UUID] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def status(self, claim_id: str, status: ClaimStatus) -> None:
        self.changes.append(StatusChange(claim_id=UUID(claim_id), new_status=status))


class SqlMemoryRepository:
    """`MemoryRepository` backed by SQLAlchemy (SQLite by default).

    Example:
        >>> repo = SqlMemoryRepository.from_url("sqlite://")
        >>> repo.search_related_claims(candidate)  # doctest: +SKIP
    """

    def __init__(self, engine: Engine, *, clock: Callable[[], datetime] = _utcnow) -> None:
        self._engine = engine
        self._clock = clock

    @classmethod
    def from_url(
        cls, url: str, *, migrate: bool = True, clock: Callable[[], datetime] = _utcnow
    ) -> Self:
        """Open (and by default migrate to head) the store at `url`."""
        engine = create_memory_engine(url)
        if migrate:
            with engine.connect() as conn:
                run_migrations(connection=conn)
                conn.commit()
        return cls(engine, clock=clock)

    @property
    def engine(self) -> Engine:
        return self._engine

    # ------------------------------------------------------------------ sessions

    @contextmanager
    def _read(self) -> Iterator[Session]:
        with Session(self._engine) as session:
            yield session

    @contextmanager
    def _write(self) -> Iterator[Session]:
        with self._engine.connect() as conn:
            conn.execution_options(**{_WRITE_OPTION: True})
            with Session(bind=conn) as session, session.begin():
                yield session

    # --------------------------------------------------------------------- reads

    def search_related_claims(self, claim: CandidateClaim) -> list[ExistingClaim]:
        """Every stored claim (any status) with the candidate's subject and predicate keys."""
        with self._read() as s:
            return self._related(s, claim)

    def get_claim(self, claim_id: UUID) -> ExistingClaim | None:
        with self._read() as s:
            row = s.get(ClaimRow, str(claim_id))
            return self._to_models(s, [row])[0] if row else None

    def get_claim_history(self, claim_id: UUID) -> list[ExistingClaim]:
        """The claim followed by every version it superseded, newest first."""
        with self._read() as s:
            rows: list[ClaimRow] = []
            seen: set[str] = set()
            current = s.get(ClaimRow, str(claim_id))
            while current is not None and current.claim_id not in seen:
                seen.add(current.claim_id)
                rows.append(current)
                nxt = current.supersedes_claim_id
                current = s.get(ClaimRow, nxt) if nxt else None
            return self._to_models(s, rows)

    def get_bundle(self, decision_id: UUID) -> CommitBundle | None:
        with self._read() as s:
            row = s.get(DecisionRow, str(decision_id))
            return CommitBundle.model_validate_json(row.bundle) if row else None

    def get_audit_trace(self, candidate_claim_id: UUID) -> AuditTrace:
        with self._read() as s:
            rows = s.scalars(
                select(DecisionRow)
                .where(DecisionRow.candidate_claim_id == str(candidate_claim_id))
                .order_by(DecisionRow.seq)
            ).all()
            return AuditTrace(
                candidate_claim_id=candidate_claim_id,
                entries=[
                    AuditEntry(
                        bundle=CommitBundle.model_validate_json(r.bundle),
                        memory_delta=MemoryDelta.model_validate_json(r.memory_delta),
                    )
                    for r in rows
                ],
            )

    def list_review_queue(self) -> list[tuple[UUID, UUID, str]]:
        """Escalated items as `(decision_id, claim_id, reason)`, oldest first."""
        with self._read() as s:
            rows = s.scalars(select(ReviewQueueRow).order_by(ReviewQueueRow.created_at)).all()
            return [(UUID(r.decision_id), UUID(r.claim_id), r.reason) for r in rows]

    def get_relationships(self, claim_id: UUID) -> list[tuple[UUID, UUID, RelationshipType]]:
        """Edges touching `claim_id` as `(from_id, to_id, type)`, in insertion order."""
        cid = str(claim_id)
        with self._read() as s:
            rows = s.scalars(
                select(ClaimRelationshipRow)
                .where((ClaimRelationshipRow.from_id == cid) | (ClaimRelationshipRow.to_id == cid))
                .order_by(ClaimRelationshipRow.relationship_id)
            ).all()
            return [
                (UUID(r.from_id), UUID(r.to_id), RelationshipType(r.relationship_type))
                for r in rows
            ]

    def decision_count(self) -> int:
        """Number of committed decisions (test and audit helper)."""
        with self._read() as s:
            return int(s.scalar(select(func.count()).select_from(DecisionRow)) or 0)

    # ---------------------------------------------------------------------- seed

    def seed_claims(
        self, claims: Sequence[ExistingClaim], evidence: Sequence[EvidenceRecord] = ()
    ) -> None:
        """Load initial memory (dataset `initial_memory`, test fixtures). Not a governed write.

        Claims are inserted as given; a claim ID that already exists is an error.
        """
        with self._write() as s:
            for ev in evidence:
                self._store_evidence(s, ev)
            for c in claims:
                if s.get(ClaimRow, str(c.claim_id)) is not None:
                    raise ValueError(f"claim {c.claim_id} already exists")
                s.add(
                    ClaimRow(
                        claim_id=str(c.claim_id),
                        claim_text=c.claim_text,
                        subject_key=c.subject_key,
                        predicate_key=c.predicate_key,
                        object_text=c.object_text,
                        status=c.status.value,
                        version=c.version,
                        valid_from=c.valid_from,
                        valid_to=c.valid_to,
                        confidence=c.confidence,
                        created_at=c.created_at,
                        updated_at=c.updated_at,
                        supersedes_claim_id=(
                            str(c.supersedes_claim_id) if c.supersedes_claim_id else None
                        ),
                        created_by_decision_id=None,
                        row_version=1,
                    )
                )
                s.flush()
                for seq, eid in enumerate(c.evidence_ids):
                    s.add(
                        ClaimEvidenceRow(
                            claim_id=str(c.claim_id),
                            evidence_id=str(eid),
                            decision_id=None,
                            seq=seq,
                        )
                    )
                key = self._key_row(s, c.subject_key, c.predicate_key)
                key.revision += 1

    # --------------------------------------------------------------------- write

    def apply_decision(self, bundle: CommitBundle) -> MemoryDelta:
        """Persist the bundle and apply its state transitions in one transaction.

        Raises `CommitRejected` (and writes nothing) for an inconsistent bundle, a reused
        `decision_id` with different inputs, a stale memory snapshot, or an illegal transition.
        """
        try:
            check_bundle_consistency(bundle)
        except ValueError as exc:
            raise CommitRejected(str(exc)) from exc
        _check_decision_shape(bundle)

        did = str(bundle.decision.decision_id)
        try:
            with self._write() as s:
                replay = self._replay(s, bundle)
                if replay is not None:
                    return replay
                key = self._key_row(s, bundle.candidate.subject_key, bundle.candidate.predicate_key)
                key.revision += 1  # serializes commits on this key (row lock outside SQLite)
                s.flush()
                current = self._related(s, bundle.candidate)
                _check_snapshot(current, bundle.memory_before)
                return self._commit(s, bundle, current)
        except IntegrityError as exc:
            # Lost a race: another transaction committed the same decision or the same key.
            stored = self.get_bundle(UUID(did))
            if stored is not None and stored.decision.request_hash == bundle.decision.request_hash:
                return _replayed(bundle)
            raise CommitRejected(
                f"concurrent commit conflict; retry with a fresh read: {exc.orig}"
            ) from exc
        except OperationalError as exc:
            raise CommitRejected(
                f"memory store unavailable; nothing was written: {exc.orig}"
            ) from exc

    # ------------------------------------------------------------------ internals

    def _replay(self, s: Session, bundle: CommitBundle) -> MemoryDelta | None:
        stored = s.get(DecisionRow, str(bundle.decision.decision_id))
        if stored is None:
            return None
        if stored.request_hash != bundle.decision.request_hash:
            raise CommitRejected("decision_id already used for different inputs")
        return _replayed(bundle)

    def _commit(
        self, s: Session, bundle: CommitBundle, current: list[ExistingClaim]
    ) -> MemoryDelta:
        d, cand = bundle.decision, bundle.candidate
        did = str(d.decision_id)
        at = d.created_at
        action = d.final_action
        delta = _DeltaBuilder()

        # The decision row goes first so every link below can reference it.
        seq = int(s.scalar(select(func.coalesce(func.max(DecisionRow.seq), 0))) or 0) + 1
        decision_row = DecisionRow(
            decision_id=did,
            seq=seq,
            candidate_claim_id=str(cand.claim_id),
            request_hash=d.request_hash,
            final_action=action.value,
            bundle=bundle.model_dump_json(),
            memory_delta="{}",
            committed_at=self._clock(),
        )
        s.add(decision_row)
        s.flush()

        if bundle.episode is not None:
            self._store_episode(s, bundle)
        s.add(
            CandidateRow(
                decision_id=did, claim_id=str(cand.claim_id), payload=cand.model_dump_json()
            )
        )
        for ev in bundle.evidence:
            self._store_evidence(s, ev)
        evidence_ids = [str(e.evidence_id) for e in bundle.evidence]
        by_id = {str(c.claim_id): c for c in current}

        if action is FinalAction.REVISE:
            target_id = str(d.previous_claim_ids[0])
            target = _require_active(by_id, target_id, action)
            row = self._claim_row(s, target_id)
            row.status = ClaimStatus.SUPERSEDED.value
            row.updated_at = at
            row.row_version += 1
            delta.status(target_id, ClaimStatus.SUPERSEDED)
            self._upsert_candidate_claim(
                s,
                bundle,
                ClaimStatus.ACTIVE,
                delta,
                version=target.version + 1,
                supersedes=target_id,
            )
            self._link_evidence(s, str(cand.claim_id), evidence_ids, did)
            self._relate(s, str(cand.claim_id), target_id, RelationshipType.SUPERSEDES, did, at)

        elif action is FinalAction.ATTACH_EVIDENCE:
            target_id = str(d.resulting_claim_id)
            _require_active(by_id, target_id, action)
            new = self._link_evidence(s, target_id, evidence_ids, did)
            delta.attached.extend(UUID(e) for e in new)
            if new:
                row = self._claim_row(s, target_id)
                row.updated_at = at
                row.row_version += 1
            else:
                delta.notes.append("evidence already attached")
            self._relate(s, str(cand.claim_id), target_id, RelationshipType.DUPLICATES, did, at)

        elif action is FinalAction.KEEP_EPISODIC:
            delta.notes.append("kept as episode/candidate only; no claim state changed")

        else:  # PROMOTE, QUARANTINE, REJECT, ESCALATE_FOR_REVIEW
            status = _CANDIDATE_STATUS[action]
            if status is None:  # unreachable: every remaining action creates a claim
                raise CommitRejected(f"no claim status defined for {action.value}")
            if self._upsert_candidate_claim(s, bundle, status, delta):
                self._link_evidence(s, str(cand.claim_id), evidence_ids, did)
            if action is FinalAction.QUARANTINE:
                for target_id in _contradicted(bundle, by_id):
                    self._relate(
                        s, str(cand.claim_id), target_id, RelationshipType.CONTRADICTS, did, at
                    )
            if action is FinalAction.ESCALATE_FOR_REVIEW:
                s.add(
                    ReviewQueueRow(
                        decision_id=did,
                        claim_id=str(cand.claim_id),
                        reason=d.explanation or "escalated for review",
                        created_at=at,
                    )
                )

        result = MemoryDelta(
            decision_id=d.decision_id,
            applied=bool(delta.created or delta.changes or delta.attached),
            created_claim_ids=delta.created,
            status_changes=delta.changes,
            attached_evidence_ids=delta.attached,
            note="; ".join(delta.notes) or f"{action.value} applied",
        )
        decision_row.memory_delta = result.model_dump_json()
        s.flush()
        return result.model_copy(deep=True)

    def _upsert_candidate_claim(
        self,
        s: Session,
        bundle: CommitBundle,
        status: ClaimStatus,
        delta: _DeltaBuilder,
        *,
        version: int = 1,
        supersedes: str | None = None,
    ) -> bool:
        """Create or re-decide the candidate's own claim.

        Returns True when this decision set the claim's status (so its evidence belongs to it),
        False when an escalation left an existing claim as it was.
        """
        d, cand = bundle.decision, bundle.candidate
        cid = str(cand.claim_id)
        confidence = _claim_confidence(bundle)
        existing = s.get(ClaimRow, cid)
        if existing is None:
            s.add(
                ClaimRow(
                    claim_id=cid,
                    claim_text=cand.claim_text,
                    subject_key=cand.subject_key,
                    predicate_key=cand.predicate_key,
                    object_text=cand.object_text,
                    status=status.value,
                    version=version,
                    valid_from=cand.valid_from,
                    valid_to=cand.valid_to,
                    confidence=confidence,
                    created_at=d.created_at,
                    updated_at=d.created_at,
                    supersedes_claim_id=supersedes,
                    created_by_decision_id=str(d.decision_id),
                    row_version=1,
                )
            )
            s.flush()
            delta.created.append(cand.claim_id)
            delta.status(cid, status)
            return True

        current = ClaimStatus(existing.status)
        if (existing.subject_key, existing.predicate_key) != (cand.subject_key, cand.predicate_key):
            raise CommitRejected(f"claim {cid} exists under different subject/predicate keys")
        stored = (existing.claim_text, existing.object_text, existing.valid_from, existing.valid_to)
        if stored != (cand.claim_text, cand.object_text, cand.valid_from, cand.valid_to):
            raise CommitRejected(f"claim {cid} already stored with different content")
        if d.final_action is FinalAction.ESCALATE_FOR_REVIEW:
            delta.notes.append(f"claim {cid} already stored as {current.value}; flagged for review")
            return False
        if current not in _REOPENABLE:
            raise CommitRejected(
                f"claim {cid} is already {current.value}; it cannot become {status.value}"
            )
        existing.status = status.value
        existing.confidence = confidence
        existing.updated_at = d.created_at
        existing.version = max(existing.version, version)
        if supersedes is not None:
            existing.supersedes_claim_id = supersedes
        existing.row_version += 1
        delta.status(cid, status)
        return True

    @staticmethod
    def _claim_row(s: Session, claim_id: str) -> ClaimRow:
        row = s.get(ClaimRow, claim_id)
        if row is None:  # guarded by the snapshot check; kept as a hard failure
            raise CommitRejected(f"claim {claim_id} not found")
        return row

    @staticmethod
    def _key_row(s: Session, subject: str, predicate: str) -> ClaimKeyRow:
        stmt = (
            select(ClaimKeyRow)
            .where(ClaimKeyRow.subject_key == subject, ClaimKeyRow.predicate_key == predicate)
            .with_for_update()
        )
        row = s.scalars(stmt).first()
        if row is None:
            row = ClaimKeyRow(subject_key=subject, predicate_key=predicate, revision=0)
            s.add(row)
            s.flush()
        return row

    @staticmethod
    def _store_evidence(s: Session, ev: EvidenceRecord) -> None:
        stored = s.get(EvidenceRow, str(ev.evidence_id))
        if stored is not None:
            if _evidence_model(stored) != ev:
                raise CommitRejected(
                    f"evidence {ev.evidence_id} already stored with different contents"
                )
            return
        s.add(
            EvidenceRow(
                evidence_id=str(ev.evidence_id),
                content=ev.content,
                source_id=ev.source_id,
                source_type=ev.source_type,
                published_at=ev.published_at,
                retrieved_at=ev.retrieved_at,
                reliability=ev.reliability,
                content_hash=ev.content_hash,
                metadata_json=dict(ev.metadata),
            )
        )
        s.flush()

    @staticmethod
    def _store_episode(s: Session, bundle: CommitBundle) -> None:
        ep = bundle.episode
        if ep is None:
            return
        stored = s.get(EpisodeRow, str(ep.episode_id))
        if stored is not None:
            same = (
                stored.content == ep.content
                and stored.source_id == ep.source_id
                and stored.observed_at == ep.observed_at
                and stored.metadata_json == ep.metadata
            )
            if not same:
                raise CommitRejected(
                    f"episode {ep.episode_id} already stored with different contents"
                )
            return
        s.add(
            EpisodeRow(
                episode_id=str(ep.episode_id),
                content=ep.content,
                source_id=ep.source_id,
                observed_at=ep.observed_at,
                metadata_json=dict(ep.metadata),
            )
        )

    @staticmethod
    def _link_evidence(s: Session, claim_id: str, evidence_ids: list[str], did: str) -> list[str]:
        """Link evidence to a claim; return the IDs that were not linked before."""
        linked = set(
            s.scalars(
                select(ClaimEvidenceRow.evidence_id).where(ClaimEvidenceRow.claim_id == claim_id)
            )
        )
        last = s.scalar(
            select(func.max(ClaimEvidenceRow.seq)).where(ClaimEvidenceRow.claim_id == claim_id)
        )
        seq = -1 if last is None else int(last)
        new: list[str] = []
        for eid in evidence_ids:
            if eid in linked:
                continue
            seq += 1
            s.add(ClaimEvidenceRow(claim_id=claim_id, evidence_id=eid, decision_id=did, seq=seq))
            linked.add(eid)
            new.append(eid)
        s.flush()
        return new

    @staticmethod
    def _relate(
        s: Session, from_id: str, to_id: str, kind: RelationshipType, did: str, at: datetime
    ) -> None:
        s.add(
            ClaimRelationshipRow(
                from_id=from_id,
                to_id=to_id,
                relationship_type=kind.value,
                decision_id=did,
                created_at=at,
            )
        )

    def _related(self, s: Session, claim: CandidateClaim) -> list[ExistingClaim]:
        rows = s.scalars(
            select(ClaimRow)
            .where(
                ClaimRow.subject_key == claim.subject_key,
                ClaimRow.predicate_key == claim.predicate_key,
                ClaimRow.claim_id != str(claim.claim_id),
            )
            .order_by(ClaimRow.created_at, ClaimRow.claim_id)
        ).all()
        return self._to_models(s, list(rows))

    @staticmethod
    def _to_models(s: Session, rows: list[ClaimRow]) -> list[ExistingClaim]:
        if not rows:
            return []
        ids = [r.claim_id for r in rows]
        links = s.execute(
            select(ClaimEvidenceRow.claim_id, ClaimEvidenceRow.evidence_id)
            .where(ClaimEvidenceRow.claim_id.in_(ids))
            .order_by(ClaimEvidenceRow.claim_id, ClaimEvidenceRow.seq)
        ).all()
        evidence: dict[str, list[UUID]] = {
            cid: [UUID(e) for _, e in group]
            for cid, group in itertools.groupby(links, key=lambda x: x[0])
        }
        return [
            ExistingClaim(
                claim_id=UUID(r.claim_id),
                claim_text=r.claim_text,
                subject_key=r.subject_key,
                predicate_key=r.predicate_key,
                object_text=r.object_text,
                status=ClaimStatus(r.status),
                version=r.version,
                valid_from=r.valid_from,
                valid_to=r.valid_to,
                confidence=r.confidence,
                created_at=r.created_at,
                updated_at=r.updated_at,
                supersedes_claim_id=UUID(r.supersedes_claim_id) if r.supersedes_claim_id else None,
                evidence_ids=evidence.get(r.claim_id, []),
            )
            for r in rows
        ]


# ----------------------------------------------------------------------- pure checks


def _replayed(bundle: CommitBundle) -> MemoryDelta:
    return MemoryDelta(
        decision_id=bundle.decision.decision_id,
        applied=False,
        replayed=True,
        note="replay: already recorded",
    )


def _claim_confidence(bundle: CommitBundle) -> float:
    """Confidence that the stored claim is true.

    SUPPORTED: the Judge's confidence. REFUTED: its complement. Otherwise (no Judge, or no
    verdict on truth) the extractor's confidence.
    """
    j = bundle.judge
    if j is not None and j.label is JudgeLabel.SUPPORTED:
        return j.confidence
    if j is not None and j.label is JudgeLabel.REFUTED:
        return round(1.0 - j.confidence, 6)
    return bundle.candidate.extractor_confidence


def _evidence_model(row: EvidenceRow) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=UUID(row.evidence_id),
        content=row.content,
        source_id=row.source_id,
        source_type=row.source_type,
        published_at=row.published_at,
        retrieved_at=row.retrieved_at,
        reliability=row.reliability,
        content_hash=row.content_hash,
        metadata=row.metadata_json,
    )


def _check_decision_shape(bundle: CommitBundle) -> None:
    """The decision's claim links must match its action (same shape as `decision_links`)."""
    d = bundle.decision
    cand = bundle.candidate.claim_id
    prev, res = d.previous_claim_ids, d.resulting_claim_id
    action = d.final_action
    ok: bool
    if action in (FinalAction.KEEP_EPISODIC, FinalAction.ESCALATE_FOR_REVIEW):
        ok = prev == [] and res is None
    elif action in (FinalAction.PROMOTE, FinalAction.QUARANTINE, FinalAction.REJECT):
        ok = prev == [] and res == cand
    elif action is FinalAction.REVISE:
        ok = len(prev) == 1 and prev[0] != cand and res == cand
    else:  # ATTACH_EVIDENCE
        ok = len(prev) == 1 and res == prev[0] and res != cand
    if not ok:
        raise CommitRejected(
            f"{action.value} decision has inconsistent claim links: "
            f"previous_claim_ids={[str(p) for p in prev]}, resulting_claim_id={res}"
        )


def _check_snapshot(current: list[ExistingClaim], memory_before: list[ExistingClaim]) -> None:
    now = {c.claim_id: c for c in current}
    before = {c.claim_id: c for c in memory_before}
    if now != before:
        changed = sorted(
            str(cid) for cid in now.keys() | before.keys() if now.get(cid) != before.get(cid)
        )
        raise CommitRejected(f"memory changed since the decision was made: {changed}")


def _require_active(
    by_id: dict[str, ExistingClaim], claim_id: str, action: FinalAction
) -> ExistingClaim:
    claim = by_id.get(claim_id)
    if claim is None or claim.status is not ClaimStatus.ACTIVE:
        raise CommitRejected(f"{action.value} target {claim_id} is not an ACTIVE related claim")
    return claim


def _contradicted(bundle: CommitBundle, by_id: dict[str, ExistingClaim]) -> list[str]:
    """ACTIVE related claims a quarantined candidate conflicts with, per the Historian."""
    h = bundle.historian
    if h is None:
        return []
    conflicted = (
        h.conflict_type in _CONFLICTS
        or h.temporal_relation is TemporalRelation.OVERLAPPING_VALIDITY
    )
    if not conflicted:
        return []
    out: list[str] = []
    for cid in h.related_claim_ids:
        claim = by_id.get(str(cid))
        if claim is not None and claim.status is ClaimStatus.ACTIVE and str(cid) not in out:
            out.append(str(cid))
    return out


def build_memory_repository(url: str | None = None) -> SqlMemoryRepository:
    """Open the persistent store at `url` or `SEKF_DB_URL` (default ./data/sekf.sqlite3).

    Creates the parent directory of a file-based SQLite database and migrates to head.

    Example:
        >>> repo = build_memory_repository("sqlite://")
    """
    resolved = url or os.environ.get("SEKF_DB_URL") or DEFAULT_DB_URL
    if resolved.startswith("sqlite") and not _is_memory_sqlite(resolved):
        path = make_url(resolved).database
        if path:
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    return SqlMemoryRepository.from_url(resolved)
