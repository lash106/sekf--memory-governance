"""SQLite-backed memory repository. Owner: Krishna."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import ForeignKey, Index, String, Text, create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from sekf.contracts import (
    AuditEntry,
    AuditTrace,
    CandidateClaim,
    ClaimStatus,
    CommitBundle,
    EpisodeRecord,
    EvidenceRecord,
    ExistingClaim,
    FinalAction,
    MemoryDelta,
    StatusChange,
    check_bundle_consistency,
)
from sekf.errors import CommitRejected


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for memory persistence."""


class ClaimRow(Base):
    __tablename__ = "claims"

    claim_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    subject_key: Mapped[str] = mapped_column(String, index=True)
    predicate_key: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    version: Mapped[int]
    supersedes_claim_id: Mapped[str | None] = mapped_column(String(36))
    claim_json: Mapped[str] = mapped_column(Text)


Index(
    "uq_claims_one_active_per_key",
    ClaimRow.subject_key,
    ClaimRow.predicate_key,
    unique=True,
    sqlite_where=ClaimRow.status == ClaimStatus.ACTIVE.value,
)


class EvidenceRow(Base):
    __tablename__ = "evidence"

    evidence_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    evidence_json: Mapped[str] = mapped_column(Text)


class ClaimEvidenceRow(Base):
    __tablename__ = "claim_evidence"

    claim_id: Mapped[str] = mapped_column(
        ForeignKey("claims.claim_id", ondelete="RESTRICT"), primary_key=True
    )
    evidence_id: Mapped[str] = mapped_column(
        ForeignKey("evidence.evidence_id", ondelete="RESTRICT"), primary_key=True
    )


class RelationshipRow(Base):
    __tablename__ = "claim_relationships"

    source_claim_id: Mapped[str] = mapped_column(
        ForeignKey("claims.claim_id", ondelete="RESTRICT"), primary_key=True
    )
    relationship_type: Mapped[str] = mapped_column(String(32), primary_key=True)
    target_claim_id: Mapped[str] = mapped_column(
        ForeignKey("claims.claim_id", ondelete="RESTRICT"), primary_key=True
    )


class EpisodeRow(Base):
    __tablename__ = "episodes"

    episode_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    episode_json: Mapped[str] = mapped_column(Text)


class DecisionRow(Base):
    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_claim_id: Mapped[str] = mapped_column(String(36), index=True)
    created_at: Mapped[datetime] = mapped_column(index=True)
    bundle_json: Mapped[str] = mapped_column(Text, nullable=False)
    delta_json: Mapped[str] = mapped_column(Text, nullable=False)


class SqlAlchemyRepository:
    """Persist claims and audit bundles atomically in a local SQLite database.

    The default in-memory database is useful for tests. File databases should be passed as
    ``sqlite:///path/to/memory.db``. SQLite's ``BEGIN IMMEDIATE`` serializes writers so the
    related-claim snapshot check also rejects concurrent phantom insertions.
    """

    def __init__(self, database_url: str | Engine = "sqlite:///:memory:") -> None:
        if isinstance(database_url, Engine):
            self._engine = database_url
        else:
            options: dict[str, Any] = {"future": True}
            if database_url in {"sqlite://", "sqlite:///:memory:"}:
                options["connect_args"] = {"check_same_thread": False}
                options["poolclass"] = StaticPool
            else:
                options["connect_args"] = {"check_same_thread": False, "timeout": 30}
            self._engine = create_engine(database_url, **options)
        if self._engine.dialect.name != "sqlite":
            raise ValueError("SqlAlchemyRepository currently supports SQLite only")
        event.listen(self._engine, "connect", self._enable_foreign_keys)
        self._sessions = sessionmaker(self._engine, expire_on_commit=False)
        Base.metadata.create_all(self._engine)

    @staticmethod
    def _enable_foreign_keys(connection: Any, _record: Any) -> None:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def search_related_claims(self, claim: CandidateClaim) -> list[ExistingClaim]:
        with self._sessions() as session:
            rows = session.scalars(
                select(ClaimRow)
                .where(
                    ClaimRow.subject_key == claim.subject_key,
                    ClaimRow.predicate_key == claim.predicate_key,
                    ClaimRow.claim_id != str(claim.claim_id),
                )
                .order_by(ClaimRow.version, ClaimRow.claim_id)
            )
            return [ExistingClaim.model_validate_json(row.claim_json) for row in rows]

    def get_claim(self, claim_id: UUID) -> ExistingClaim | None:
        with self._sessions() as session:
            row = session.get(ClaimRow, str(claim_id))
            return ExistingClaim.model_validate_json(row.claim_json) if row else None

    def get_claim_history(self, claim_id: UUID) -> list[ExistingClaim]:
        history: list[ExistingClaim] = []
        visited: set[UUID] = set()
        current_id: UUID | None = claim_id
        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            current = self.get_claim(current_id)
            if current is None:
                break
            history.append(current)
            current_id = current.supersedes_claim_id
        return history

    def get_bundle(self, decision_id: UUID) -> CommitBundle | None:
        with self._sessions() as session:
            row = session.get(DecisionRow, str(decision_id))
            return CommitBundle.model_validate_json(row.bundle_json) if row else None

    def apply_decision(self, bundle: CommitBundle) -> MemoryDelta:
        """Validate and commit a complete decision in one serialized transaction."""
        try:
            check_bundle_consistency(bundle)
        except ValueError as exc:
            raise CommitRejected(str(exc)) from exc

        decision = bundle.decision
        try:
            with self._sessions() as session:
                session.connection().exec_driver_sql("BEGIN IMMEDIATE")
                stored = session.get(DecisionRow, str(decision.decision_id))
                if stored is not None:
                    if stored.request_hash != decision.request_hash:
                        raise CommitRejected("decision_id already used for different inputs")
                    delta = MemoryDelta.model_validate_json(stored.delta_json)
                    delta.applied = False
                    delta.replayed = True
                    delta.note = "replay: already recorded"
                    session.rollback()
                    return delta

                self._validate_snapshot(session, bundle)
                self._insert_evidence(session, bundle.evidence)
                delta = self._apply_action(session, bundle)
                if bundle.episode is not None:
                    self._insert_episode(session, bundle.episode)
                session.add(
                    DecisionRow(
                        decision_id=str(decision.decision_id),
                        request_hash=decision.request_hash,
                        candidate_claim_id=str(bundle.candidate.claim_id),
                        created_at=decision.created_at.astimezone(UTC),
                        bundle_json=bundle.model_dump_json(),
                        delta_json=delta.model_dump_json(),
                    )
                )
                session.commit()
                return delta
        except IntegrityError as exc:
            raise CommitRejected(f"memory commit violates a persistence constraint: {exc}") from exc

    def get_audit_trace(self, candidate_claim_id: UUID) -> AuditTrace:
        with self._sessions() as session:
            rows = session.scalars(
                select(DecisionRow)
                .where(DecisionRow.candidate_claim_id == str(candidate_claim_id))
                .order_by(DecisionRow.created_at, DecisionRow.decision_id)
            )
            entries = [
                AuditEntry(
                    bundle=CommitBundle.model_validate_json(row.bundle_json),
                    memory_delta=MemoryDelta.model_validate_json(row.delta_json),
                )
                for row in rows
            ]
            return AuditTrace(candidate_claim_id=candidate_claim_id, entries=entries)

    def seed_claim(self, claim: ExistingClaim) -> None:
        """Insert a claim for initial-memory setup and tests, without replacing an existing ID."""
        try:
            with self._sessions.begin() as session:
                session.connection().exec_driver_sql("BEGIN IMMEDIATE")
                self._insert_claim(session, claim)
                if claim.supersedes_claim_id is not None:
                    session.add(
                        RelationshipRow(
                            source_claim_id=str(claim.claim_id),
                            relationship_type="SUPERSEDES",
                            target_claim_id=str(claim.supersedes_claim_id),
                        )
                    )
        except IntegrityError as exc:
            raise CommitRejected(f"claim could not be seeded: {exc}") from exc

    @staticmethod
    def _validate_snapshot(session: Session, bundle: CommitBundle) -> None:
        candidate = bundle.candidate
        rows = session.scalars(
            select(ClaimRow).where(
                ClaimRow.subject_key == candidate.subject_key,
                ClaimRow.predicate_key == candidate.predicate_key,
                ClaimRow.claim_id != str(candidate.claim_id),
            )
        )
        current = {
            UUID(row.claim_id): ExistingClaim.model_validate_json(row.claim_json) for row in rows
        }
        snapshot = {claim.claim_id: claim for claim in bundle.memory_before}
        if current != snapshot:
            changed = sorted(
                str(claim_id)
                for claim_id in current.keys() | snapshot.keys()
                if current.get(claim_id) != snapshot.get(claim_id)
            )
            raise CommitRejected(f"memory changed since the decision was made: {changed}")

    def _apply_action(self, session: Session, bundle: CommitBundle) -> MemoryDelta:
        candidate = bundle.candidate
        decision = bundle.decision
        delta = MemoryDelta(decision_id=decision.decision_id, applied=False)
        action = decision.final_action

        if action in (
            FinalAction.PROMOTE,
            FinalAction.REVISE,
            FinalAction.QUARANTINE,
            FinalAction.REJECT,
        ):
            status = {
                FinalAction.PROMOTE: ClaimStatus.ACTIVE,
                FinalAction.REVISE: ClaimStatus.ACTIVE,
                FinalAction.QUARANTINE: ClaimStatus.QUARANTINED,
                FinalAction.REJECT: ClaimStatus.REJECTED,
            }[action]
            version = 1 + max(
                (claim.version for claim in bundle.memory_before),
                default=0,
            )
            if action is FinalAction.REVISE:
                previous = self._require_active_target(session, bundle)
                previous_claim = ExistingClaim.model_validate_json(previous.claim_json)
                previous_claim.status = ClaimStatus.SUPERSEDED
                previous_claim.updated_at = decision.created_at
                previous.status = previous_claim.status.value
                previous.claim_json = previous_claim.model_dump_json()
                version = previous_claim.version + 1
                delta.status_changes.append(
                    StatusChange(
                        claim_id=previous_claim.claim_id,
                        new_status=ClaimStatus.SUPERSEDED,
                    )
                )
                session.flush()
                session.add(
                    RelationshipRow(
                        source_claim_id=str(candidate.claim_id),
                        relationship_type="SUPERSEDES",
                        target_claim_id=str(previous_claim.claim_id),
                    )
                )
            elif action is FinalAction.PROMOTE and any(
                claim.status is ClaimStatus.ACTIVE for claim in bundle.memory_before
            ):
                raise CommitRejected("PROMOTE cannot add a second active claim for the same key")
            new_claim = ExistingClaim(
                claim_id=candidate.claim_id,
                claim_text=candidate.claim_text,
                subject_key=candidate.subject_key,
                predicate_key=candidate.predicate_key,
                object_text=candidate.object_text,
                status=status,
                version=version,
                valid_from=candidate.valid_from,
                valid_to=candidate.valid_to,
                confidence=candidate.extractor_confidence,
                created_at=decision.created_at,
                updated_at=decision.created_at,
                supersedes_claim_id=(
                    decision.previous_claim_ids[0]
                    if action is FinalAction.REVISE and decision.previous_claim_ids
                    else None
                ),
                evidence_ids=[e.evidence_id for e in bundle.evidence],
            )
            self._insert_claim(session, new_claim)
            delta.created_claim_ids.append(new_claim.claim_id)
            delta.status_changes.append(
                StatusChange(claim_id=new_claim.claim_id, new_status=status)
            )
            delta.attached_evidence_ids.extend(new_claim.evidence_ids)
            delta.applied = True
        elif action is FinalAction.ATTACH_EVIDENCE:
            target = self._require_active_target(session, bundle)
            existing = ExistingClaim.model_validate_json(target.claim_json)
            attached = list(
                dict.fromkeys([*existing.evidence_ids, *(e.evidence_id for e in bundle.evidence)])
            )
            delta.attached_evidence_ids = [
                item for item in attached if item not in existing.evidence_ids
            ]
            existing.evidence_ids = attached
            existing.updated_at = decision.created_at
            target.claim_json = existing.model_dump_json()
            delta.applied = bool(delta.attached_evidence_ids)

        for evidence_id in delta.attached_evidence_ids:
            session.add(
                ClaimEvidenceRow(
                    claim_id=str(
                        decision.previous_claim_ids[0]
                        if action is FinalAction.ATTACH_EVIDENCE
                        else candidate.claim_id
                    ),
                    evidence_id=str(evidence_id),
                )
            )
        return delta

    @staticmethod
    def _require_active_target(session: Session, bundle: CommitBundle) -> ClaimRow:
        target_ids = bundle.decision.previous_claim_ids
        if len(target_ids) != 1:
            raise CommitRejected(f"{bundle.decision.final_action} requires exactly one prior claim")
        target = session.get(ClaimRow, str(target_ids[0]))
        if target is None or target.status != ClaimStatus.ACTIVE.value:
            raise CommitRejected("decision target must be an existing ACTIVE claim")
        if (
            target.subject_key != bundle.candidate.subject_key
            or target.predicate_key != bundle.candidate.predicate_key
        ):
            raise CommitRejected("decision target is not related to the candidate")
        if target.claim_id not in {str(claim.claim_id) for claim in bundle.memory_before}:
            raise CommitRejected("decision target is missing from memory_before")
        return target

    @staticmethod
    def _insert_claim(session: Session, claim: ExistingClaim) -> None:
        session.add(
            ClaimRow(
                claim_id=str(claim.claim_id),
                subject_key=claim.subject_key,
                predicate_key=claim.predicate_key,
                status=claim.status.value,
                version=claim.version,
                supersedes_claim_id=(
                    str(claim.supersedes_claim_id)
                    if claim.supersedes_claim_id is not None
                    else None
                ),
                claim_json=claim.model_dump_json(),
            )
        )
        for evidence_id in claim.evidence_ids:
            session.add(
                ClaimEvidenceRow(claim_id=str(claim.claim_id), evidence_id=str(evidence_id))
            )

    @staticmethod
    def _insert_episode(session: Session, episode: EpisodeRecord) -> None:
        session.add(
            EpisodeRow(episode_id=str(episode.episode_id), episode_json=episode.model_dump_json())
        )

    @staticmethod
    def _insert_evidence(session: Session, evidence: list[EvidenceRecord]) -> None:
        for item in evidence:
            existing = session.get(EvidenceRow, str(item.evidence_id))
            if existing is not None:
                stored = EvidenceRecord.model_validate_json(existing.evidence_json)
                if stored != item:
                    raise CommitRejected("evidence_id already used for different evidence")
                continue
            session.add(
                EvidenceRow(evidence_id=str(item.evidence_id), evidence_json=item.model_dump_json())
            )
