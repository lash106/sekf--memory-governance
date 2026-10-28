"""SQLAlchemy models for the persistent memory store. Owner: Krishna.

Design and rationale: docs/decisions/0004-memory-store-schema.md. Schema changes need a new
Alembic revision in `sekf/memory/migrations/versions/`; a test checks the two agree.

UUIDs are stored as 36-character strings and timestamps as timezone-aware UTC values so the
store behaves the same on SQLite and PostgreSQL.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

UUID_LEN = 36

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class UtcDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC timestamps on every backend.

    SQLite drops tzinfo, so values are normalized to UTC on the way in and tagged UTC on the
    way out. Naive datetimes are rejected: SPEC.md section 9 requires aware UTC timestamps.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime; memory timestamps must be timezone-aware")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {datetime: UtcDateTime, dict[str, Any]: JSON}


class ClaimRow(Base):
    """A governed claim (mirrors `ExistingClaim`). Rows are never deleted."""

    __tablename__ = "claims"

    claim_id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True)
    claim_text: Mapped[str] = mapped_column(Text)
    subject_key: Mapped[str] = mapped_column(String(255))
    predicate_key: Mapped[str] = mapped_column(String(255))
    object_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))
    version: Mapped[int] = mapped_column(Integer)
    valid_from: Mapped[datetime | None]
    valid_to: Mapped[datetime | None]
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    supersedes_claim_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("claims.claim_id")
    )
    created_by_decision_id: Mapped[str | None] = mapped_column(String(UUID_LEN))
    # Optimistic-lock counter: bumped on every change to this row.
    row_version: Mapped[int] = mapped_column(Integer, default=1)

    __table_args__ = (Index(None, "subject_key", "predicate_key"),)


class EvidenceRow(Base):
    """Immutable evidence record (mirrors `EvidenceRecord`)."""

    __tablename__ = "evidence"

    evidence_id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    source_id: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(64))
    published_at: Mapped[datetime | None]
    retrieved_at: Mapped[datetime]
    reliability: Mapped[float] = mapped_column(Float)
    content_hash: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON)


class ClaimEvidenceRow(Base):
    """Evidence linked to a claim, in attach order."""

    __tablename__ = "claim_evidence"

    claim_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("claims.claim_id"), primary_key=True
    )
    # No foreign key: seeded initial memory may cite evidence whose record is not stored.
    evidence_id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True)
    # None for links that came with seeded initial memory rather than a decision.
    decision_id: Mapped[str | None] = mapped_column(
        String(UUID_LEN), ForeignKey("decisions.decision_id")
    )
    seq: Mapped[int] = mapped_column(Integer)


class ClaimRelationshipRow(Base):
    """Typed edge between claims (`RelationshipType`).

    `from_id` may be a candidate that never became a claim (ATTACH_EVIDENCE), so it carries no
    foreign key.
    """

    __tablename__ = "claim_relationships"

    relationship_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    from_id: Mapped[str] = mapped_column(String(UUID_LEN), index=True)
    to_id: Mapped[str] = mapped_column(String(UUID_LEN), ForeignKey("claims.claim_id"), index=True)
    relationship_type: Mapped[str] = mapped_column(String(16))
    decision_id: Mapped[str] = mapped_column(String(UUID_LEN), ForeignKey("decisions.decision_id"))
    created_at: Mapped[datetime]


class EpisodeRow(Base):
    __tablename__ = "episodes"

    episode_id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    source_id: Mapped[str] = mapped_column(String(255))
    observed_at: Mapped[datetime]
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON)


class CandidateRow(Base):
    """Raw candidate as submitted, one row per committed decision.

    Keyed by decision: the same candidate may be re-evaluated with a new `idempotency_key`.
    """

    __tablename__ = "candidates"

    decision_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("decisions.decision_id"), primary_key=True
    )
    claim_id: Mapped[str] = mapped_column(String(UUID_LEN), index=True)
    payload: Mapped[str] = mapped_column(Text)  # CandidateClaim JSON


class DecisionRow(Base):
    """One committed decision: the full bundle and the delta it produced. Append-only."""

    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(String(UUID_LEN), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, unique=True)
    candidate_claim_id: Mapped[str] = mapped_column(String(UUID_LEN), index=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    final_action: Mapped[str] = mapped_column(String(32))
    bundle: Mapped[str] = mapped_column(Text)  # CommitBundle JSON
    memory_delta: Mapped[str] = mapped_column(Text)  # MemoryDelta JSON
    committed_at: Mapped[datetime]


class ReviewQueueRow(Base):
    """Claims escalated for human review."""

    __tablename__ = "review_queue"

    decision_id: Mapped[str] = mapped_column(
        String(UUID_LEN), ForeignKey("decisions.decision_id"), primary_key=True
    )
    claim_id: Mapped[str] = mapped_column(String(UUID_LEN), index=True)
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime]


class ClaimKeyRow(Base):
    """Per (subject_key, predicate_key) lock row; `revision` bumps on every commit for the key."""

    __tablename__ = "claim_keys"

    subject_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    predicate_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
