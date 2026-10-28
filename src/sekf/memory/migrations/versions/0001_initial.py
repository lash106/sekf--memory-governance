"""Initial memory schema (ADR 0004). Never edit after merge; add a new revision instead.

Revision ID: 0001_initial
Revises: (none)
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "claim_keys",
        sa.Column("subject_key", sa.String(length=255), nullable=False),
        sa.Column("predicate_key", sa.String(length=255), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("subject_key", "predicate_key", name=op.f("pk_claim_keys")),
    )
    op.create_table(
        "claims",
        sa.Column("claim_id", sa.String(length=36), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("subject_key", sa.String(length=255), nullable=False),
        sa.Column("predicate_key", sa.String(length=255), nullable=False),
        sa.Column("object_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("supersedes_claim_id", sa.String(length=36), nullable=True),
        sa.Column("created_by_decision_id", sa.String(length=36), nullable=True),
        sa.Column("row_version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["supersedes_claim_id"],
            ["claims.claim_id"],
            name=op.f("fk_claims_supersedes_claim_id_claims"),
        ),
        sa.PrimaryKeyConstraint("claim_id", name=op.f("pk_claims")),
    )
    with op.batch_alter_table("claims", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_claims_subject_key_predicate_key"),
            ["subject_key", "predicate_key"],
            unique=False,
        )

    op.create_table(
        "decisions",
        sa.Column("decision_id", sa.String(length=36), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("candidate_claim_id", sa.String(length=36), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("final_action", sa.String(length=32), nullable=False),
        sa.Column("bundle", sa.Text(), nullable=False),
        sa.Column("memory_delta", sa.Text(), nullable=False),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("decision_id", name=op.f("pk_decisions")),
        sa.UniqueConstraint("seq", name=op.f("uq_decisions_seq")),
    )
    with op.batch_alter_table("decisions", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_decisions_candidate_claim_id"), ["candidate_claim_id"], unique=False
        )

    op.create_table(
        "episodes",
        sa.Column("episode_id", sa.String(length=36), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("episode_id", name=op.f("pk_episodes")),
    )
    op.create_table(
        "evidence",
        sa.Column("evidence_id", sa.String(length=36), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reliability", sa.Float(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("evidence_id", name=op.f("pk_evidence")),
    )
    op.create_table(
        "candidates",
        sa.Column("decision_id", sa.String(length=36), nullable=False),
        sa.Column("claim_id", sa.String(length=36), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.decision_id"],
            name=op.f("fk_candidates_decision_id_decisions"),
        ),
        sa.PrimaryKeyConstraint("decision_id", name=op.f("pk_candidates")),
    )
    with op.batch_alter_table("candidates", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_candidates_claim_id"), ["claim_id"], unique=False)

    op.create_table(
        "claim_evidence",
        sa.Column("claim_id", sa.String(length=36), nullable=False),
        sa.Column("evidence_id", sa.String(length=36), nullable=False),
        sa.Column("decision_id", sa.String(length=36), nullable=True),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["claim_id"], ["claims.claim_id"], name=op.f("fk_claim_evidence_claim_id_claims")
        ),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.decision_id"],
            name=op.f("fk_claim_evidence_decision_id_decisions"),
        ),
        sa.PrimaryKeyConstraint("claim_id", "evidence_id", name=op.f("pk_claim_evidence")),
    )
    op.create_table(
        "claim_relationships",
        sa.Column("relationship_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("from_id", sa.String(length=36), nullable=False),
        sa.Column("to_id", sa.String(length=36), nullable=False),
        sa.Column("relationship_type", sa.String(length=16), nullable=False),
        sa.Column("decision_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.decision_id"],
            name=op.f("fk_claim_relationships_decision_id_decisions"),
        ),
        sa.ForeignKeyConstraint(
            ["to_id"], ["claims.claim_id"], name=op.f("fk_claim_relationships_to_id_claims")
        ),
        sa.PrimaryKeyConstraint("relationship_id", name=op.f("pk_claim_relationships")),
    )
    with op.batch_alter_table("claim_relationships", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_claim_relationships_from_id"), ["from_id"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_claim_relationships_to_id"), ["to_id"], unique=False)

    op.create_table(
        "review_queue",
        sa.Column("decision_id", sa.String(length=36), nullable=False),
        sa.Column("claim_id", sa.String(length=36), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.decision_id"],
            name=op.f("fk_review_queue_decision_id_decisions"),
        ),
        sa.PrimaryKeyConstraint("decision_id", name=op.f("pk_review_queue")),
    )
    with op.batch_alter_table("review_queue", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_review_queue_claim_id"), ["claim_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("review_queue", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_review_queue_claim_id"))

    op.drop_table("review_queue")
    with op.batch_alter_table("claim_relationships", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_claim_relationships_to_id"))
        batch_op.drop_index(batch_op.f("ix_claim_relationships_from_id"))

    op.drop_table("claim_relationships")
    op.drop_table("claim_evidence")
    with op.batch_alter_table("candidates", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_candidates_claim_id"))

    op.drop_table("candidates")
    op.drop_table("evidence")
    op.drop_table("episodes")
    with op.batch_alter_table("decisions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_decisions_candidate_claim_id"))

    op.drop_table("decisions")
    with op.batch_alter_table("claims", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_claims_subject_key_predicate_key"))

    op.drop_table("claims")
    op.drop_table("claim_keys")
