"""Create the initial persistent memory schema.

Revision ID: 0001_memory_schema
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_memory_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "claims",
        sa.Column("claim_id", sa.String(length=36), primary_key=True),
        sa.Column("subject_key", sa.String(), nullable=False),
        sa.Column("predicate_key", sa.String(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("supersedes_claim_id", sa.String(length=36)),
        sa.Column("claim_json", sa.Text(), nullable=False),
    )
    op.create_index("ix_claims_subject_key", "claims", ["subject_key"])
    op.create_index("ix_claims_predicate_key", "claims", ["predicate_key"])
    op.create_index("ix_claims_status", "claims", ["status"])
    op.create_index(
        "uq_claims_one_active_per_key",
        "claims",
        ["subject_key", "predicate_key"],
        unique=True,
        sqlite_where=sa.text("status = 'ACTIVE'"),
    )
    op.create_table(
        "evidence",
        sa.Column("evidence_id", sa.String(length=36), primary_key=True),
        sa.Column("evidence_json", sa.Text(), nullable=False),
    )
    op.create_table(
        "claim_evidence",
        sa.Column(
            "claim_id",
            sa.String(length=36),
            sa.ForeignKey("claims.claim_id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column(
            "evidence_id",
            sa.String(length=36),
            sa.ForeignKey("evidence.evidence_id", ondelete="RESTRICT"),
            primary_key=True,
        ),
    )
    op.create_table(
        "claim_relationships",
        sa.Column(
            "source_claim_id",
            sa.String(length=36),
            sa.ForeignKey("claims.claim_id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column("relationship_type", sa.String(length=32), primary_key=True),
        sa.Column(
            "target_claim_id",
            sa.String(length=36),
            sa.ForeignKey("claims.claim_id", ondelete="RESTRICT"),
            primary_key=True,
        ),
    )
    op.create_table(
        "episodes",
        sa.Column("episode_id", sa.String(length=36), primary_key=True),
        sa.Column("episode_json", sa.Text(), nullable=False),
    )
    op.create_table(
        "decisions",
        sa.Column("decision_id", sa.String(length=36), primary_key=True),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("candidate_claim_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bundle_json", sa.Text(), nullable=False),
        sa.Column("delta_json", sa.Text(), nullable=False),
    )
    op.create_index("ix_decisions_candidate_claim_id", "decisions", ["candidate_claim_id"])
    op.create_index("ix_decisions_created_at", "decisions", ["created_at"])


def downgrade() -> None:
    raise NotImplementedError(
        "Memory and audit rows are immutable; destructive downgrade is disabled"
    )
