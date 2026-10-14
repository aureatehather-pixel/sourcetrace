"""create evidence_items

Reviewed by hand after autogenerate.

Note the index on ``content_hash`` is deliberately NOT unique. Two people can
independently tell you the same thing; those are two observations with two
provenances and both must survive. The fingerprint exists for unchanged-content
detection, caching and diagnostics - never for identity.

Revision ID: 0001
Revises:
Create Date: 2026-08-27

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "evidence_items",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=True),
        sa.Column("source_label", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_evidence_items_content_hash_is_sha256_hex"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(metadata) = 'object'",
            name=op.f("ck_evidence_items_metadata_is_object"),
        ),
        sa.CheckConstraint("length(content) > 0", name=op.f("ck_evidence_items_content_not_empty")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evidence_items")),
    )
    op.create_index(
        "ix_evidence_items_content_hash", "evidence_items", ["content_hash"], unique=False
    )
    # Matches the list endpoint's ORDER BY exactly so keyset pagination is an
    # index scan at any depth.
    op.create_index(
        "ix_evidence_items_created_at_id",
        "evidence_items",
        [sa.literal_column("created_at DESC"), sa.literal_column("id DESC")],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_evidence_items_created_at_id", table_name="evidence_items")
    op.drop_index("ix_evidence_items_content_hash", table_name="evidence_items")
    op.drop_table("evidence_items")
