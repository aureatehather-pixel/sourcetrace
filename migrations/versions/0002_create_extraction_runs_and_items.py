"""create extraction runs and items

Reviewed by hand after autogenerate.

Adds the derived-information layer. `evidence_items` is deliberately untouched:
M2 never writes model output back into raw evidence, so this migration adds
tables and changes nothing that exists.

Two foreign keys, two different delete semantics, both intentional:

- `extraction_runs.evidence_id` -> RESTRICT. Evidence cannot be deleted while
  extraction history references it. Losing the provenance chain silently is the
  failure this system exists to prevent.
- `extraction_items.extraction_run_id` -> CASCADE. An item has no meaning apart
  from its run; it is part of that aggregate.

There is deliberately NO unique constraint on
(evidence_id, extractor_name, extractor_version). Retries are legitimate
history: a timeout, then a malformed response, then a success is three rows and
all three are worth keeping. Reuse is served by the partial index over
successful runs only.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-27

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "extraction_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("evidence_id", sa.UUID(), nullable=False),
        sa.Column("input_content_hash", sa.String(length=64), nullable=False),
        sa.Column("extractor_name", sa.String(length=64), nullable=False),
        sa.Column("extractor_version", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "running",
                "succeeded",
                "failed",
                name="extraction_run_status",
                native_enum=False,
                length=16,
            ),
            nullable=False,
        ),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(status = 'failed' AND error_code IS NOT NULL) OR (status <> 'failed' AND error_code IS NULL)",
            name=op.f("ck_extraction_runs_error_code_matches_status"),
        ),
        sa.CheckConstraint(
            "(status = 'running' AND completed_at IS NULL) OR (status <> 'running' AND completed_at IS NOT NULL)",
            name=op.f("ck_extraction_runs_completed_at_matches_status"),
        ),
        sa.CheckConstraint(
            "input_content_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_extraction_runs_input_content_hash_is_sha256_hex"),
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name=op.f("ck_extraction_runs_input_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name=op.f("ck_extraction_runs_output_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            "schema_version > 0", name=op.f("ck_extraction_runs_schema_version_positive")
        ),
        sa.ForeignKeyConstraint(
            ["evidence_id"],
            ["evidence_items.id"],
            name=op.f("fk_extraction_runs_evidence_id_evidence_items"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_extraction_runs")),
    )
    op.create_index(
        "ix_extraction_runs_evidence_id_created_at",
        "extraction_runs",
        ["evidence_id", sa.literal_column("created_at DESC")],
        unique=False,
    )
    op.create_index(
        "ix_extraction_runs_reuse",
        "extraction_runs",
        [
            "evidence_id",
            "input_content_hash",
            "extractor_name",
            "extractor_version",
            "schema_version",
        ],
        unique=False,
        postgresql_where=sa.text("status = 'succeeded'"),
    )
    op.create_table(
        "extraction_items",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("extraction_run_id", sa.UUID(), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "reported_claim",
                "observation",
                "entity_mention",
                "project",
                "resource",
                "event",
                "opportunity",
                "idea",
                name="extraction_item_kind",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("grounding_quote", sa.Text(), nullable=False),
        sa.Column(
            "model_confidence",
            sa.Enum(
                "low", "medium", "high", name="extraction_confidence", native_enum=False, length=8
            ),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "jsonb_typeof(payload) = 'object'", name=op.f("ck_extraction_items_payload_is_object")
        ),
        sa.CheckConstraint(
            "length(grounding_quote) > 0",
            name=op.f("ck_extraction_items_grounding_quote_not_empty"),
        ),
        sa.CheckConstraint("position >= 0", name=op.f("ck_extraction_items_position_non_negative")),
        sa.ForeignKeyConstraint(
            ["extraction_run_id"],
            ["extraction_runs.id"],
            name=op.f("fk_extraction_items_extraction_run_id_extraction_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_extraction_items")),
        sa.UniqueConstraint("extraction_run_id", "position", name="unique_position_per_run"),
    )
    op.create_index(
        "ix_extraction_items_extraction_run_id",
        "extraction_items",
        ["extraction_run_id"],
        unique=False,
    )
    op.create_index("ix_extraction_items_kind", "extraction_items", ["kind"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_extraction_items_kind", table_name="extraction_items")
    op.drop_index("ix_extraction_items_extraction_run_id", table_name="extraction_items")
    op.drop_table("extraction_items")
    op.drop_index(
        "ix_extraction_runs_reuse",
        table_name="extraction_runs",
        postgresql_where=sa.text("status = 'succeeded'"),
    )
    op.drop_index("ix_extraction_runs_evidence_id_created_at", table_name="extraction_runs")
    op.drop_table("extraction_runs")
