"""create relation analysis runs and intelligence relations

Reviewed by hand after autogenerate. Unlike 0004 the ordering came out correct
on its own — `relation_analysis_runs` is created before the table whose foreign
key references it, and dropped after it — so the only change was one explicit
constraint name.

Adds M4's derived-intelligence layer. `evidence_items`, `extraction_runs`,
`extraction_items`, `entities` and `entity_resolutions` keep every column,
constraint and index they had. **This migration changes no existing table at
all** — not even the redundant unique constraint 0004 had to add, because
`unique_id_kind` on `extraction_items` already exists and is exactly the target
both composite foreign keys below need.

Two tables, because batching makes them different things:

- `relation_analysis_runs` — one analyse-relations operation. Holds the cost and
  narrowing telemetry, which is *only* meaningful at this level: several
  independent pairs share one model call, so a per-relation token count would be
  invented.
- `intelligence_relations` — the derived assertions. Independently reusable, so
  the foreign key to the run is **RESTRICT** rather than M2's run→item CASCADE.
  A relation outlives the particular operation that produced it.

Delete rules, all RESTRICT, all deliberate:

- to `extraction_items` (two composite FKs) — an assertion cannot outlive the
  grounded material it was derived from;
- to `entities` — an entity with derived history cannot be deleted out from
  under it;
- to `relation_analysis_runs` — the run is the assertion's provenance.

The two composite foreign keys, on `(item_id, item_kind)`, are what make the
`*_item_kind_is_relatable` CHECKs trustworthy. Without them the denormalised kind
could disagree with the item it points at and the CHECK would validate the lie —
the M3 learning, applied again.

Three constraints carry the epistemics rather than mere hygiene:

- `pair_is_canonically_ordered` (`left < right`) gives a pair exactly one row, so
  "A contradicts B" and "B contradicts A" cannot both be stored and disagree. It
  also forbids relating an item to itself.
- `direction_matches_relation_type` — only `qualifies` may point a direction, and
  it must. Directionality is part of what the assertion means.
- `unique_settled_pair_per_classifier` — one settled answer per pair per
  classifier version, **including rejected ones**. This is the whole reuse
  mechanism: a question a human already answered "no" to is not paid for again,
  and bumping the classifier version is what re-opens it.

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-27

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "relation_analysis_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("entity_id", sa.UUID(), nullable=False),
        sa.Column("classifier_name", sa.String(length=64), nullable=False),
        sa.Column("classifier_version", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "running",
                "succeeded",
                "failed",
                name="relation_analysis_status",
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
        sa.Column("theoretical_pairs", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("candidate_pairs", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("reused_pairs", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("classified_pairs", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("batch_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("external_call_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
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
            name=op.f("ck_relation_analysis_runs_error_code_matches_status"),
        ),
        sa.CheckConstraint(
            "(status = 'running' AND completed_at IS NULL) OR (status <> 'running' AND completed_at IS NOT NULL)",
            name=op.f("ck_relation_analysis_runs_completed_at_matches_status"),
        ),
        sa.CheckConstraint(
            "candidate_pairs <= theoretical_pairs",
            name=op.f("ck_relation_analysis_runs_candidates_within_theoretical"),
        ),
        sa.CheckConstraint(
            "classified_pairs <= candidate_pairs",
            name=op.f("ck_relation_analysis_runs_classified_within_candidates"),
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name=op.f("ck_relation_analysis_runs_input_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name=op.f("ck_relation_analysis_runs_output_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            "schema_version > 0", name=op.f("ck_relation_analysis_runs_schema_version_positive")
        ),
        sa.CheckConstraint(
            "theoretical_pairs >= 0 AND candidate_pairs >= 0 AND reused_pairs >= 0 AND classified_pairs >= 0 AND batch_count >= 0 AND external_call_count >= 0",
            name=op.f("ck_relation_analysis_runs_counters_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["entity_id"],
            ["entities.id"],
            name=op.f("fk_relation_analysis_runs_entity_id_entities"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_relation_analysis_runs")),
    )
    op.create_index(
        "ix_relation_analysis_runs_entity_id_created_at",
        "relation_analysis_runs",
        ["entity_id", sa.literal_column("created_at DESC")],
        unique=False,
    )
    op.create_table(
        "intelligence_relations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("analysis_run_id", sa.UUID(), nullable=False),
        sa.Column("entity_id", sa.UUID(), nullable=False),
        sa.Column("left_extraction_item_id", sa.UUID(), nullable=False),
        sa.Column("right_extraction_item_id", sa.UUID(), nullable=False),
        sa.Column(
            "left_item_kind",
            sa.Enum(
                "reported_claim",
                "observation",
                "entity_mention",
                "project",
                "resource",
                "event",
                "opportunity",
                "idea",
                name="intelligence_relation_item_kind",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "right_item_kind",
            sa.Enum(
                "reported_claim",
                "observation",
                "entity_mention",
                "project",
                "resource",
                "event",
                "opportunity",
                "idea",
                name="intelligence_relation_item_kind",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "relation_type",
            sa.Enum(
                "reinforces",
                "qualifies",
                "contradicts",
                "no_meaningful_relation",
                name="intelligence_relation_type",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "direction",
            sa.Enum(
                "symmetric",
                "left_to_right",
                "right_to_left",
                name="intelligence_relation_direction",
                native_enum=False,
                length=16,
            ),
            nullable=False,
        ),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("classifier_name", sa.String(length=64), nullable=False),
        sa.Column("classifier_version", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("input_fingerprint", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "current",
                "rejected",
                name="intelligence_relation_status",
                native_enum=False,
                length=16,
            ),
            server_default=sa.text("'current'"),
            nullable=False,
        ),
        sa.Column("rejected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rejection_note", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(relation_type = 'no_meaningful_relation') OR (explanation IS NOT NULL AND length(explanation) > 0)",
            name=op.f("ck_intelligence_relations_meaningful_relation_is_explained"),
        ),
        sa.CheckConstraint(
            "(relation_type = 'qualifies' AND direction <> 'symmetric') OR (relation_type <> 'qualifies' AND direction = 'symmetric')",
            name=op.f("ck_intelligence_relations_direction_matches_relation_type"),
        ),
        sa.CheckConstraint(
            "(status = 'current' AND rejected_at IS NULL) OR (status = 'rejected' AND rejected_at IS NOT NULL)",
            name=op.f("ck_intelligence_relations_rejected_state_is_consistent"),
        ),
        sa.CheckConstraint(
            "input_fingerprint ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_intelligence_relations_input_fingerprint_is_sha256_hex"),
        ),
        sa.CheckConstraint(
            "left_item_kind IN ('observation', 'reported_claim')",
            name=op.f("ck_intelligence_relations_left_item_kind_is_relatable"),
        ),
        sa.CheckConstraint(
            "right_item_kind IN ('observation', 'reported_claim')",
            name=op.f("ck_intelligence_relations_right_item_kind_is_relatable"),
        ),
        sa.CheckConstraint(
            "explanation IS NULL OR length(explanation) > 0",
            name=op.f("ck_intelligence_relations_explanation_not_empty"),
        ),
        sa.CheckConstraint(
            "left_extraction_item_id < right_extraction_item_id",
            name=op.f("ck_intelligence_relations_pair_is_canonically_ordered"),
        ),
        sa.CheckConstraint(
            "rejection_note IS NULL OR length(rejection_note) > 0",
            name=op.f("ck_intelligence_relations_rejection_note_not_empty"),
        ),
        sa.CheckConstraint(
            "schema_version > 0", name=op.f("ck_intelligence_relations_schema_version_positive")
        ),
        sa.ForeignKeyConstraint(
            ["analysis_run_id"],
            ["relation_analysis_runs.id"],
            name="fk_intelligence_relations_analysis_run",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["entity_id"],
            ["entities.id"],
            name=op.f("fk_intelligence_relations_entity_id_entities"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["left_extraction_item_id", "left_item_kind"],
            ["extraction_items.id", "extraction_items.kind"],
            name="fk_intelligence_relations_left_item",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["right_extraction_item_id", "right_item_kind"],
            ["extraction_items.id", "extraction_items.kind"],
            name="fk_intelligence_relations_right_item",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_intelligence_relations")),
        sa.UniqueConstraint(
            "left_extraction_item_id",
            "right_extraction_item_id",
            "classifier_name",
            "classifier_version",
            "schema_version",
            name="unique_settled_pair_per_classifier",
        ),
    )
    op.create_index(
        "ix_intelligence_relations_analysis_run_id",
        "intelligence_relations",
        ["analysis_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_intelligence_relations_entity_id_current",
        "intelligence_relations",
        ["entity_id"],
        unique=False,
        postgresql_where=sa.text("status = 'current'"),
    )
    op.create_index(
        "ix_intelligence_relations_left_extraction_item_id",
        "intelligence_relations",
        ["left_extraction_item_id"],
        unique=False,
    )
    op.create_index(
        "ix_intelligence_relations_right_extraction_item_id",
        "intelligence_relations",
        ["right_extraction_item_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_intelligence_relations_right_extraction_item_id", table_name="intelligence_relations"
    )
    op.drop_index(
        "ix_intelligence_relations_left_extraction_item_id", table_name="intelligence_relations"
    )
    op.drop_index(
        "ix_intelligence_relations_entity_id_current",
        table_name="intelligence_relations",
        postgresql_where=sa.text("status = 'current'"),
    )
    op.drop_index("ix_intelligence_relations_analysis_run_id", table_name="intelligence_relations")
    op.drop_table("intelligence_relations")
    op.drop_index(
        "ix_relation_analysis_runs_entity_id_created_at", table_name="relation_analysis_runs"
    )
    op.drop_table("relation_analysis_runs")
