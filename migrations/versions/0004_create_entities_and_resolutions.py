"""create entities and entity resolutions

Reviewed by hand after autogenerate. Two corrections were needed, both about
ordering rather than shape:

1. `unique_id_kind` on `extraction_items` must be created **before**
   `entity_resolutions`, because that table's composite foreign key references
   `(extraction_items.id, extraction_items.kind)` and PostgreSQL requires a
   unique constraint on the referenced columns. Autogenerate emitted it last.
2. The downgrade must drop it **after** `entity_resolutions`, for the same
   reason in reverse. Autogenerate emitted it first.

Adds the connection layer. `evidence_items`, `extraction_runs` and
`extraction_items` keep every column, constraint and index they had; the single
change to existing schema is the redundant unique constraint in (1), which adds
no new truth about `extraction_items` (`id` is already its primary key) and
exists only so the composite FK has a legal target.

Delete rules, all RESTRICT, all deliberate:

- `entity_resolutions -> extraction_items` — an assertion cannot outlive the
  material it was made from. This also means a run whose items carry
  resolutions can no longer be cascade-deleted through `extraction_items`, which
  is the intended protection rather than a side effect.
- `entity_resolutions -> entities` — an entity with history cannot be deleted
  out from under it.
- `entity_resolutions -> entity_resolutions` (superseded_by) — the correction
  chain is the audit trail.

The load-bearing index is `uq_entity_resolutions_active_item_role`: a **partial**
unique index over `is_active`. One extraction item can have at most one active
resolution per role, so a mention can never simultaneously refer to two
entities; superseded history is left unconstrained and unbounded.

Note what is deliberately absent: **no unique constraint on any entity name**.
Two people are both called John Smith. A unique name would either reject the
second one or force a false merge.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-27

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Must precede entity_resolutions: its composite FK references these columns.
    op.create_unique_constraint("unique_id_kind", "extraction_items", ["id", "kind"])

    op.create_table(
        "entities",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "entity_type",
            sa.Enum(
                "person",
                "organization",
                "product",
                "place",
                "other",
                name="entity_type",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("canonical_name", sa.Text(), nullable=False),
        sa.Column("normalized_name", sa.Text(), nullable=False),
        sa.Column("disambiguator", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "disambiguator IS NULL OR length(disambiguator) > 0",
            name=op.f("ck_entities_disambiguator_not_empty"),
        ),
        sa.CheckConstraint(
            "length(canonical_name) > 0", name=op.f("ck_entities_canonical_name_not_empty")
        ),
        sa.CheckConstraint(
            "length(normalized_name) > 0", name=op.f("ck_entities_normalized_name_not_empty")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_entities")),
    )
    op.create_index(
        "ix_entities_created_at_id",
        "entities",
        [sa.literal_column("created_at DESC"), sa.literal_column("id DESC")],
        unique=False,
    )
    op.create_index(
        "ix_entities_entity_type_normalized_name",
        "entities",
        ["entity_type", "normalized_name"],
        unique=False,
    )

    op.create_table(
        "entity_resolutions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("extraction_item_id", sa.UUID(), nullable=False),
        sa.Column(
            "source_item_kind",
            sa.Enum(
                "reported_claim",
                "observation",
                "entity_mention",
                "project",
                "resource",
                "event",
                "opportunity",
                "idea",
                name="entity_resolution_source_item_kind",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("entity_id", sa.UUID(), nullable=False),
        sa.Column(
            "role",
            sa.Enum(
                "mention",
                "claimed_by",
                name="entity_resolution_role",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "method",
            sa.Enum(
                "exact_match",
                "manual",
                name="entity_resolution_method",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("superseded_by_id", sa.UUID(), nullable=True),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("note", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(role = 'mention' AND source_item_kind = 'entity_mention') OR (role = 'claimed_by' AND source_item_kind = 'reported_claim')",
            name=op.f("ck_entity_resolutions_role_matches_source_item_kind"),
        ),
        sa.CheckConstraint(
            "(is_active AND superseded_at IS NULL AND superseded_by_id IS NULL) OR (NOT is_active AND superseded_at IS NOT NULL)",
            name=op.f("ck_entity_resolutions_superseded_state_is_consistent"),
        ),
        sa.CheckConstraint(
            "note IS NULL OR length(note) > 0", name=op.f("ck_entity_resolutions_note_not_empty")
        ),
        sa.CheckConstraint(
            "superseded_by_id IS NULL OR superseded_by_id <> id",
            name=op.f("ck_entity_resolutions_cannot_supersede_itself"),
        ),
        sa.ForeignKeyConstraint(
            ["entity_id"],
            ["entities.id"],
            name=op.f("fk_entity_resolutions_entity_id_entities"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["extraction_item_id", "source_item_kind"],
            ["extraction_items.id", "extraction_items.kind"],
            name="fk_entity_resolutions_extraction_item",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["superseded_by_id"],
            ["entity_resolutions.id"],
            name=op.f("fk_entity_resolutions_superseded_by_id_entity_resolutions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_entity_resolutions")),
    )
    op.create_index(
        "ix_entity_resolutions_entity_id_active",
        "entity_resolutions",
        ["entity_id"],
        unique=False,
        postgresql_where=sa.text("is_active"),
    )
    op.create_index(
        "ix_entity_resolutions_extraction_item_id",
        "entity_resolutions",
        ["extraction_item_id"],
        unique=False,
    )
    op.create_index(
        "uq_entity_resolutions_active_item_role",
        "entity_resolutions",
        ["extraction_item_id", "role"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_entity_resolutions_active_item_role",
        table_name="entity_resolutions",
        postgresql_where=sa.text("is_active"),
    )
    op.drop_index("ix_entity_resolutions_extraction_item_id", table_name="entity_resolutions")
    op.drop_index(
        "ix_entity_resolutions_entity_id_active",
        table_name="entity_resolutions",
        postgresql_where=sa.text("is_active"),
    )
    op.drop_table("entity_resolutions")
    op.drop_index("ix_entities_entity_type_normalized_name", table_name="entities")
    op.drop_index("ix_entities_created_at_id", table_name="entities")
    op.drop_table("entities")
    # Last: the composite FK above depended on it.
    op.drop_constraint("unique_id_kind", "extraction_items", type_="unique")
