"""make the generating model part of what reusable output is keyed on

Earned by an adversarial finding with a demonstrated failure, not by tidying.
Both derived layers decide whether to pay for a model call by looking up stored
output, and neither lookup consulted the provider or the model that produced it.
So a run extracted by one vendor answered a lookup made while the system was
configured for another, and a relation classified by one model was served as
though the currently configured model had produced it.

**Why that is worse than an ordinary cache miss.** The lookup does not merely
return the wrong row; it returns a row that is *indistinguishable* from a right
one, because everything else in the key genuinely matches. The evidence is the
same, the extractor version is the same, the schema version is the same. The
only thing that differs is the fact the key omitted. A comparison between two
providers run on top of this measures the first provider twice and reports it as
a result about the second — which is precisely the measurement that was about to
be run.

Two schema changes, one per layer, and they do different work.

**M2 — `ix_extraction_runs_reuse`.** Purely an index: the reuse lookup gained
`provider` and `model` in its `WHERE`, and the partial index that serves it
follows. No constraint changes and no column changes. Extraction runs may
legitimately carry a null provider — a run that failed before one was recorded
is real history — and a null in a lookup index simply never matches, which is
the correct answer for "output whose generator is unknown".

**M4 — `unique_settled_pair_per_classifier`, and two columns made NOT NULL.**
This one is a constraint, and it *had* to move with the lookup rather than after
it. The constraint said one settled answer per pair per classifier version. Widen
only the lookup and a model swap misses it, inserts, and collides here: asking a
second model about a pair the first already answered would raise an integrity
error instead of asking. Reuse identity and settled-answer identity are the same
identity; they are now spelled the same way.

The columns become NOT NULL because they are joining a unique constraint.
PostgreSQL treats nulls as distinct, so a nullable member would let any number of
null-identity rows coexist for one pair — silently repealing the guarantee this
constraint exists to give, in exactly the case where nobody could tell which
model wrote which row. Every relation row in every local database already carries
both values (`add_relation` has always set them), so the tightening rejects
nothing that exists; if a deployment does hold a null, the `ALTER` fails loudly
here rather than the constraint quietly weakening there.

**What this migration does not do.** It does not backfill, rewrite or reinterpret
any stored value. Rows written before it keep the provider and model they were
written with. What changes for them is only that a lookup now has to name the
same generator to be served their output, which for the existing corpus
(`anthropic` / `claude-opus-5`, `anthropic` / `claude-sonnet-5`) it still does.

**The downgrade is exact.** Both objects are restored to their previous
definition and both columns to nullable. Nothing is dropped and no row is
touched, so downgrading loses only the guarantee, not the data. The one thing it
cannot undo is a database that has since stored two answers for one pair under
two models: recreating the narrower constraint will fail on such a pair, loudly
and before any data is changed, which is the honest outcome — the narrower
constraint is a false statement about that database.

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-31

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RUNS = "extraction_runs"
_REUSE_INDEX = "ix_extraction_runs_reuse"
_RELATIONS = "intelligence_relations"
_SETTLED = "unique_settled_pair_per_classifier"

#: The reuse index as 0008 left it. Named once so upgrade and downgrade cannot
#: drift apart, which is how a downgrade comes to restore something that was
#: never there.
_REUSE_COLUMNS_BEFORE = (
    "evidence_id",
    "window_start",
    "window_end",
    "input_content_hash",
    "extractor_name",
    "extractor_version",
    "schema_version",
)
_REUSE_COLUMNS_AFTER = (*_REUSE_COLUMNS_BEFORE, "provider", "model")

_SETTLED_COLUMNS_BEFORE = (
    "left_extraction_item_id",
    "right_extraction_item_id",
    "classifier_name",
    "classifier_version",
    "schema_version",
)
_SETTLED_COLUMNS_AFTER = (*_SETTLED_COLUMNS_BEFORE, "provider", "model")


def upgrade() -> None:
    op.drop_index(_REUSE_INDEX, table_name=_RUNS)
    op.create_index(
        _REUSE_INDEX,
        _RUNS,
        list(_REUSE_COLUMNS_AFTER),
        unique=False,
        postgresql_where=sa.text("status = 'succeeded'"),
    )

    # NOT NULL before the constraint, not after: a null slipping in between
    # would be admitted by the constraint and then rejected by the ALTER, and
    # the failure would name the wrong thing.
    op.alter_column(_RELATIONS, "provider", existing_type=sa.String(32), nullable=False)
    op.alter_column(_RELATIONS, "model", existing_type=sa.String(128), nullable=False)

    op.drop_constraint(_SETTLED, _RELATIONS, type_="unique")
    op.create_unique_constraint(_SETTLED, _RELATIONS, list(_SETTLED_COLUMNS_AFTER))


def downgrade() -> None:
    op.drop_constraint(_SETTLED, _RELATIONS, type_="unique")
    op.create_unique_constraint(_SETTLED, _RELATIONS, list(_SETTLED_COLUMNS_BEFORE))

    op.alter_column(_RELATIONS, "model", existing_type=sa.String(128), nullable=True)
    op.alter_column(_RELATIONS, "provider", existing_type=sa.String(32), nullable=True)

    op.drop_index(_REUSE_INDEX, table_name=_RUNS)
    op.create_index(
        _REUSE_INDEX,
        _RUNS,
        list(_REUSE_COLUMNS_BEFORE),
        unique=False,
        postgresql_where=sa.text("status = 'succeeded'"),
    )
