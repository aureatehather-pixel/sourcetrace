"""make the source interval the durable identity of extraction work

Earned by an audit finding with a concrete failure path, not by tidying. 0007
gave a run an ordinal, `window_index`, and the size of the plan it belonged to,
`window_count`. Both are safe only while the set of intervals is fixed for the
life of a source, and adaptive extraction makes it dynamic: an interval that
overflows is replaced by children covering exactly its characters, so the leaf
set grows during the run that discovers it.

**What the ordinal breaks.** `packet/repository._current_run_ids()` selects the
current run per `(evidence_id, window_index)`. Subdivide the interval numbered 2
into three children and number them 2, 3, 4 and child 3 — created later — now
outranks the untouched, still-valid run for the original interval 3 and silently
removes its items from every packet. The reuse index has the same shape and the
same flaw: a lookup for one interval would be answered by another's stored
output. Renumbering siblings instead is not an escape; it redefines work that
was already paid for and invalidates their reuse keys.

**What the plan size breaks.** `window_count` is a snapshot of a plan. Once one
interval subdivides, every sibling row carries a total that is no longer true,
and completeness computed as `count(distinct window_index) = max(window_count)`
compares a polluted count against a stale total. Both halves are wrong at once.

So identity becomes the thing that does not move: the half-open character range
`[window_start, window_end)` the run actually read. Two runs sharing it read the
same characters by definition, which is exactly when reuse is correct, and a
subdivided parent's children tile precisely what the parent covered, so
completeness — now an interval-tiling question over `[0, length(content))` —
gives the same answer before and after any subdivision anywhere else.

**Why the columns go rather than being deprecated in place.** Leaving them
behind was considered first, since a column nobody reads harms nothing at
runtime. It does not survive this project's gates: `alembic check` compares the
mapped models against the database, so a column dropped from the model and kept
in the schema is reported as drift and fails the build. Keeping them in the
model instead would leave two competing identities in the mapping, which is the
defect being removed. Neither column is read by anything after this revision,
and `window_start`/`window_end` — the data that matters — are untouched, so
nothing about which text a run read is lost.

**The downgrade is faithful but approximate, and says so.** Ordinals are
reconstructed by ordering each evidence item's runs on their offsets, which
restores a valid, constraint-satisfying numbering rather than the exact numbers
0007 stored: a source whose runs span several extraction generations will be
renumbered across all of them, and a subdivided source has more leaves than any
single plan ever had. No run, no item and no interval is lost either way.

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-30

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "extraction_runs"
_REUSE_INDEX = "ix_extraction_runs_reuse"


def upgrade() -> None:
    # The reuse key first: it references window_index, so it cannot outlive it.
    # Rebuilt rather than altered, because the key changed shape.
    op.drop_index(_REUSE_INDEX, table_name=_TABLE)
    op.create_index(
        _REUSE_INDEX,
        _TABLE,
        [
            "evidence_id",
            "window_start",
            "window_end",
            "input_content_hash",
            "extractor_name",
            "extractor_version",
            "schema_version",
        ],
        unique=False,
        postgresql_where=sa.text("status = 'succeeded'"),
    )

    # These three constrain columns that are about to stop existing.
    for name in (
        "window_index_within_count",
        "window_count_positive",
        "window_index_non_negative",
    ):
        op.drop_constraint(op.f(f"ck_{_TABLE}_{name}"), _TABLE, type_="check")

    op.drop_column(_TABLE, "window_count")
    op.drop_column(_TABLE, "window_index")

    # window_start / window_end and their CHECKs are deliberately untouched:
    # they were already correct, and they are now the whole identity.


def downgrade() -> None:
    op.add_column(_TABLE, sa.Column("window_index", sa.Integer(), nullable=True))
    op.add_column(_TABLE, sa.Column("window_count", sa.Integer(), nullable=True))

    # Reconstructed from the offsets, which is the only source of truth left and
    # the reason this direction is approximate. Ordering on (window_start,
    # window_end, created_at, id) is total, so the numbering is deterministic;
    # row_number() is 1-based and counts every run of the evidence item, so
    # `window_index < window_count` holds for every row by construction and the
    # CHECK recreated below cannot fail on data this statement wrote.
    #
    # Literal table names, no interpolation: nothing here is caller-supplied,
    # and a constant spliced into DDL still reads like an injection site.
    op.execute(
        """
        UPDATE extraction_runs AS r
        SET window_index = ordered.position - 1,
            window_count = ordered.total
        FROM (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY evidence_id
                       ORDER BY window_start, window_end, created_at, id
                   ) AS position,
                   count(*) OVER (PARTITION BY evidence_id) AS total
            FROM extraction_runs
        ) AS ordered
        WHERE ordered.id = r.id
        """
    )

    for column in ("window_index", "window_count"):
        op.alter_column(_TABLE, column, nullable=False)

    op.create_check_constraint("window_index_non_negative", _TABLE, "window_index >= 0")
    op.create_check_constraint("window_count_positive", _TABLE, "window_count > 0")
    op.create_check_constraint("window_index_within_count", _TABLE, "window_index < window_count")

    op.drop_index(_REUSE_INDEX, table_name=_TABLE)
    op.create_index(
        _REUSE_INDEX,
        _TABLE,
        [
            "evidence_id",
            "window_index",
            "input_content_hash",
            "extractor_name",
            "extractor_version",
            "schema_version",
        ],
        unique=False,
        postgresql_where=sa.text("status = 'succeeded'"),
    )
