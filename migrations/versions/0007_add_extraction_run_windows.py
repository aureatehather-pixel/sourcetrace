"""record the extraction window a run covered

Earned by a reproduced real-data defect, not by design taste. Event 2 — one
60,241-character conversation export, ingested as one evidence item — was sent
to the provider as a single prompt and came back with `stop_reason: max_tokens`,
`output_tokens` exactly on the configured 8,000 ceiling, JSON truncated
mid-string, and was correctly rejected. That same response had already emitted
69 items, so raising the output ceiling would only have moved the failure onto
`MAX_ITEMS_PER_RUN = 50`. Extraction had to become able to read one source in
several passes.

**Why this is a column on `extraction_runs` and not a new table, and above all
not more rows in `evidence_items`.** Splitting the export into several evidence
items was the obvious shortcut and it is the one thing that would have broken
the system's meaning. M4's cross-evidence filter is exactly
`first.evidence_id == second.evidence_id`; give one conversation six evidence
ids and two paraphrases of a single remark, landing in different chunks, become
an eligible pair the classifier can label `reinforces` — manufactured
corroboration from a single utterance. The packet would have reported six
sources where a reader should see one. So the source stays singular and the
*run* learns which slice of it it read.

Four columns, all NOT NULL, all with a default that describes the world before
windowing existed: one window, index 0, spanning the whole document. Every
existing row is therefore already correct under the new meaning, which is why
this migration rewrites no data beyond filling `window_end`.

`window_end` is the one value that cannot come from a constant, so it is
backfilled from the parent's own content length. It is computed with `length()`
over `evidence_items.content` — characters, matching Python's `len()` on the
same text, so the stored range and the slice that produced it agree.

The reuse index gains `window_index`. Without that, a lookup for window 3 would
be satisfied by the succeeded run for window 0, and a six-window export would
extract once and declare itself finished.

**The downgrade is lossy and says so.** Dropping these columns collapses every
window of a multi-window source into indistinguishable runs over one evidence
id, and the reuse index reverts to a key that cannot tell them apart. Any
document that needed more than one window must be re-extracted after a
downgrade; nothing is deleted, but the runs stop being individually meaningful.

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-29

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "extraction_runs"
_REUSE_INDEX = "ix_extraction_runs_reuse"


def upgrade() -> None:
    # Added nullable, backfilled, then made NOT NULL — the standard three-step,
    # because a NOT NULL column with no default cannot be added to a populated
    # table in one statement.
    op.add_column(_TABLE, sa.Column("window_index", sa.Integer(), nullable=True))
    op.add_column(_TABLE, sa.Column("window_start", sa.Integer(), nullable=True))
    op.add_column(_TABLE, sa.Column("window_end", sa.Integer(), nullable=True))
    op.add_column(_TABLE, sa.Column("window_count", sa.Integer(), nullable=True))

    # Every pre-windowing run read its evidence item whole.
    # Literal table names, no interpolation: nothing here is caller-supplied,
    # and a constant spliced into DDL still reads like an injection site.
    op.execute(
        """
        UPDATE extraction_runs AS r
        SET window_index = 0,
            window_start = 0,
            window_count = 1,
            window_end = length(e.content)
        FROM evidence_items AS e
        WHERE e.id = r.evidence_id
        """
    )

    for column in ("window_index", "window_start", "window_end", "window_count"):
        op.alter_column(_TABLE, column, nullable=False)

    op.create_check_constraint("window_index_non_negative", _TABLE, "window_index >= 0")
    op.create_check_constraint("window_count_positive", _TABLE, "window_count > 0")
    op.create_check_constraint("window_index_within_count", _TABLE, "window_index < window_count")
    op.create_check_constraint("window_start_non_negative", _TABLE, "window_start >= 0")
    op.create_check_constraint("window_end_after_start", _TABLE, "window_end >= window_start")

    # Rebuilt rather than altered: the reuse key changed shape.
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


def downgrade() -> None:
    op.drop_index(_REUSE_INDEX, table_name=_TABLE)
    op.create_index(
        _REUSE_INDEX,
        _TABLE,
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

    for name in (
        "window_end_after_start",
        "window_start_non_negative",
        "window_index_within_count",
        "window_count_positive",
        "window_index_non_negative",
    ):
        op.drop_constraint(op.f(f"ck_{_TABLE}_{name}"), _TABLE, type_="check")

    for column in ("window_count", "window_end", "window_start", "window_index"):
        op.drop_column(_TABLE, column)
