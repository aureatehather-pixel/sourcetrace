"""drop extraction_items.model_confidence

Reviewed by hand.

`model_confidence` stored the model's own three-level rating of how well the
evidence supported an item. Nothing calibrated it: no ground truth, no scoring
against outcomes, no reliability data. "high" and "low" therefore carried no
defensible meaning while reading as if they did, which is the fake precision
this project forbids. Uncertainty is now represented only where the evidence
itself expresses it - in the extracted wording and in `reported_claim.stance`.

**This migration is destructive.** The stored ratings are dropped and cannot be
recovered by downgrading; the downgrade recreates the column and backfills a
placeholder purely so the 0002 schema shape is restorable. That is deliberate -
the values are exactly what we decided should not exist.

The accompanying extractor bump (2.0.0 / schema 2) means every run recorded
under the old schema falls out of the reuse key, so nothing produced by the old
pipeline is ever served as current.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-27

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ENUM_TYPE = sa.Enum(
    "low",
    "medium",
    "high",
    name="extraction_confidence",
    native_enum=False,
    length=8,
)


def upgrade() -> None:
    op.drop_column("extraction_items", "model_confidence")


def downgrade() -> None:
    # Restores the 0002 column shape, not the 0002 data. Added with a
    # server_default so existing rows satisfy NOT NULL, then the default is
    # dropped so 0002's shape (no default) is reproduced exactly.
    op.add_column(
        "extraction_items",
        sa.Column(
            "model_confidence",
            _ENUM_TYPE,
            nullable=False,
            server_default="medium",
        ),
    )
    op.alter_column("extraction_items", "model_confidence", server_default=None)
