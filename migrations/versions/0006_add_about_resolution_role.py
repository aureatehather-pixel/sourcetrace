"""allow the `about` resolution role for observation items

Written by hand; there is nothing here autogenerate can see. The role column is
`VARCHAR(32)` with no value CHECK of its own — `native_enum=False` defaults to
`create_constraint=False` — so the *only* thing that has ever constrained the
role vocabulary is `role_matches_source_item_kind`. Widening that one constraint
is therefore the whole migration, and it is also what admits the new role at all.

Before this, no resolution role could describe an `observation`, so an
observation had no path to an entity and M4 — which already accepts observations
on both sides of a relation — never saw one. That is the gap being closed.

The third arm is deliberately as narrow as the other two. `about` asserts that
the observation *concerns* the entity and nothing else; it can never be attached
to a `reported_claim`, so an observation cannot acquire an attribution the
evidence never gave it, and a claim cannot quietly be re-read as a fact about its
own speaker. The composite foreign key on `(extraction_item_id, source_item_kind)`
still makes the denormalised kind unforgeable, so this CHECK constrains the real
item kind rather than a copy that could drift.

No table is rewritten and no data changes. Existing `mention` and `claimed_by`
rows satisfy the new constraint unchanged, and PostgreSQL validates the new CHECK
against them on creation.

**The downgrade is intentionally not defensive.** It restores the two-arm
constraint, which PostgreSQL will refuse to create if any `about` row exists.
That failure is the correct outcome: silently deleting resolutions to make a
downgrade succeed would destroy asserted history, which this system never does.
Detach the `about` resolutions deliberately first if the downgrade is really
wanted.

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-29

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# `op.f()` at every use site: the metadata naming convention would otherwise
# re-prefix this already-final name into
# `ck_entity_resolutions_ck_entity_resolutions_role_matche_ef39`.
_CONSTRAINT = "ck_entity_resolutions_role_matches_source_item_kind"
_TABLE = "entity_resolutions"

_TWO_ARM = (
    "(role = 'mention' AND source_item_kind = 'entity_mention') "
    "OR (role = 'claimed_by' AND source_item_kind = 'reported_claim')"
)

_THREE_ARM = (
    "(role = 'mention' AND source_item_kind = 'entity_mention') "
    "OR (role = 'claimed_by' AND source_item_kind = 'reported_claim') "
    "OR (role = 'about' AND source_item_kind = 'observation')"
)


def upgrade() -> None:
    op.drop_constraint(op.f(_CONSTRAINT), _TABLE, type_="check")
    op.create_check_constraint(
        op.f(_CONSTRAINT),
        _TABLE,
        _THREE_ARM,
    )


def downgrade() -> None:
    # Fails loudly if any `about` resolution exists. See the module docstring:
    # that is the intended behaviour, not an oversight to work around.
    op.drop_constraint(op.f(_CONSTRAINT), _TABLE, type_="check")
    op.create_check_constraint(
        op.f(_CONSTRAINT),
        _TABLE,
        _TWO_ARM,
    )
