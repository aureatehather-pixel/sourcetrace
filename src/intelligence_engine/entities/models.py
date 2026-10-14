"""Entity identity and resolution tables.

    EvidenceItem 1--* ExtractionRun 1--* ExtractionItem 1--* EntityResolution *--1 Entity

`entities` is persistent identity. `entity_resolutions` is the *assertion* that
some grounded extraction item refers to one of those identities — derived
information that can be wrong, and therefore carries its own provenance and its
own history, exactly as extraction runs do.

There is deliberately **no mentions table**. M2 already stores grounded mentions:
an `extraction_items` row with `kind = 'entity_mention'`, a validated payload and
a verbatim `grounding_quote` *is* the mention record. Copying it into a second
table would create two versions of one fact and a reason for them to disagree.

Three delete rules, all chosen rather than defaulted:

- `entity_resolutions -> extraction_items` is **RESTRICT** (composite, see below).
  A resolution is evidence of a decision; deleting the material it was made from
  would leave an assertion about nothing. This also means a run whose items have
  resolutions can no longer be cascade-deleted, which is the intended protection.
- `entity_resolutions -> entities` is **RESTRICT**. An entity with history cannot
  be deleted out from under it. Merging or retiring an entity is a future
  operation with its own record, never a DELETE.
- `entity_resolutions.superseded_by_id -> entity_resolutions` is **RESTRICT**.
  The correction chain is the audit trail.
"""

from __future__ import annotations

import datetime as dt
import enum
import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from intelligence_engine.db.base import Base
from intelligence_engine.entities.schemas import (
    MAX_NOTE_LENGTH,
    EntityType,
    ResolutionMethod,
    ResolutionRole,
)
from intelligence_engine.extraction.schemas import ItemKind


def _values(enum_class: type[enum.Enum]) -> list[str]:
    """Store the enum's *values*, not its member names."""
    return [str(member.value) for member in enum_class]


# native_enum=False renders VARCHAR + CHECK, matching the M2 convention: adding
# a value costs one CHECK rewrite rather than an ALTER TYPE.
_ENTITY_TYPE = Enum(
    EntityType,
    native_enum=False,
    length=32,
    name="entity_type",
    values_callable=lambda e: _values(EntityType),
)
_ROLE_TYPE = Enum(
    ResolutionRole,
    native_enum=False,
    length=32,
    name="entity_resolution_role",
    values_callable=lambda e: _values(ResolutionRole),
)
_METHOD_TYPE = Enum(
    ResolutionMethod,
    native_enum=False,
    length=32,
    name="entity_resolution_method",
    values_callable=lambda e: _values(ResolutionMethod),
)
# Same value set and same width as `extraction_items.kind`, because the two
# columns are joined by a composite foreign key. Derived from `ItemKind` so the
# two cannot drift apart.
_SOURCE_KIND_TYPE = Enum(
    ItemKind,
    native_enum=False,
    length=32,
    name="entity_resolution_source_item_kind",
    values_callable=lambda e: _values(ItemKind),
)


class Entity(Base):
    """A persistent identity. Identity is the id; the name is an attribute.

    There is **no unique constraint on any name column**, and that is the whole
    point. Two different people are both called John Smith; two organisations
    share an acronym. A unique name would either reject the second one or force
    a merge, and both are wrong answers to a real situation.
    """

    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    entity_type: Mapped[EntityType] = mapped_column(_ENTITY_TYPE, nullable=False)

    # What a human should see. Free to be edited later without affecting identity.
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)

    # The deterministic match key, computed by `matching.normalize_name` and
    # stored so lookups are an index scan. NOT a PostgreSQL generated column:
    # the normalisation is Unicode NFKC + casefold, which SQL's lower() does not
    # reproduce. Written on exactly one code path - the entity repository.
    normalized_name: Mapped[str] = mapped_column(Text, nullable=False)

    # How a human tells this entity apart from a same-named one. Human-written
    # only; nothing generates it, so it can never become invented biography.
    disambiguator: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("length(canonical_name) > 0", name="canonical_name_not_empty"),
        # An empty key would match every other empty key and merge unrelated
        # entities on the first auto-resolution.
        CheckConstraint("length(normalized_name) > 0", name="normalized_name_not_empty"),
        CheckConstraint(
            "disambiguator IS NULL OR length(disambiguator) > 0",
            name="disambiguator_not_empty",
        ),
        # The exact-match lookup key. Deliberately NOT unique - see the class
        # docstring. Duplicates here are legitimate data, and the matcher's job
        # is to refuse to choose between them.
        Index("ix_entities_entity_type_normalized_name", "entity_type", "normalized_name"),
        # Backs the keyset list ordering exactly, as evidence_items does.
        Index("ix_entities_created_at_id", text("created_at DESC"), text("id DESC")),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Entity id={self.id} type={self.entity_type}>"


class EntityResolution(Base):
    """The assertion that one extraction item refers to one entity.

    This row is a *claim about* derived data, not the data itself. It records
    which item, which entity, in what role, decided how, when, whether it is
    still current, and — if it was corrected — which resolution replaced it.

    Correction is append-only. A wrong resolution is deactivated and stamped
    with `superseded_at`, never edited and never deleted, so "the system thought
    X, then a human corrected it to Y" stays readable forever. Raw evidence and
    the M2 extraction items are untouched by any of it.
    """

    __tablename__ = "entity_resolutions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    extraction_item_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Denormalised from `extraction_items.kind` so the role/kind rule below can
    # be a real CHECK constraint. It cannot drift: the composite foreign key
    # below makes any value other than the item's own kind unstorable.
    source_item_kind: Mapped[ItemKind] = mapped_column(_SOURCE_KIND_TYPE, nullable=False)

    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="RESTRICT"),
        nullable=False,
    )

    role: Mapped[ResolutionRole] = mapped_column(_ROLE_TYPE, nullable=False)

    # A category, not a score. `exact_match` and `manual` describe *how the
    # decision was made*, which is checkable. A probability would not be:
    # nothing in this system calibrates one.
    method: Mapped[ResolutionMethod] = mapped_column(_METHOD_TYPE, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity_resolutions.id", ondelete="RESTRICT"),
        nullable=True,
    )
    superseded_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Why a human did this. Bounded, and never populated from model output.
    note: Mapped[str | None] = mapped_column(String(MAX_NOTE_LENGTH), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # Composite FK to (id, kind). This is what makes the role/kind CHECK
        # below trustworthy: `source_item_kind` cannot be anything other than
        # the referenced item's actual kind.
        ForeignKeyConstraint(
            ["extraction_item_id", "source_item_kind"],
            ["extraction_items.id", "extraction_items.kind"],
            name="fk_entity_resolutions_extraction_item",
            ondelete="RESTRICT",
        ),
        # Each role describes exactly one item kind. This is what keeps "the
        # entity said it" and "the observation is about the entity" from ever
        # collapsing into one another: an `observation` can only ever carry
        # `about`, so it cannot acquire an attribution the evidence never gave
        # it, and a `reported_claim` cannot quietly become a subject-bearing
        # fact about its own speaker.
        CheckConstraint(
            "(role = 'mention' AND source_item_kind = 'entity_mention') "
            "OR (role = 'claimed_by' AND source_item_kind = 'reported_claim') "
            "OR (role = 'about' AND source_item_kind = 'observation')",
            name="role_matches_source_item_kind",
        ),
        # Active means current and unsuperseded; inactive means stamped with
        # when it stopped being true. Mirrors M2's completed_at/status rule.
        CheckConstraint(
            "(is_active AND superseded_at IS NULL AND superseded_by_id IS NULL) "
            "OR (NOT is_active AND superseded_at IS NOT NULL)",
            name="superseded_state_is_consistent",
        ),
        CheckConstraint(
            "superseded_by_id IS NULL OR superseded_by_id <> id",
            name="cannot_supersede_itself",
        ),
        CheckConstraint("note IS NULL OR length(note) > 0", name="note_not_empty"),
        # The load-bearing constraint: one item can have at most one *active*
        # resolution per role. A mention cannot currently refer to two entities
        # at once, so a correction has to supersede rather than accumulate.
        # Partial, so the superseded history is unbounded and unconstrained.
        Index(
            "uq_entity_resolutions_active_item_role",
            "extraction_item_id",
            "role",
            unique=True,
            postgresql_where=text("is_active"),
        ),
        # The profile query's exact key: active resolutions for one entity.
        Index(
            "ix_entity_resolutions_entity_id_active",
            "entity_id",
            postgresql_where=text("is_active"),
        ),
        # Full history for one item, including superseded rows.
        Index("ix_entity_resolutions_extraction_item_id", "extraction_item_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<EntityResolution id={self.id} entity={self.entity_id} "
            f"role={self.role} active={self.is_active}>"
        )
