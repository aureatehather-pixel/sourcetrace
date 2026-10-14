"""Derived-intelligence tables.

    ExtractionItem ─┐
                    ├─ IntelligenceRelation ─* RelationAnalysisRun
    ExtractionItem ─┘

Two tables, and the split between them is not cosmetic. Pairs are **batched** —
several independent pairs go to the model in one call — so per-relation token
counts do not exist. Attributing a shared call's tokens to each relation would
invent precision. Cost telemetry therefore lives on the run; the assertions live
on the relations.

They are not one aggregate either. An M2 run owns its items (CASCADE) because a
run is one indivisible extraction of one evidence item. An M4 relation is
independently reusable: a second analysis of the same entity reuses every settled
pair and classifies only what is new, so relations outlive the particular run
that produced them. The foreign key is therefore **RESTRICT**, and the run is
provenance rather than ownership.

Every foreign key out of this module is RESTRICT. Derived intelligence never
deletes, and never lets the material it was derived from be deleted underneath
it.
"""

from __future__ import annotations

import datetime as dt
import enum
import uuid

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from intelligence_engine.db.base import Base
from intelligence_engine.evidence.models import CONTENT_HASH_LENGTH
from intelligence_engine.extraction.schemas import ItemKind
from intelligence_engine.intelligence.schemas import (
    MAX_REJECTION_NOTE_LENGTH,
    RELATABLE_ITEM_KINDS,
    AnalysisStatus,
    RelationDirection,
    RelationStatus,
    RelationType,
)


def _values(enum_class: type[enum.Enum]) -> list[str]:
    """Store the enum's *values*, not its member names."""
    return [str(member.value) for member in enum_class]


# native_enum=False renders VARCHAR + CHECK, matching the M2/M3 convention:
# adding a value costs one CHECK rewrite rather than an ALTER TYPE.
_ANALYSIS_STATUS_TYPE = Enum(
    AnalysisStatus,
    native_enum=False,
    length=16,
    name="relation_analysis_status",
    values_callable=lambda e: _values(AnalysisStatus),
)
_RELATION_TYPE = Enum(
    RelationType,
    native_enum=False,
    length=32,
    name="intelligence_relation_type",
    values_callable=lambda e: _values(RelationType),
)
_DIRECTION_TYPE = Enum(
    RelationDirection,
    native_enum=False,
    length=16,
    name="intelligence_relation_direction",
    values_callable=lambda e: _values(RelationDirection),
)
_RELATION_STATUS_TYPE = Enum(
    RelationStatus,
    native_enum=False,
    length=16,
    name="intelligence_relation_status",
    values_callable=lambda e: _values(RelationStatus),
)
# Same value set and width as `extraction_items.kind`, because both item columns
# are joined to it by composite foreign keys. Derived from `ItemKind` so they
# cannot drift apart.
_ITEM_KIND_TYPE = Enum(
    ItemKind,
    native_enum=False,
    length=32,
    name="intelligence_relation_item_kind",
    values_callable=lambda e: _values(ItemKind),
)

MAX_ERROR_MESSAGE_LENGTH = 500

# Rendered into CHECK constraints so the database, not only Pydantic, refuses a
# relation between two items that assert nothing.
_RELATABLE_KIND_SQL = ", ".join(f"'{kind.value}'" for kind in sorted(RELATABLE_ITEM_KINDS))


class RelationAnalysisRun(Base):
    """One analyse-relations operation over one entity.

    Exists only when a model call was actually attempted. An analysis where every
    candidate pair was already settled writes no row: there is nothing to record
    but a decision not to spend money, and the response carries that.
    """

    __tablename__ = "relation_analysis_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # The analysis scope. RESTRICT for the M3 reason: an entity with derived
    # history cannot be deleted out from under it.
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="RESTRICT"),
        nullable=False,
    )

    classifier_name: Mapped[str] = mapped_column(String(64), nullable=False)
    classifier_version: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)

    status: Mapped[AnalysisStatus] = mapped_column(_ANALYSIS_STATUS_TYPE, nullable=False)

    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(
        String(MAX_ERROR_MESSAGE_LENGTH), nullable=True
    )

    # Candidate-generation and cost telemetry. Enough to answer "how much model
    # work did this intelligence operation require"; deliberately not a billing
    # system, and deliberately free of prices, which change.
    theoretical_pairs: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    candidate_pairs: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    reused_pairs: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    classified_pairs: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    batch_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    external_call_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # A finished run has a finish time; a running one does not. Mirrors M2.
        CheckConstraint(
            "(status = 'running' AND completed_at IS NULL) "
            "OR (status <> 'running' AND completed_at IS NOT NULL)",
            name="completed_at_matches_status",
        ),
        CheckConstraint(
            "(status = 'failed' AND error_code IS NOT NULL) "
            "OR (status <> 'failed' AND error_code IS NULL)",
            name="error_code_matches_status",
        ),
        CheckConstraint("schema_version > 0", name="schema_version_positive"),
        CheckConstraint(
            "theoretical_pairs >= 0 AND candidate_pairs >= 0 AND reused_pairs >= 0 "
            "AND classified_pairs >= 0 AND batch_count >= 0 AND external_call_count >= 0",
            name="counters_non_negative",
        ),
        # Narrowing can only ever reduce. If this ever fires, candidate
        # generation is broken in a way that costs money.
        CheckConstraint(
            "candidate_pairs <= theoretical_pairs", name="candidates_within_theoretical"
        ),
        CheckConstraint("classified_pairs <= candidate_pairs", name="classified_within_candidates"),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0", name="input_tokens_non_negative"
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0", name="output_tokens_non_negative"
        ),
        Index(
            "ix_relation_analysis_runs_entity_id_created_at", "entity_id", text("created_at DESC")
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RelationAnalysisRun id={self.id} entity={self.entity_id} status={self.status}>"


class IntelligenceRelation(Base):
    """The assertion that two grounded items stand in some relation.

    This is a *new* claim, not a stored fact drawn out of the evidence. Neither
    source item states it; the classifier inferred it, and it can be wrong. It
    therefore carries the same discipline M2 applies to extraction runs and M3 to
    resolutions: exact source identifiers, the classifier name/version/schema that
    produced it, the provider and model, a fingerprint of what was actually sent,
    and a review state.

    **Canonical ordering** (`left_extraction_item_id < right_extraction_item_id`,
    enforced by CHECK) means one pair has exactly one row. Without it,
    "A contradicts B" and "B contradicts A" would be two rows asserting one thing
    and free to disagree. Direction, where it matters, is a column.
    """

    __tablename__ = "intelligence_relations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Provenance, not ownership - see the module docstring for why this is not a
    # CASCADE the way M2's run -> item is.
    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "relation_analysis_runs.id",
            # Named explicitly: the convention-generated name is exactly 63
            # characters, which is PostgreSQL's identifier limit. One more
            # character anywhere and it would silently truncate.
            name="fk_intelligence_relations_analysis_run",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    # The scope under which these two items were compared. Both items being
    # resolved to this entity is *why* the pair was a candidate at all, so it is
    # an input to the assertion rather than a denormalisation for speed.
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="RESTRICT"),
        nullable=False,
    )

    left_extraction_item_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    right_extraction_item_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Denormalised from `extraction_items.kind` so the relatable-kind CHECKs below
    # are real constraints. They cannot drift: the composite foreign keys make any
    # value other than the referenced item's own kind unstorable - the same
    # mechanism M3 uses for `source_item_kind`.
    left_item_kind: Mapped[ItemKind] = mapped_column(_ITEM_KIND_TYPE, nullable=False)
    right_item_kind: Mapped[ItemKind] = mapped_column(_ITEM_KIND_TYPE, nullable=False)

    relation_type: Mapped[RelationType] = mapped_column(_RELATION_TYPE, nullable=False)
    direction: Mapped[RelationDirection] = mapped_column(_DIRECTION_TYPE, nullable=False)

    # Derived reasoning. Deliberately NOT called a quote, and never checked
    # against the evidence the way M2 checks grounding quotes - it is the
    # classifier's own words about two items, and the schema says so.
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    classifier_name: Mapped[str] = mapped_column(String(64), nullable=False)
    classifier_version: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)

    # The *requested* generator identity, not the one a response reported. Part
    # of `unique_settled_pair_per_classifier` below and of the reuse lookup, so
    # NOT NULL: a nullable column inside a unique constraint stops conflicting
    # with anything in PostgreSQL, which would silently repeal the one guarantee
    # that constraint exists to give.
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)

    # SHA-256 of the exact pair block the classifier was given. M2 extraction
    # items are immutable - a re-extraction creates new rows rather than editing
    # old ones - so the two item ids already determine the content. This is the
    # auditable proof of that, and it is part of the reuse lookup: if it ever
    # disagreed, the lookup would miss and the unique constraint below would
    # raise rather than let stale intelligence be served quietly.
    input_fingerprint: Mapped[str] = mapped_column(String(CONTENT_HASH_LENGTH), nullable=False)

    status: Mapped[RelationStatus] = mapped_column(
        _RELATION_STATUS_TYPE, nullable=False, server_default=text("'current'")
    )
    rejected_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Why a human rejected it. Bounded, and never populated from model output.
    rejection_note: Mapped[str | None] = mapped_column(
        String(MAX_REJECTION_NOTE_LENGTH), nullable=True
    )

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["left_extraction_item_id", "left_item_kind"],
            ["extraction_items.id", "extraction_items.kind"],
            name="fk_intelligence_relations_left_item",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["right_extraction_item_id", "right_item_kind"],
            ["extraction_items.id", "extraction_items.kind"],
            name="fk_intelligence_relations_right_item",
            ondelete="RESTRICT",
        ),
        # One pair, one row. Also forbids a relation from an item to itself,
        # which would be a tautology with an explanation attached.
        CheckConstraint(
            "left_extraction_item_id < right_extraction_item_id",
            name="pair_is_canonically_ordered",
        ),
        # Only assertion-bearing items can stand in a relation. An entity mention
        # names a thing; naming a thing cannot qualify or contradict anything.
        CheckConstraint(
            f"left_item_kind IN ({_RELATABLE_KIND_SQL})", name="left_item_kind_is_relatable"
        ),
        CheckConstraint(
            f"right_item_kind IN ({_RELATABLE_KIND_SQL})", name="right_item_kind_is_relatable"
        ),
        # Directionality is part of the assertion's meaning, so it is a database
        # truth. `qualifies` must say which way; nothing else may.
        CheckConstraint(
            "(relation_type = 'qualifies' AND direction <> 'symmetric') "
            "OR (relation_type <> 'qualifies' AND direction = 'symmetric')",
            name="direction_matches_relation_type",
        ),
        # A meaningful relation surfaced without a reason is an unexplained
        # assertion about someone. `no_meaningful_relation` needs no explanation
        # and is never surfaced, so it may omit one.
        CheckConstraint(
            "(relation_type = 'no_meaningful_relation') "
            "OR (explanation IS NOT NULL AND length(explanation) > 0)",
            name="meaningful_relation_is_explained",
        ),
        CheckConstraint(
            "explanation IS NULL OR length(explanation) > 0", name="explanation_not_empty"
        ),
        CheckConstraint(
            "(status = 'current' AND rejected_at IS NULL) "
            "OR (status = 'rejected' AND rejected_at IS NOT NULL)",
            name="rejected_state_is_consistent",
        ),
        CheckConstraint(
            "rejection_note IS NULL OR length(rejection_note) > 0",
            name="rejection_note_not_empty",
        ),
        CheckConstraint(
            f"input_fingerprint ~ '^[0-9a-f]{{{CONTENT_HASH_LENGTH}}}$'",
            name="input_fingerprint_is_sha256_hex",
        ),
        CheckConstraint("schema_version > 0", name="schema_version_positive"),
        # The load-bearing constraint, and the whole reuse mechanism: one settled
        # answer per pair per classifier version per generator. Deliberately
        # covers rejected rows too - a rejected relation is a settled question,
        # and regenerating it would pay for the same wrong answer again. A
        # version bump is what re-opens it.
        #
        # `provider` and `model` are in the key because the reuse lookup reads
        # them, and the two must agree exactly. When the lookup alone had them,
        # classifying a pair with a second model missed the lookup and then
        # collided here - so "ask the other model" raised an integrity error
        # instead of asking. Two models may now hold two settled answers for one
        # pair, which is what a model comparison is; neither may hold two.
        UniqueConstraint(
            "left_extraction_item_id",
            "right_extraction_item_id",
            "classifier_name",
            "classifier_version",
            "schema_version",
            "provider",
            "model",
            name="unique_settled_pair_per_classifier",
        ),
        # The intelligence read model's exact key: current relations for one entity.
        Index(
            "ix_intelligence_relations_entity_id_current",
            "entity_id",
            postgresql_where=text("status = 'current'"),
        ),
        Index("ix_intelligence_relations_analysis_run_id", "analysis_run_id"),
        Index("ix_intelligence_relations_left_extraction_item_id", "left_extraction_item_id"),
        Index("ix_intelligence_relations_right_extraction_item_id", "right_extraction_item_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<IntelligenceRelation id={self.id} type={self.relation_type} "
            f"direction={self.direction} status={self.status}>"
        )
