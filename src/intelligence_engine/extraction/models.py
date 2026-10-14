"""Derived-information tables.

Two tables, one relationship, no workflow engine:

    EvidenceItem  1---*  ExtractionRun  1---*  ExtractionItem

`extraction_runs` is execution history — every attempt, including the failures.
`extraction_items` is the validated structured output of a successful attempt.
Neither ever writes back into `evidence_items`.

Delete semantics differ between the two foreign keys, on purpose:

- `extraction_runs.evidence_id` is **RESTRICT**. Evidence cannot be deleted out
  from under its own extraction history; losing the provenance chain silently
  is exactly the failure this system exists to prevent.
- `extraction_items.extraction_run_id` is **CASCADE**. An item has no meaning
  apart from its run, so it is part of that aggregate.
"""

from __future__ import annotations

import datetime as dt
import enum
import uuid
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from intelligence_engine.db.base import Base
from intelligence_engine.evidence.models import CONTENT_HASH_LENGTH
from intelligence_engine.extraction.schemas import ItemKind, RunStatus


# native_enum=False renders VARCHAR + CHECK rather than a PostgreSQL ENUM type.
# Adding a value then costs one CHECK rewrite instead of ALTER TYPE, which
# cannot run inside a transaction with other DDL on older servers.
def _values(enum_class: type[enum.Enum]) -> list[str]:
    """Store the enum's *values*, not its member names."""
    return [str(member.value) for member in enum_class]


_STATUS_TYPE = Enum(
    RunStatus,
    native_enum=False,
    length=16,
    name="extraction_run_status",
    values_callable=lambda e: _values(RunStatus),
)
_KIND_TYPE = Enum(
    ItemKind,
    native_enum=False,
    length=32,
    name="extraction_item_kind",
    values_callable=lambda e: _values(ItemKind),
)

MAX_ERROR_MESSAGE_LENGTH = 500


class ExtractionRun(Base):
    """One attempt to extract structure from one evidence item."""

    __tablename__ = "extraction_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evidence_items.id", ondelete="RESTRICT"),
        nullable=False,
    )

    # The fingerprint of the text *actually sent* at extraction time. Together
    # with the version columns this is the full cache key, and the audit trail
    # for what text the output actually describes.
    #
    # For a document that fits one window this is the whole evidence content
    # hash, exactly as before windowing existed. For a windowed document it is
    # the hash of that window's slice, which is what lets one window be reused
    # while another is retried.
    input_content_hash: Mapped[str] = mapped_column(String(CONTENT_HASH_LENGTH), nullable=False)

    # --- The interval this run read ---------------------------------------
    # A window is extraction mechanics, not an evidence source. These two
    # columns exist so one `EvidenceItem` can be extracted in several passes
    # while remaining exactly one logical source: `evidence_id` above is
    # unchanged and still singular, which is what keeps M4's cross-evidence
    # filter (`first.evidence_id == second.evidence_id`) suppressing pairs
    # drawn from two intervals of the same conversation.
    #
    # Half-open `[start, end)` over the parent content, and **the durable
    # identity of the work**. Stored rather than recomputed so provenance is
    # auditable from the row alone: the interval's text is
    # `evidence.content[window_start:window_end]`, and that slice must hash to
    # `input_content_hash`. That equality is the mechanical proof that
    # per-interval grounding implies source-level grounding.
    #
    # There is deliberately no ordinal and no plan size here. An interval that
    # overflows is replaced by children covering exactly its characters, so the
    # set of leaves is dynamic: any stored `window_index` would be renumbered
    # by a sibling's subdivision, silently redefining work already paid for,
    # and any stored `window_count` would be a snapshot of a plan that no
    # longer exists. Offsets have neither problem — they mean the same thing
    # before and after any subdivision anywhere else in the source.
    #
    # Defaults describe a run over a whole small document, which is what most
    # evidence is and what every pre-windowing row was.
    window_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    window_end: Mapped[int] = mapped_column(Integer, nullable=False)

    extractor_name: Mapped[str] = mapped_column(String(64), nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)

    status: Mapped[RunStatus] = mapped_column(_STATUS_TYPE, nullable=False)

    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(
        String(MAX_ERROR_MESSAGE_LENGTH), nullable=True
    )

    # Cost awareness starts here. Not a billing system - two integers the
    # provider already hands us.
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    items: Mapped[list[ExtractionItem]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ExtractionItem.position",
        lazy="selectin",
    )

    __table_args__ = (
        # A finished run has a finish time; a running one does not. Without
        # this a crashed process could leave a 'succeeded' row with no
        # completed_at and nothing would notice.
        CheckConstraint(
            "(status = 'running' AND completed_at IS NULL) "
            "OR (status <> 'running' AND completed_at IS NOT NULL)",
            name="completed_at_matches_status",
        ),
        # Failure must say why; success must not pretend it failed.
        CheckConstraint(
            "(status = 'failed' AND error_code IS NOT NULL) "
            "OR (status <> 'failed' AND error_code IS NULL)",
            name="error_code_matches_status",
        ),
        CheckConstraint(
            f"input_content_hash ~ '^[0-9a-f]{{{CONTENT_HASH_LENGTH}}}$'",
            name="input_content_hash_is_sha256_hex",
        ),
        CheckConstraint("schema_version > 0", name="schema_version_positive"),
        # An interval is a real range inside its parent. Enforced in the
        # database because a row violating either of these would make the
        # tiling invariant — and therefore both the grounding argument and the
        # completeness computation — silently false.
        CheckConstraint("window_start >= 0", name="window_start_non_negative"),
        CheckConstraint("window_end >= window_start", name="window_end_after_start"),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0", name="input_tokens_non_negative"
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0", name="output_tokens_non_negative"
        ),
        # Deliberately NOT unique on (evidence_id, extractor_name,
        # extractor_version): retries are legitimate history. A timeout, then a
        # malformed response, then a success is three rows, all worth keeping.
        # Partial index over successes only - the reuse lookup's exact key.
        #
        # Keyed on the interval, `(window_start, window_end)`. An ordinal here
        # would be actively unsafe once intervals can subdivide: a child
        # reusing ordinal 3 would answer a lookup for the untouched original
        # window 3, serving one interval's items as another's. The offsets
        # cannot collide that way, because two runs sharing them read the same
        # characters by definition.
        Index(
            "ix_extraction_runs_reuse",
            "evidence_id",
            "window_start",
            "window_end",
            "input_content_hash",
            "extractor_name",
            "extractor_version",
            "schema_version",
            # The generator is part of the reuse key, so it belongs in the
            # index that serves the lookup. Nullable here and deliberately not
            # tightened: a run that failed before a provider was recorded is
            # real history, and this is a lookup index rather than a
            # constraint, so a null simply never matches.
            "provider",
            "model",
            postgresql_where=text("status = 'succeeded'"),
        ),
        Index("ix_extraction_runs_evidence_id_created_at", "evidence_id", text("created_at DESC")),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ExtractionRun id={self.id} status={self.status} evidence={self.evidence_id}>"


class ExtractionItem(Base):
    """One validated structured item produced by a successful run."""

    __tablename__ = "extraction_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    extraction_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("extraction_runs.id", ondelete="CASCADE"),
        nullable=False,
    )

    kind: Mapped[ItemKind] = mapped_column(_KIND_TYPE, nullable=False)

    # JSONB because the per-kind shapes are still moving. It holds a *validated*
    # payload - the strict Pydantic model for this kind - never raw model
    # output. Concepts that stabilise get promoted to columns, as grounding_quote
    # already was.
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # Promoted out of the payload: every item has one, humans read it to audit
    # the item, and it is the field most likely to be queried.
    grounding_quote: Mapped[str] = mapped_column(Text, nullable=False)

    # Model output order is meaningful and UUIDs are not ordered; created_at is
    # identical across a single transaction. Without this, item order would be
    # nondeterministic.
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    run: Mapped[ExtractionRun] = relationship(back_populates="items")

    __table_args__ = (
        CheckConstraint("length(grounding_quote) > 0", name="grounding_quote_not_empty"),
        CheckConstraint("jsonb_typeof(payload) = 'object'", name="payload_is_object"),
        CheckConstraint("position >= 0", name="position_non_negative"),
        UniqueConstraint("extraction_run_id", "position", name="unique_position_per_run"),
        # Redundant on its own - `id` is already the primary key - and present
        # solely so M3's `entity_resolutions` can carry a composite foreign key
        # on (id, kind). That is what lets the database, not just the service,
        # refuse a `claimed_by` resolution pointed at an `observation`. It adds
        # no new truth about this table and changes no M2 semantics.
        UniqueConstraint("id", "kind", name="unique_id_kind"),
        Index("ix_extraction_items_extraction_run_id", "extraction_run_id"),
        Index("ix_extraction_items_kind", "kind"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ExtractionItem id={self.id} kind={self.kind} run={self.extraction_run_id}>"
