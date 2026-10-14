"""API request/response models.

Kept separate from the ORM model on purpose: the wire contract and the storage
schema evolve for different reasons.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Annotated, Any

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator

MAX_CONTENT_LENGTH = 1_000_000  # ~1MB of text; larger material belongs in a future blob store

SourceType = Annotated[str, Field(min_length=1, max_length=64)]


class EvidenceCreate(BaseModel):
    """Inbound evidence. This is a trust boundary - everything is validated here."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CONTENT_LENGTH,
        description="Raw evidence, stored verbatim.",
    )
    source_type: SourceType | None = None
    source_label: str | None = Field(default=None, max_length=500)
    source_url: AnyHttpUrl | None = None
    captured_at: dt.datetime | None = Field(
        default=None,
        description="When the evidence occurred/was captured. Must be timezone-aware.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """Reject whitespace-only content, but never rewrite accepted content."""
        if not value.strip():
            raise ValueError("content must contain at least one non-whitespace character")
        return value

    @field_validator("captured_at")
    @classmethod
    def _require_timezone(cls, value: dt.datetime | None) -> dt.datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware (e.g. 2026-08-22T15:30:00+08:00)")
        return value


class EvidenceRead(BaseModel):
    """The persisted representation returned to clients."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    content: str
    content_hash: str
    source_type: str | None
    source_label: str | None
    source_url: str | None
    captured_at: dt.datetime | None
    created_at: dt.datetime
    metadata: dict[str, Any]

    @classmethod
    def from_model(cls, item: Any) -> EvidenceRead:
        """Build from an EvidenceItem, mapping the ``meta`` attribute to ``metadata``.

        The ORM attribute cannot be called ``metadata`` - that name belongs to
        SQLAlchemy's declarative machinery - so the mapping is explicit.
        """
        return cls(
            id=item.id,
            content=item.content,
            content_hash=item.content_hash,
            source_type=item.source_type,
            source_label=item.source_label,
            source_url=item.source_url,
            captured_at=item.captured_at,
            created_at=item.created_at,
            metadata=item.meta,
        )


class EvidencePage(BaseModel):
    """A bounded page of evidence, newest first."""

    items: list[EvidenceRead]
    next_cursor: str | None = Field(
        default=None, description="Pass as ?cursor= to fetch the next page. Null when exhausted."
    )
