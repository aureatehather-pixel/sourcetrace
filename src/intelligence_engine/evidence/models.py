"""The evidence_items table.

An evidence item is original raw material supplied to the system. It is treated
as immutable: the application offers no update path, and future derived
information will live in its own tables rather than overwriting this one.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from intelligence_engine.db.base import Base

CONTENT_HASH_LENGTH = 64  # SHA-256, lowercase hex


class EvidenceItem(Base):
    __tablename__ = "evidence_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(CONTENT_HASH_LENGTH), nullable=False)

    # Provenance. Unknown stays unknown - nothing here is fabricated.
    source_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_label: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    captured_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Sparse ingestion/source context only. Concepts that become stable and
    # frequently queried get promoted into real columns.
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    __table_args__ = (
        # Database-level truths, not merely Pydantic-level ones.
        CheckConstraint("length(content) > 0", name="content_not_empty"),
        CheckConstraint(
            f"content_hash ~ '^[0-9a-f]{{{CONTENT_HASH_LENGTH}}}$'",
            name="content_hash_is_sha256_hex",
        ),
        CheckConstraint("jsonb_typeof(metadata) = 'object'", name="metadata_is_object"),
        # Deliberately NOT unique: identical text from different sources is two
        # independent observations and both must survive with their provenance.
        Index("ix_evidence_items_content_hash", "content_hash"),
        # Backs the keyset pagination order exactly.
        Index(
            "ix_evidence_items_created_at_id",
            text("created_at DESC"),
            text("id DESC"),
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - never includes content
        return f"<EvidenceItem id={self.id} hash={self.content_hash[:12]}...>"
