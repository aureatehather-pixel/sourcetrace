"""Persistence operations for evidence items.

Only the three operations M1 actually needs. No generic base repository - there
is exactly one aggregate, and an abstraction over one thing is noise.

The repository never commits. Transaction boundaries belong to the caller.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import literal, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from intelligence_engine.core.pagination import Cursor
from intelligence_engine.evidence.models import EvidenceItem


class EvidenceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(
        self,
        *,
        content: str,
        content_hash: str,
        source_type: str | None,
        source_label: str | None,
        source_url: str | None,
        captured_at: dt.datetime | None,
        metadata: dict[str, Any],
    ) -> EvidenceItem:
        """Stage a new evidence item. Not durable until the caller commits."""
        item = EvidenceItem(
            id=uuid.uuid4(),
            content=content,
            content_hash=content_hash,
            source_type=source_type,
            source_label=source_label,
            source_url=source_url,
            captured_at=captured_at,
            meta=metadata,
        )
        self._session.add(item)
        return item

    async def get(self, evidence_id: uuid.UUID) -> EvidenceItem | None:
        return await self._session.get(EvidenceItem, evidence_id)

    async def find_by_content_hash(self, content_hash: str) -> list[EvidenceItem]:
        """Every item whose stored content hashes to ``content_hash``, newest first.

        Deliberately a list, not an optional. The fingerprint is not unique and
        must not become so: identical text genuinely observed from two sources is
        two observations, and collapsing them would destroy provenance. Callers
        that need one row have to decide what several mean.
        """
        stmt = (
            select(EvidenceItem)
            .where(EvidenceItem.content_hash == content_hash)
            .order_by(EvidenceItem.created_at.desc(), EvidenceItem.id.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_page(self, *, limit: int, after: Cursor | None = None) -> list[EvidenceItem]:
        """Return up to ``limit`` items ordered created_at DESC, id DESC.

        ``after`` is the sort key of the last row already seen; the row-value
        comparison gives a strict total order with no gaps or duplicates.
        """
        stmt = (
            select(EvidenceItem)
            .order_by(EvidenceItem.created_at.desc(), EvidenceItem.id.desc())
            .limit(limit)
        )
        if after is not None:
            stmt = stmt.where(
                tuple_(EvidenceItem.created_at, EvidenceItem.id)
                < tuple_(literal(after.created_at), literal(after.id))
            )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
