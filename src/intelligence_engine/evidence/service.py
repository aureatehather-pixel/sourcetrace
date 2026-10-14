"""Evidence application behaviour.

Owns the commit boundary. Creating evidence either fully persists or fails
cleanly - there is no intermediate state where the API reports success but the
row is not durable.
"""

from __future__ import annotations

import logging
import time
import uuid

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from intelligence_engine.core.errors import EvidenceNotFound, PersistenceError
from intelligence_engine.core.pagination import decode_cursor, encode_cursor
from intelligence_engine.evidence.fingerprint import content_fingerprint
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.evidence.repository import EvidenceRepository
from intelligence_engine.evidence.schemas import EvidenceCreate, EvidencePage, EvidenceRead

logger = logging.getLogger(__name__)


class EvidenceService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repository = EvidenceRepository(session)

    async def create(self, payload: EvidenceCreate) -> EvidenceRead:
        """Fingerprint and durably persist raw evidence.

        Content is stored exactly as supplied. No cleaning, no rewriting - the
        whole point of the system is that the original survives.
        """
        started = time.perf_counter()
        fingerprint = content_fingerprint(payload.content)

        item = self._repository.add(
            content=payload.content,
            content_hash=fingerprint,
            source_type=payload.source_type,
            source_label=payload.source_label,
            source_url=str(payload.source_url) if payload.source_url else None,
            captured_at=payload.captured_at,
            metadata=payload.metadata,
        )

        try:
            await self._session.commit()
        except SQLAlchemyError as exc:
            await self._session.rollback()
            logger.exception(
                "evidence.create.failed",
                extra={"content_hash": fingerprint, "content_length": len(payload.content)},
            )
            raise PersistenceError("Failed to persist evidence item") from exc

        await self._session.refresh(item)  # pick up server-generated created_at

        logger.info(
            "evidence.created",
            extra={
                "evidence_id": str(item.id),
                "content_hash": item.content_hash,
                "content_length": len(item.content),
                "source_type": item.source_type,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return EvidenceRead.from_model(item)

    async def get(self, evidence_id: uuid.UUID) -> EvidenceRead:
        item: EvidenceItem | None = await self._repository.get(evidence_id)
        if item is None:
            raise EvidenceNotFound(evidence_id)
        return EvidenceRead.from_model(item)

    async def find_by_content(self, content: str) -> list[EvidenceRead]:
        """Evidence already on file whose content is byte-identical to ``content``.

        A read, and only a read. It exists so a caller re-running a pipeline over
        the same note can tell "already ingested" from "new", instead of silently
        creating a second record of one observation. It does not deduplicate:
        the store still accepts identical content from two sources, because that
        is two observations.
        """
        items = await self._repository.find_by_content_hash(content_fingerprint(content))
        return [EvidenceRead.from_model(item) for item in items]

    async def list(self, *, limit: int, cursor: str | None = None) -> EvidencePage:
        """Return one bounded page, newest first.

        Fetches limit+1 rows so 'is there another page' is answered without a
        second query and without a count.
        """
        after = decode_cursor(cursor) if cursor else None
        rows = await self._repository.list_page(limit=limit + 1, after=after)

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        next_cursor = (
            encode_cursor(page_rows[-1].created_at, page_rows[-1].id)
            if has_more and page_rows
            else None
        )
        return EvidencePage(
            items=[EvidenceRead.from_model(row) for row in page_rows],
            next_cursor=next_cursor,
        )
