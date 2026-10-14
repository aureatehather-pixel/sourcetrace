"""Persistence for extraction runs and their items.

Like the evidence repository, this one never commits — transaction boundaries
belong to the service.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from intelligence_engine.extraction.models import ExtractionItem, ExtractionRun
from intelligence_engine.extraction.schemas import ItemKind, RunStatus


class ExtractionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def start_run(
        self,
        *,
        evidence_id: uuid.UUID,
        input_content_hash: str,
        extractor_name: str,
        extractor_version: str,
        schema_version: int,
        provider: str,
        model: str,
        started_at: dt.datetime,
        window_start: int = 0,
        window_end: int,
    ) -> ExtractionRun:
        """Stage a `running` run. The caller commits it before calling out.

        `window_start` defaults to the start of the document, so a caller that
        does not divide its evidence records exactly what it recorded before
        windowing existed. `window_end` has no honest default — it is the length
        of the text actually sent — so it is required.

        `provider` and `model` are the *requested* identity and are written here,
        at the attempt, rather than on success from the response. Two reasons,
        and the first is the one that matters:

        - **Reuse compares them, and reuse runs before the call.** A value
          recorded from the response is knowable only after the spending it was
          meant to prevent.
        - A failed attempt then says which generator failed, instead of leaving
          the two columns null exactly when the question is most interesting.
        """
        run = ExtractionRun(
            id=uuid.uuid4(),
            evidence_id=evidence_id,
            input_content_hash=input_content_hash,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
            schema_version=schema_version,
            status=RunStatus.RUNNING,
            provider=provider,
            model=model,
            started_at=started_at,
            window_start=window_start,
            window_end=window_end,
        )
        self._session.add(run)
        return run

    def add_item(
        self,
        *,
        run: ExtractionRun,
        kind: ItemKind,
        payload: dict[str, Any],
        grounding_quote: str,
        position: int,
    ) -> ExtractionItem:
        item = ExtractionItem(
            id=uuid.uuid4(),
            kind=kind,
            payload=payload,
            grounding_quote=grounding_quote,
            position=position,
        )
        # Appended through the relationship rather than session.add(): the FK
        # is set by the cascade, and run.items stays correct in memory so the
        # response does not need a re-query.
        run.items.append(item)
        return item

    async def get_run(self, run_id: uuid.UUID) -> ExtractionRun | None:
        return await self._session.get(ExtractionRun, run_id)

    async def get_item(self, item_id: uuid.UUID) -> ExtractionItem | None:
        """One extraction item by id.

        Read-only, and the only door M3's entity resolution goes through to
        reach M2 data. Extraction items stay owned by this module; nothing
        outside it constructs, mutates or deletes one.
        """
        return await self._session.get(ExtractionItem, item_id)

    async def find_reusable_run(
        self,
        *,
        evidence_id: uuid.UUID,
        input_content_hash: str,
        extractor_name: str,
        extractor_version: str,
        schema_version: int,
        provider: str,
        model: str,
        window_start: int = 0,
        window_end: int,
    ) -> ExtractionRun | None:
        """The newest succeeded run for this exact pipeline, input and interval.

        Every component of the key must match. A different content hash, a
        bumped extractor version or a bumped schema version all mean the stored
        output no longer describes what would be produced now.

        **`provider` and `model` are part of that key**, and were the one part
        missing. Everything else in it describes what was *asked*; the generator
        that answered decides what comes back. Without them a run extracted by
        one vendor answers a lookup made while configured for another, and a
        comparison between two models silently measures the first one twice.
        They are compared against the requested identity `start_run` recorded,
        never against a response, so the two sides of the comparison are the same
        kind of fact.

        The interval `[window_start, window_end)` is part of that key rather
        than an afterthought: without it, the succeeded run for the first
        interval would answer a lookup for the fourth, and a document read in
        six passes would extract one of them and report itself done. The content
        hash alone will not do the job either — two intervals of one document
        could in principle carry identical text, and they are still two
        different parts of it.

        Offsets rather than an ordinal, because ordinals move. When an interval
        overflows it is replaced by children, and under ordinal keying a child
        would inherit a number an untouched sibling already holds and be served
        that sibling's stored items. Offsets cannot collide without describing
        the same characters, which is precisely when reuse is correct.
        """
        stmt = (
            select(ExtractionRun)
            .where(
                ExtractionRun.evidence_id == evidence_id,
                ExtractionRun.window_start == window_start,
                ExtractionRun.window_end == window_end,
                ExtractionRun.input_content_hash == input_content_hash,
                ExtractionRun.extractor_name == extractor_name,
                ExtractionRun.extractor_version == extractor_version,
                ExtractionRun.schema_version == schema_version,
                ExtractionRun.provider == provider,
                ExtractionRun.model == model,
                ExtractionRun.status == RunStatus.SUCCEEDED,
            )
            .order_by(ExtractionRun.created_at.desc(), ExtractionRun.id.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def list_runs_for_evidence(
        self, evidence_id: uuid.UUID, *, limit: int
    ) -> list[ExtractionRun]:
        """All attempts for one evidence item, newest first — successes and failures."""
        stmt = (
            select(ExtractionRun)
            .where(ExtractionRun.evidence_id == evidence_id)
            .order_by(ExtractionRun.created_at.desc(), ExtractionRun.id.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
