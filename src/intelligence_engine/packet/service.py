"""Assembling the packet. Deterministic, read-only, no model call, no writes.

The whole assembly is a pure function of database state plus a limit. Given the
same rows it returns the same packet, in the same order, every time — which is
what makes `pie packet` usable as the input to something else, and what makes a
citation into it checkable.

Nothing here is persisted. There is no `briefs` table and no migration `0007`:
persistence has not earned itself, because nothing yet re-reads an old packet.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from intelligence_engine.entities.models import Entity, EntityResolution
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.extraction.models import ExtractionItem
from intelligence_engine.extraction.payloads import read_statement
from intelligence_engine.extraction.windowing import Coverage
from intelligence_engine.packet.repository import PacketRepository, PacketRow
from intelligence_engine.packet.schemas import (
    DEFAULT_PACKET_LIMIT,
    MAX_PACKET_LIMIT,
    EntityContext,
    EvidencePacket,
    PacketItem,
    PacketSource,
)


class PacketService:
    def __init__(self, session: AsyncSession) -> None:
        self._repository = PacketRepository(session)

    async def assemble(self, *, limit: int = DEFAULT_PACKET_LIMIT) -> EvidencePacket:
        """Every grounded item currently stored, with its provenance.

        The corpus is read whole. There is deliberately no topic or entity
        filter: the measured advantage this package captures is *completeness*,
        and the obvious cheap filter — lexical overlap with a topic string — is
        the one `intelligence/candidates.py` measured and rejected as a gate,
        because the corpus's most valuable pair shares no content words at all.
        Selection earns itself when the limit actually binds, on the same
        measured-precondition rule that has kept embeddings out. Today the whole
        corpus is well inside it.
        """
        if limit < 1 or limit > MAX_PACKET_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_PACKET_LIMIT}")

        total = await self._repository.count_items()
        rows = await self._repository.list_items(limit=limit)
        coverage = await self._repository.source_coverage()

        item_ids = [row[0].id for row in rows]
        context = await self._repository.list_entity_context(item_ids)
        by_item: dict[uuid.UUID, list[EntityContext]] = defaultdict(list)
        for resolution, entity in context:
            by_item[resolution.extraction_item_id].append(_context(resolution, entity))

        items = tuple(
            _item(item, evidence, tuple(by_item.get(item.id, ()))) for item, _run, evidence in rows
        )
        return EvidencePacket(
            items=items,
            sources=_sources(rows, coverage),
            total_items_available=total,
            truncated=total > len(items),
        )


def _context(resolution: EntityResolution, entity: Entity) -> EntityContext:
    return EntityContext(
        entity_id=entity.id,
        canonical_name=entity.canonical_name,
        role=resolution.role,
        resolution_id=resolution.id,
    )


def _item(
    item: ExtractionItem, evidence: EvidenceItem, entities: tuple[EntityContext, ...]
) -> PacketItem:
    """Flatten one stored row into a citable packet entry.

    The payload is read through `read_statement`, which re-validates it against
    the M2 model for its kind. A payload written under an older schema fails
    loudly here rather than becoming a plausible sentence that means something
    else.
    """
    statement = read_statement(item.kind, item.payload, item_label=f"Extraction item {item.id}")
    return PacketItem(
        item_id=item.id,
        kind=item.kind,
        text=statement.text,
        stance=statement.stance,
        claimed_by=statement.claimed_by,
        subject=statement.subject,
        attributes=statement.attributes,
        grounding_quote=item.grounding_quote,
        extraction_item_position=item.position,
        extraction_run_id=item.extraction_run_id,
        evidence_id=evidence.id,
        source_label=evidence.source_label,
        captured_at=evidence.captured_at,
        entities=entities,
    )


def _sources(
    rows: Sequence[PacketRow], coverage: dict[uuid.UUID, Coverage]
) -> tuple[PacketSource, ...]:
    """One entry per evidence item represented, in the order it first appears.

    First-appearance order rather than a re-sort, so the source list and the item
    list tell the same story about arrival.

    A source with no coverage entry cannot occur — every row here came from a
    current succeeded run, which is exactly what coverage is computed over — but
    the fallback reads the source as fully covered rather than raising, because
    a packet is a read path and an assertion here would take down a report over
    a bookkeeping disagreement. The fallback is stated, not silent: it is the
    same reading a single-interval source has always had.
    """
    counts: dict[uuid.UUID, int] = {}
    seen: dict[uuid.UUID, EvidenceItem] = {}
    for _item_row, _run, evidence in rows:
        counts[evidence.id] = counts.get(evidence.id, 0) + 1
        seen.setdefault(evidence.id, evidence)
    return tuple(
        _source(evidence, counts[evidence.id], coverage.get(evidence.id))
        for evidence in seen.values()
    )


def _source(evidence: EvidenceItem, item_count: int, covered: Coverage | None) -> PacketSource:
    length = covered.length if covered is not None else len(evidence.content)
    return PacketSource(
        evidence_id=evidence.id,
        source_type=evidence.source_type,
        source_label=evidence.source_label,
        source_url=evidence.source_url,
        captured_at=evidence.captured_at,
        created_at=evidence.created_at,
        item_count=item_count,
        source_length=length,
        covered_chars=covered.covered_chars if covered is not None else length,
        uncovered=covered.gaps if covered is not None else (),
    )
