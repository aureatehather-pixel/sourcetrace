"""The packet's read path. Read-only: this module contains no INSERT, no UPDATE.

Deliberately **not** `intelligence.repository.list_relatable_items`. That query
joins through `EntityResolution` and filters on `RELATABLE_ROLES` and
`RELATABLE_ITEM_KINDS`, which makes it entity-scoped and two-kinds-wide by
construction. Reusing it would couple packet inclusion to M4 relation
eligibility — the coupling this package exists to avoid — and would also starve
the packet of `idea`, `opportunity`, `project` and `event`, none of which can
hold a resolution at all under the `role_matches_source_item_kind` CHECK.

Two queries rather than one join, on purpose. An item can hold at most one active
resolution (the partial unique index on `(extraction_item_id, role)` plus the
role/kind CHECK together guarantee it), so a LEFT JOIN would in fact be safe
today — but "safe today because of two constraints in another module" is a fact
that rots. Fetching entity context separately cannot multiply item rows no matter
what the schema later permits.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence

from sqlalchemy import Row, Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from intelligence_engine.entities.models import Entity, EntityResolution
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.extraction.models import ExtractionItem, ExtractionRun
from intelligence_engine.extraction.schemas import RunStatus
from intelligence_engine.extraction.windowing import Coverage, coverage
from intelligence_engine.packet.schemas import PACKET_ITEM_KINDS

PacketRow = Row[tuple[ExtractionItem, ExtractionRun, EvidenceItem]]
ContextRow = Row[tuple[EntityResolution, Entity]]


def _current_run_ids() -> Select[tuple[uuid.UUID]]:
    """The newest succeeded extraction run per evidence item **per interval**.

    Re-extraction at a bumped extractor version leaves two succeeded runs over
    one note, both with full item sets. Showing both would double-count the
    evidence and make the packet's item counts meaningless, so the packet reads
    one run per interval and says which: the newest that succeeded.

    **The partition is the interval `[window_start, window_end)`, and that is
    load-bearing.** A large source is extracted in several intervals, each its
    own succeeded run over a different slice of one document. Partitioning by
    `evidence_id` alone — which is what this did before windowing existed —
    would rank those runs against each other and keep one, silently discarding
    the rest while the packet reported no truncation at all. That is the exact
    invisible compression this packet exists to prevent.

    Partitioning by an *ordinal* would reintroduce the same loss by a subtler
    route. When an interval overflows it is replaced by children, and a child
    reusing ordinal 3 would rank against the untouched run for the original
    interval 3 — same partition, later timestamp — and quietly evict a sibling
    that is still perfectly valid. Two runs share this partition only when they
    read the same characters, which is the only case where keeping one is right.

    Ties are broken by id so the choice is stable rather than
    insertion-order-dependent, which is what makes the whole packet reproducible.
    """
    ranked = (
        select(
            ExtractionRun.id.label("run_id"),
            func.row_number()
            .over(
                partition_by=(
                    ExtractionRun.evidence_id,
                    ExtractionRun.window_start,
                    ExtractionRun.window_end,
                ),
                order_by=(ExtractionRun.created_at.desc(), ExtractionRun.id.desc()),
            )
            .label("rank"),
        )
        .where(ExtractionRun.status == RunStatus.SUCCEEDED)
        .subquery()
    )
    return select(ranked.c.run_id).where(ranked.c.rank == 1)


class PacketRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def count_items(self) -> int:
        """How many grounded items exist, before any limit is applied.

        Queried separately from `list_items` so a truncated packet can state what
        it truncated *from*. A packet that silently stopped at its limit would be
        an invisible compression, which is the failure the packet exists to
        prevent.
        """
        stmt = (
            select(func.count())
            .select_from(ExtractionItem)
            .where(
                ExtractionItem.extraction_run_id.in_(_current_run_ids()),
                ExtractionItem.kind.in_(sorted(PACKET_ITEM_KINDS)),
            )
        )
        result = await self._session.execute(stmt)
        return int(result.scalar_one())

    async def list_items(self, *, limit: int) -> Sequence[PacketRow]:
        """Grounded items from the current run of every window of every source.

        Ordered by evidence arrival, then evidence id, then **where in the
        source the item's interval begins**, then item position — the same order
        `list_relatable_items` uses, so the packet and M4 disagree about which
        items exist but never about what order they came in.

        `window_start` sits ahead of `position` because position numbers restart
        at zero in every run. Without it, item 0 of a late interval would sort
        beside item 0 of the first and one document would be rendered shuffled —
        the source read out of order, which is worse than unreadable because it
        still looks fine. Ordering on the offset rather than an ordinal also
        keeps a subdivided interval's children in their true place: they sort
        exactly where their parent's characters sit, among siblings that never
        moved.
        """
        stmt = (
            select(ExtractionItem, ExtractionRun, EvidenceItem)
            .join(ExtractionRun, ExtractionRun.id == ExtractionItem.extraction_run_id)
            .join(EvidenceItem, EvidenceItem.id == ExtractionRun.evidence_id)
            .where(
                ExtractionItem.extraction_run_id.in_(_current_run_ids()),
                ExtractionItem.kind.in_(sorted(PACKET_ITEM_KINDS)),
            )
            .order_by(
                EvidenceItem.created_at.asc(),
                EvidenceItem.id.asc(),
                ExtractionRun.window_start.asc(),
                ExtractionItem.position.asc(),
            )
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.all()

    async def source_coverage(self) -> dict[uuid.UUID, Coverage]:
        """Per evidence item: which characters a current succeeded run covers.

        Read straight from the runs table rather than by re-planning intervals,
        so the packet never has to import the planning algorithm and cannot
        drift from it. The intervals are what each run actually read, which is
        why they are stored.

        Answers exactly one question — may this source be presented as fully
        extracted? — so a partially extracted document can be labelled instead
        of quietly appearing whole. It is answered as a tiling rather than as a
        count of successes against a plan size, because the set of intervals is
        dynamic: an overflowing interval is replaced by children covering the
        same characters, and a tiling is indifferent to that while any count is
        not. `length(content)` comes from the evidence row, so the question is
        always asked against the real source rather than against a stored total
        that a later subdivision could have made stale.

        Evidence with no succeeded run at all is absent from the result, not
        present with zero coverage: the caller distinguishes "read nothing" from
        "not in this packet", and only rows that contributed items are here.
        """
        stmt = (
            select(
                ExtractionRun.evidence_id,
                ExtractionRun.window_start,
                ExtractionRun.window_end,
                func.length(EvidenceItem.content),
            )
            .join(EvidenceItem, EvidenceItem.id == ExtractionRun.evidence_id)
            .where(ExtractionRun.id.in_(_current_run_ids()))
        )
        result = await self._session.execute(stmt)

        intervals: dict[uuid.UUID, list[tuple[int, int]]] = defaultdict(list)
        lengths: dict[uuid.UUID, int] = {}
        for evidence_id, start, end, length in result.all():
            intervals[evidence_id].append((int(start), int(end)))
            lengths[evidence_id] = int(length)
        return {
            evidence_id: coverage(ranges, length=lengths[evidence_id])
            for evidence_id, ranges in intervals.items()
        }

    async def list_entity_context(self, item_ids: Sequence[uuid.UUID]) -> Sequence[ContextRow]:
        """Active resolutions for these items, with the entity each names.

        Every role, including `mention` — but a `mention` resolution belongs to
        an `entity_mention` item, which is not a packet kind, so none reach a
        packet item in practice. The filter is left off rather than duplicated
        here: this query answers "what did M3 assert about these items", and
        narrowing it by role would encode M4's eligibility rule in a third place.
        """
        if not item_ids:
            return []
        stmt = (
            select(EntityResolution, Entity)
            .join(Entity, Entity.id == EntityResolution.entity_id)
            .where(
                EntityResolution.extraction_item_id.in_(item_ids),
                EntityResolution.is_active.is_(True),
            )
            .order_by(
                EntityResolution.extraction_item_id.asc(),
                EntityResolution.role.asc(),
                EntityResolution.id.asc(),
            )
        )
        result = await self._session.execute(stmt)
        return result.all()
