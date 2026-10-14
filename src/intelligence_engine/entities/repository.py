"""Persistence for entities and resolutions.

Like the evidence and extraction repositories, this one never commits.
Transaction boundaries belong to the service.

One invariant is owned here rather than by callers: `normalized_name` is
computed inside `add`, from `canonical_name`, on the single write path. A caller
cannot supply an inconsistent key because a caller is never asked for one.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence

from sqlalchemy import Row, literal, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from intelligence_engine.core.pagination import Cursor
from intelligence_engine.entities.matching import normalize_name
from intelligence_engine.entities.models import Entity, EntityResolution
from intelligence_engine.entities.schemas import (
    EntityType,
    ResolutionMethod,
    ResolutionRole,
)
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.extraction.models import ExtractionItem, ExtractionRun
from intelligence_engine.extraction.schemas import ItemKind

# A same-name candidate set larger than this is pathological; the matcher
# refuses at two anyway, so the query never needs to drag back more.
MAX_CANDIDATES = 50

ProfileRow = Row[tuple[EntityResolution, ExtractionItem, ExtractionRun, EvidenceItem]]


class EntityRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(
        self,
        *,
        entity_type: EntityType,
        canonical_name: str,
        disambiguator: str | None,
    ) -> Entity:
        """Stage a new entity. Not durable until the caller commits."""
        entity = Entity(
            id=uuid.uuid4(),
            entity_type=entity_type,
            canonical_name=canonical_name,
            normalized_name=normalize_name(canonical_name),
            disambiguator=disambiguator,
        )
        self._session.add(entity)
        return entity

    async def get(self, entity_id: uuid.UUID) -> Entity | None:
        return await self._session.get(Entity, entity_id)

    async def find_candidates(
        self, *, normalized_name: str, entity_type: EntityType | None
    ) -> list[Entity]:
        """Every entity sharing this exact match key.

        `entity_type` is `None` only for `claimed_by` matching, where the
        attribution string carries no type of its own. That search is therefore
        wider and more likely to come back ambiguous — which is the conservative
        failure, not a silent cross-type merge.
        """
        stmt = select(Entity).where(Entity.normalized_name == normalized_name)
        if entity_type is not None:
            stmt = stmt.where(Entity.entity_type == entity_type)
        stmt = stmt.order_by(Entity.created_at.asc(), Entity.id.asc()).limit(MAX_CANDIDATES)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_page(
        self,
        *,
        limit: int,
        after: Cursor | None = None,
        entity_type: EntityType | None = None,
    ) -> list[Entity]:
        """Up to ``limit`` entities ordered created_at DESC, id DESC."""
        stmt = select(Entity).order_by(Entity.created_at.desc(), Entity.id.desc()).limit(limit)
        if entity_type is not None:
            stmt = stmt.where(Entity.entity_type == entity_type)
        if after is not None:
            stmt = stmt.where(
                tuple_(Entity.created_at, Entity.id)
                < tuple_(literal(after.created_at), literal(after.id))
            )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class ResolutionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(
        self,
        *,
        extraction_item_id: uuid.UUID,
        source_item_kind: ItemKind,
        entity_id: uuid.UUID,
        role: ResolutionRole,
        method: ResolutionMethod,
        note: str | None,
    ) -> EntityResolution:
        """Stage a new active resolution."""
        resolution = EntityResolution(
            id=uuid.uuid4(),
            extraction_item_id=extraction_item_id,
            source_item_kind=source_item_kind,
            entity_id=entity_id,
            role=role,
            method=method,
            is_active=True,
            note=note,
        )
        self._session.add(resolution)
        return resolution

    def deactivate(
        self,
        resolution: EntityResolution,
        *,
        at: dt.datetime,
        superseded_by_id: uuid.UUID | None,
        note: str | None,
    ) -> None:
        """Retire a resolution in place.

        The row keeps its entity, role, method and creation time: it remains a
        faithful record of what the system asserted and when. Only its currency
        changes. `superseded_by_id` is None for a detachment — the operator said
        "not that entity" without naming another.
        """
        resolution.is_active = False
        resolution.superseded_at = at
        resolution.superseded_by_id = superseded_by_id
        if note is not None:
            resolution.note = note

    async def get(self, resolution_id: uuid.UUID) -> EntityResolution | None:
        return await self._session.get(EntityResolution, resolution_id)

    async def get_active_for_item(
        self, *, extraction_item_id: uuid.UUID, role: ResolutionRole
    ) -> EntityResolution | None:
        stmt = select(EntityResolution).where(
            EntityResolution.extraction_item_id == extraction_item_id,
            EntityResolution.role == role,
            EntityResolution.is_active.is_(True),
        )
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def list_for_item(
        self, extraction_item_id: uuid.UUID, *, limit: int
    ) -> list[EntityResolution]:
        """Full history for one item, newest first — superseded rows included."""
        stmt = (
            select(EntityResolution)
            .where(EntityResolution.extraction_item_id == extraction_item_id)
            .order_by(EntityResolution.created_at.desc(), EntityResolution.id.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_profile_rows(self, entity_id: uuid.UUID, *, limit: int) -> Sequence[ProfileRow]:
        """The whole grounded chain behind one entity, in one query.

        resolution -> extraction item -> extraction run -> evidence item.

        Active resolutions only: a superseded one records what the system used
        to believe, and a profile states what it believes now. The full history
        stays available per item through `list_for_item`.

        Ordered by evidence creation then item position, so a profile reads in
        the order the material arrived rather than in whatever order the planner
        returned rows.
        """
        stmt = (
            select(EntityResolution, ExtractionItem, ExtractionRun, EvidenceItem)
            .join(ExtractionItem, ExtractionItem.id == EntityResolution.extraction_item_id)
            .join(ExtractionRun, ExtractionRun.id == ExtractionItem.extraction_run_id)
            .join(EvidenceItem, EvidenceItem.id == ExtractionRun.evidence_id)
            .where(
                EntityResolution.entity_id == entity_id,
                EntityResolution.is_active.is_(True),
            )
            .order_by(
                EvidenceItem.created_at.asc(),
                EvidenceItem.id.asc(),
                ExtractionItem.position.asc(),
            )
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.all()
