"""Entity application behaviour: identity, resolution, correction, profile.

Four operations and the rules that keep them honest:

- **create** — entities exist because a human said so, never as a side effect of
  ingestion or extraction. That is the defence against entity pollution.
- **resolve** — either a human assertion (`manual`) or the conservative
  deterministic matcher (`exact_match`), which resolves on exactly one candidate
  and otherwise declines with a reason.
- **supersede** — the correction path. Append-only: the old assertion is retired
  and stamped, never edited away, and neither evidence nor extraction items are
  touched.
- **profile** — a read model assembled by query from evidence, extraction items
  and *active* resolutions. Nothing about a profile is stored.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from intelligence_engine.core.errors import PersistenceError
from intelligence_engine.core.pagination import decode_cursor, encode_cursor
from intelligence_engine.entities.errors import (
    EntityNotFound,
    ExtractionItemNotFound,
    IncompatibleResolutionRole,
    ProfileAssemblyError,
    ResolutionAlreadyExists,
    ResolutionAlreadySuperseded,
    ResolutionNotFound,
)
from intelligence_engine.entities.matching import (
    ROLE_SOURCE_KINDS,
    decide,
    matchable_name,
    normalize_name,
)
from intelligence_engine.entities.models import Entity, EntityResolution
from intelligence_engine.entities.repository import (
    EntityRepository,
    ResolutionRepository,
)
from intelligence_engine.entities.schemas import (
    EntityCreate,
    EntityPage,
    EntityProfile,
    EntityRead,
    EntityResolutionRead,
    EntityType,
    ProfileAttributedClaim,
    ProfileMention,
    ProfileSource,
    ProfileSubjectObservation,
    ResolutionCreate,
    ResolutionHistory,
    ResolutionMethod,
    ResolutionOutcome,
    ResolutionRole,
    ResolutionSupersede,
    UnresolvedReason,
)
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.extraction.models import ExtractionItem
from intelligence_engine.extraction.repository import ExtractionRepository
from intelligence_engine.extraction.schemas import (
    EntityMentionData,
    ItemKind,
    ObservationData,
    ReportedClaimData,
)

logger = logging.getLogger(__name__)

MAX_RESOLUTIONS_LISTED = 200
MAX_PROFILE_ROWS = 500


class EntityService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._entities = EntityRepository(session)
        self._resolutions = ResolutionRepository(session)
        self._extraction = ExtractionRepository(session)

    # -- entities ----------------------------------------------------------

    async def create_entity(self, payload: EntityCreate) -> EntityRead:
        entity = self._entities.add(
            entity_type=payload.entity_type,
            canonical_name=payload.canonical_name,
            disambiguator=payload.disambiguator,
        )
        await self._commit("entity.create.failed")
        await self._session.refresh(entity)

        logger.info(
            "entity.created",
            extra={
                # Identifiers and types only. `canonical_name` is drawn from
                # evidence and can name a private individual, so it stays out
                # of the log for the same reason evidence bodies do.
                "entity_id": str(entity.id),
                "entity_type": entity.entity_type.value,
            },
        )
        return EntityRead.model_validate(entity)

    async def get_entity(self, entity_id: uuid.UUID) -> EntityRead:
        return EntityRead.model_validate(await self._require_entity(entity_id))

    async def list_entities(
        self,
        *,
        limit: int,
        cursor: str | None = None,
        entity_type: EntityType | None = None,
    ) -> EntityPage:
        after = decode_cursor(cursor) if cursor else None
        rows = await self._entities.list_page(limit=limit + 1, after=after, entity_type=entity_type)
        has_more = len(rows) > limit
        page_rows = rows[:limit]
        next_cursor = (
            encode_cursor(page_rows[-1].created_at, page_rows[-1].id)
            if has_more and page_rows
            else None
        )
        return EntityPage(
            items=[EntityRead.model_validate(row) for row in page_rows],
            next_cursor=next_cursor,
        )

    # -- resolution --------------------------------------------------------

    async def resolve(
        self, extraction_item_id: uuid.UUID, payload: ResolutionCreate
    ) -> ResolutionOutcome:
        """Connect one grounded extraction item to one entity, or decline."""
        item = await self._require_item(extraction_item_id)
        self._require_compatible_role(item, payload.role)

        existing = await self._resolutions.get_active_for_item(
            extraction_item_id=extraction_item_id, role=payload.role
        )
        if existing is not None:
            # Not overwritten. Replacing an assertion in place would destroy the
            # record of what the system previously believed; that is what
            # supersession is for.
            raise ResolutionAlreadyExists(
                f"Extraction item {extraction_item_id} already has an active "
                f"'{payload.role.value}' resolution; supersede it to change it"
            )

        if payload.entity_id is not None:
            entity = await self._require_entity(payload.entity_id)
            return await self._assert_resolution(
                item=item,
                entity=entity,
                role=payload.role,
                method=ResolutionMethod.MANUAL,
                note=payload.note,
                candidate_ids=[],
            )

        return await self._attempt_exact_match(item=item, role=payload.role, note=payload.note)

    async def supersede(
        self, resolution_id: uuid.UUID, payload: ResolutionSupersede
    ) -> ResolutionOutcome:
        """Correct a resolution without rewriting history.

        The old row stays exactly as it was written, plus `is_active = false`
        and a `superseded_at` stamp. Evidence and extraction items are not
        touched at all — a wrong entity decision never propagates backwards into
        the material it was decided from.
        """
        resolution = await self._resolutions.get(resolution_id)
        if resolution is None:
            raise ResolutionNotFound(resolution_id)
        if not resolution.is_active:
            raise ResolutionAlreadySuperseded(
                f"Entity resolution {resolution_id} was already superseded at "
                f"{resolution.superseded_at.isoformat() if resolution.superseded_at else 'unknown'}"
            )

        now = dt.datetime.now(dt.UTC)
        replacement: EntityResolution | None = None
        entity = (
            await self._require_entity(payload.entity_id) if payload.entity_id is not None else None
        )

        # Three flushes, in this order, because two immediate constraints pull
        # in opposite directions and neither is deferrable:
        #
        #   - the partial unique index allows one *active* resolution per
        #     (item, role), so the old row must go inactive before the new row
        #     is inserted;
        #   - `superseded_by_id` is a self-foreign-key, so the new row must
        #     exist before the old row can point at it.
        #
        # Retiring first with a NULL pointer satisfies both, and the
        # superseded-state CHECK permits an inactive row that names no
        # successor - which is exactly what a detachment is.
        self._resolutions.deactivate(
            resolution,
            at=now,
            superseded_by_id=None,
            # A detachment's note explains the retirement itself; a
            # replacement's note belongs on the new assertion instead.
            note=payload.note if entity is None else None,
        )
        await self._session.flush()

        if entity is not None:
            replacement = self._resolutions.add(
                extraction_item_id=resolution.extraction_item_id,
                source_item_kind=resolution.source_item_kind,
                entity_id=entity.id,
                role=resolution.role,
                method=ResolutionMethod.MANUAL,
                note=payload.note,
            )
            await self._session.flush()
            resolution.superseded_by_id = replacement.id
            await self._session.flush()

        # All of it in one transaction: a correction never half-lands, leaving
        # a mention detached from an entity it was meant to be moved to.
        await self._commit("resolution.supersede.failed")
        await self._session.refresh(resolution)
        if replacement is not None:
            await self._session.refresh(replacement)

        logger.info(
            "entity.resolution.superseded",
            extra={
                "resolution_id": str(resolution.id),
                "extraction_item_id": str(resolution.extraction_item_id),
                "entity_id": str(resolution.entity_id),
                "resolution_role": resolution.role.value,
                "resolution_action": "replaced" if replacement else "detached",
                "replacement_resolution_id": str(replacement.id) if replacement else None,
                "replacement_entity_id": str(replacement.entity_id) if replacement else None,
            },
        )

        superseded = EntityResolutionRead.model_validate(resolution)
        if replacement is None:
            return ResolutionOutcome(
                status="unresolved",
                reason=UnresolvedReason.DETACHED_BY_CORRECTION,
                superseded=superseded,
            )
        return ResolutionOutcome(
            status="resolved",
            resolution=EntityResolutionRead.model_validate(replacement),
            superseded=superseded,
        )

    async def list_resolutions(self, extraction_item_id: uuid.UUID) -> ResolutionHistory:
        """Every assertion ever made about one item, current and retired."""
        await self._require_item(extraction_item_id)
        rows = await self._resolutions.list_for_item(
            extraction_item_id, limit=MAX_RESOLUTIONS_LISTED
        )
        return ResolutionHistory(items=[EntityResolutionRead.model_validate(row) for row in rows])

    # -- profile -----------------------------------------------------------

    async def get_profile(self, entity_id: uuid.UUID) -> EntityProfile:
        """Assemble the profile from grounded state, at request time.

        Every line is produced from a row that exists because some resolution
        says it belongs to this entity. Nothing is summarised, inferred, or
        pulled in because it happened to share an evidence item.
        """
        entity = await self._require_entity(entity_id)
        rows = await self._resolutions.list_profile_rows(entity_id, limit=MAX_PROFILE_ROWS)

        sources: dict[uuid.UUID, ProfileSource] = {}
        mentions: list[ProfileMention] = []
        claims: list[ProfileAttributedClaim] = []
        observations: list[ProfileSubjectObservation] = []

        for resolution, item, _run, evidence in rows:
            if evidence.id not in sources:
                sources[evidence.id] = ProfileSource(
                    evidence_id=evidence.id,
                    source_type=evidence.source_type,
                    source_label=evidence.source_label,
                    source_url=evidence.source_url,
                    captured_at=evidence.captured_at,
                    created_at=evidence.created_at,
                )
            common = _provenance(resolution, item, evidence)
            if resolution.role is ResolutionRole.ABOUT:
                observed = _observation_data(item)
                observations.append(
                    ProfileSubjectObservation(
                        **common,
                        observation=observed.observation,
                        subject=observed.subject,
                    )
                )
            elif resolution.role is ResolutionRole.MENTION:
                mention = _mention_data(item)
                mentions.append(
                    ProfileMention(
                        **common,
                        name_as_mentioned=mention.name,
                        mentioned_as_type=EntityType(mention.entity_type),
                        described_as=mention.described_as,
                    )
                )
            else:
                claim = _claim_data(item)
                claims.append(
                    ProfileAttributedClaim(
                        **common,
                        claim=claim.claim,
                        claimed_by=claim.claimed_by,
                        stance=claim.stance,
                    )
                )

        return EntityProfile(
            entity=EntityRead.model_validate(entity),
            evidence_count=len(sources),
            sources=list(sources.values()),
            mentions=mentions,
            attributed_claims=claims,
            subject_observations=observations,
        )

    # -- internals ---------------------------------------------------------

    async def _attempt_exact_match(
        self, *, item: ExtractionItem, role: ResolutionRole, note: str | None
    ) -> ResolutionOutcome:
        """The conservative deterministic baseline. High precision by design."""
        raw_name = matchable_name(item.kind, item.payload)
        match_key = normalize_name(raw_name) if raw_name else ""

        candidates: Sequence[Entity] = []
        if match_key:
            candidates = await self._entities.find_candidates(
                normalized_name=match_key,
                # A mention carries its own type, so matching is type-aware. An
                # attribution string does not, so its search cannot be narrowed
                # and is correspondingly more likely to come back ambiguous.
                entity_type=(
                    EntityType(EntityMentionData.model_validate(item.payload).entity_type)
                    if item.kind is ItemKind.ENTITY_MENTION
                    else None
                ),
            )

        candidate_ids = [candidate.id for candidate in candidates]
        decision = decide(match_key, candidate_ids)

        if decision.entity_id is None:
            logger.info(
                "entity.resolution.declined",
                extra={
                    "extraction_item_id": str(item.id),
                    "resolution_role": role.value,
                    "resolution_method": ResolutionMethod.EXACT_MATCH.value,
                    "resolution_action": "unresolved",
                    "reason": decision.reason.value if decision.reason else None,
                    "candidate_count": len(candidate_ids),
                },
            )
            return ResolutionOutcome(
                status="unresolved",
                reason=decision.reason,
                candidate_entity_ids=candidate_ids,
            )

        entity = candidates[0]
        return await self._assert_resolution(
            item=item,
            entity=entity,
            role=role,
            method=ResolutionMethod.EXACT_MATCH,
            note=note,
            candidate_ids=candidate_ids,
        )

    async def _assert_resolution(
        self,
        *,
        item: ExtractionItem,
        entity: Entity,
        role: ResolutionRole,
        method: ResolutionMethod,
        note: str | None,
        candidate_ids: list[uuid.UUID],
    ) -> ResolutionOutcome:
        resolution = self._resolutions.add(
            extraction_item_id=item.id,
            source_item_kind=item.kind,
            entity_id=entity.id,
            role=role,
            method=method,
            note=note,
        )
        await self._commit("resolution.create.failed")
        await self._session.refresh(resolution)

        logger.info(
            "entity.resolution.created",
            extra={
                "resolution_id": str(resolution.id),
                "entity_id": str(entity.id),
                "entity_type": entity.entity_type.value,
                "extraction_item_id": str(item.id),
                "resolution_role": role.value,
                "resolution_method": method.value,
                "resolution_action": "resolved",
            },
        )
        return ResolutionOutcome(
            status="resolved",
            candidate_entity_ids=candidate_ids,
            resolution=EntityResolutionRead.model_validate(resolution),
        )

    async def _require_entity(self, entity_id: uuid.UUID) -> Entity:
        entity = await self._entities.get(entity_id)
        if entity is None:
            raise EntityNotFound(entity_id)
        return entity

    async def _require_item(self, extraction_item_id: uuid.UUID) -> ExtractionItem:
        item = await self._extraction.get_item(extraction_item_id)
        if item is None:
            raise ExtractionItemNotFound(extraction_item_id)
        return item

    @staticmethod
    def _require_compatible_role(item: ExtractionItem, role: ResolutionRole) -> None:
        expected = ROLE_SOURCE_KINDS[role]
        if item.kind is not expected:
            raise IncompatibleResolutionRole(
                f"Role '{role.value}' applies to '{expected.value}' items, "
                f"but this item is a '{item.kind.value}'"
            )

    async def _commit(self, event: str) -> None:
        try:
            await self._session.commit()
        except SQLAlchemyError as exc:
            await self._session.rollback()
            logger.exception(event)
            raise PersistenceError("Failed to persist entity state") from exc


def _provenance(
    resolution: EntityResolution,
    item: ExtractionItem,
    evidence: EvidenceItem,
) -> dict[str, Any]:
    """The provenance fields every profile line carries, without exception.

    Four identifiers and the verbatim quote. Given these, a reader can walk any
    profile line back to the evidence it came from and the decision that put it
    here, without trusting the profile itself.
    """
    return {
        "resolution_id": resolution.id,
        "extraction_item_id": item.id,
        "extraction_run_id": item.extraction_run_id,
        "evidence_id": evidence.id,
        "grounding_quote": item.grounding_quote,
        "method": resolution.method,
        "resolved_at": resolution.created_at,
    }


def _mention_data(item: ExtractionItem) -> EntityMentionData:
    try:
        return EntityMentionData.model_validate(item.payload)
    except ValidationError as exc:
        raise ProfileAssemblyError(
            f"Extraction item {item.id} payload is not a readable entity_mention"
        ) from exc


def _claim_data(item: ExtractionItem) -> ReportedClaimData:
    try:
        return ReportedClaimData.model_validate(item.payload)
    except ValidationError as exc:
        raise ProfileAssemblyError(
            f"Extraction item {item.id} payload is not a readable reported_claim"
        ) from exc


def _observation_data(item: ExtractionItem) -> ObservationData:
    try:
        return ObservationData.model_validate(item.payload)
    except ValidationError as exc:
        raise ProfileAssemblyError(
            f"Extraction item {item.id} payload is not a readable observation"
        ) from exc
