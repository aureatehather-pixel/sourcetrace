"""Entity and entity-resolution HTTP routes.

Three routers, one per resource the operations actually act on. The endpoints
are product operations rather than table CRUD:

- entities are created and read, never updated or deleted through the API. An
  entity with resolution history is referenced by an audit trail, and DELETE
  would either break it or silently take it with it.
- resolutions are asserted and superseded. There is no PUT and no DELETE:
  correcting a resolution is a new record, never an edit of the old one.

Status codes follow the M2 convention that a code and a body both carry the
outcome: 201 when a resolution row was created, 200 when the request was
handled but nothing was asserted, with `status` and `reason` in the body for
clients that ignore status codes.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from intelligence_engine.api.dependencies import EntityServiceDep, SettingsDep
from intelligence_engine.entities.schemas import (
    EntityCreate,
    EntityPage,
    EntityProfile,
    EntityRead,
    EntityType,
    ResolutionCreate,
    ResolutionHistory,
    ResolutionOutcome,
    ResolutionSupersede,
)

router = APIRouter(prefix="/entities", tags=["entities"])
resolution_router = APIRouter(
    prefix="/extraction-items/{extraction_item_id}", tags=["entity resolution"]
)
correction_router = APIRouter(
    prefix="/entity-resolutions/{resolution_id}", tags=["entity resolution"]
)


@router.post(
    "",
    response_model=EntityRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a persistent entity",
)
async def create_entity(payload: EntityCreate, service: EntityServiceDep) -> EntityRead:
    """Assert that this identity exists.

    Nothing else creates entities. Ingesting evidence does not, and neither does
    extraction: a name appearing in text is a mention, not a decision that a
    real-world entity is worth tracking.
    """
    return await service.create_entity(payload)


@router.get("", response_model=EntityPage, summary="List entities, newest first")
async def list_entities(
    service: EntityServiceDep,
    settings: SettingsDep,
    entity_type: Annotated[EntityType | None, Query()] = None,
    limit: Annotated[int | None, Query(ge=1, le=500)] = None,
    cursor: Annotated[str | None, Query(description="Opaque cursor from a previous page")] = None,
) -> EntityPage:
    effective_limit = min(limit or settings.default_page_size, settings.max_page_size)
    return await service.list_entities(
        limit=effective_limit, cursor=cursor, entity_type=entity_type
    )


@router.get(
    "/{entity_id}",
    response_model=EntityRead,
    summary="Retrieve one entity",
    responses={404: {"description": "Entity not found"}},
)
async def get_entity(entity_id: uuid.UUID, service: EntityServiceDep) -> EntityRead:
    return await service.get_entity(entity_id)


@router.get(
    "/{entity_id}/profile",
    response_model=EntityProfile,
    summary="The grounded profile assembled for one entity",
    responses={404: {"description": "Entity not found"}},
)
async def get_entity_profile(entity_id: uuid.UUID, service: EntityServiceDep) -> EntityProfile:
    """Everything the system currently believes about this entity, and why.

    Computed from evidence, extraction items and active resolutions on every
    request. There is no stored profile to go stale, and every line carries the
    evidence id, run id and verbatim quote it came from.
    """
    return await service.get_profile(entity_id)


@resolution_router.post(
    "/resolution",
    response_model=ResolutionOutcome,
    summary="Resolve one extraction item to an entity",
    responses={
        200: {"description": "Handled, but nothing was asserted - see `reason`"},
        201: {"description": "A resolution was asserted"},
        404: {"description": "Extraction item or entity not found"},
        409: {"description": "This item already has an active resolution in this role"},
        422: {"description": "The role does not apply to this item kind"},
    },
)
async def resolve_extraction_item(
    extraction_item_id: uuid.UUID,
    payload: ResolutionCreate,
    service: EntityServiceDep,
    response: Response,
) -> ResolutionOutcome:
    """201 when an assertion was recorded, 200 when the system declined to make one.

    Declining is a normal outcome, not an error: `no_candidate` and
    `ambiguous_candidates` are the matcher correctly refusing to guess.
    """
    outcome = await service.resolve(extraction_item_id, payload)
    response.status_code = (
        status.HTTP_201_CREATED if outcome.status == "resolved" else status.HTTP_200_OK
    )
    return outcome


@resolution_router.get(
    "/resolutions",
    response_model=ResolutionHistory,
    summary="Resolution history for one extraction item",
    responses={404: {"description": "Extraction item not found"}},
)
async def list_item_resolutions(
    extraction_item_id: uuid.UUID, service: EntityServiceDep
) -> ResolutionHistory:
    """Newest first, superseded rows included — the audit trail for one mention."""
    return await service.list_resolutions(extraction_item_id)


@correction_router.post(
    "/supersede",
    response_model=ResolutionOutcome,
    summary="Correct a resolution",
    responses={
        200: {"description": "The resolution was superseded"},
        404: {"description": "Resolution or replacement entity not found"},
        409: {"description": "This resolution was already superseded"},
    },
)
async def supersede_resolution(
    resolution_id: uuid.UUID,
    payload: ResolutionSupersede,
    service: EntityServiceDep,
) -> ResolutionOutcome:
    """Say "that is the wrong entity", with or without naming the right one.

    Supply `entity_id` to re-point the item; omit it to detach and return the
    item to unresolved. Either way the original assertion survives, stamped with
    when it stopped being current, and the underlying evidence and extraction
    item are untouched.
    """
    return await service.supersede(resolution_id, payload)
