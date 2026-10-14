"""Evidence HTTP routes.

There is deliberately no PUT or PATCH. Raw evidence is immutable in M1;
corrections will arrive later as revision/supersession records rather than by
rewriting history.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status

from intelligence_engine.api.dependencies import EvidenceServiceDep, SettingsDep
from intelligence_engine.evidence.schemas import EvidenceCreate, EvidencePage, EvidenceRead

router = APIRouter(prefix="/evidence", tags=["evidence"])


@router.post(
    "",
    response_model=EvidenceRead,
    status_code=status.HTTP_201_CREATED,
    summary="Submit raw evidence",
)
async def create_evidence(payload: EvidenceCreate, service: EvidenceServiceDep) -> EvidenceRead:
    return await service.create(payload)


@router.get(
    "",
    response_model=EvidencePage,
    summary="List evidence, newest first",
)
async def list_evidence(
    service: EvidenceServiceDep,
    settings: SettingsDep,
    limit: Annotated[int | None, Query(ge=1, le=500)] = None,
    cursor: Annotated[str | None, Query(description="Opaque cursor from a previous page")] = None,
) -> EvidencePage:
    effective_limit = min(limit or settings.default_page_size, settings.max_page_size)
    return await service.list(limit=effective_limit, cursor=cursor)


@router.get(
    "/{evidence_id}",
    response_model=EvidenceRead,
    summary="Retrieve one evidence item",
    responses={404: {"description": "Evidence item not found"}},
)
async def get_evidence(evidence_id: uuid.UUID, service: EvidenceServiceDep) -> EvidenceRead:
    return await service.get(evidence_id)
