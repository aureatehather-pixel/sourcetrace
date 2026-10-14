"""Extraction HTTP routes.

Two endpoints. No administration API, no update or delete: extraction runs are
history, and history is not edited.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Response, status

from intelligence_engine.api.dependencies import ExtractionServiceDep
from intelligence_engine.extraction.schemas import ExtractionResponse, ExtractionRunList

router = APIRouter(prefix="/evidence/{evidence_id}", tags=["extraction"])


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="Extract structured information from one evidence item",
    responses={
        200: {"description": "A matching successful extraction was reused"},
        201: {"description": "A new extraction was performed"},
        404: {"description": "Evidence item not found"},
        502: {"description": "Provider failed or returned unusable output"},
    },
)
async def extract_evidence(
    evidence_id: uuid.UUID, service: ExtractionServiceDep, response: Response
) -> ExtractionResponse:
    """201 when a model call happened, 200 when a stored result was reused.

    `reused` is also in the body, so the distinction survives clients that
    ignore status codes.
    """
    result = await service.extract(evidence_id)
    response.status_code = status.HTTP_200_OK if result.reused else status.HTTP_201_CREATED
    return result


@router.get(
    "/extractions",
    response_model=ExtractionRunList,
    summary="List extraction attempts for one evidence item",
    responses={404: {"description": "Evidence item not found"}},
)
async def list_extractions(
    evidence_id: uuid.UUID, service: ExtractionServiceDep
) -> ExtractionRunList:
    """Newest first, including failed attempts — the operational history."""
    return await service.list_runs(evidence_id)
