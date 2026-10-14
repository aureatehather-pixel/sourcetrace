"""Cross-evidence intelligence HTTP routes.

Three operations, all of them things a person actually does, none of them CRUD
over a table that happens to exist:

- **analyse** an entity's grounded claims for relations between them;
- **read** the resulting intelligence, with the grounded material underneath it;
- **reject** a relation that is wrong.

There is no PUT and no DELETE. A derived relation is a record of what the system
asserted; rejecting it changes whether it is served, never what it said.

Status codes follow the M2/M3 convention that the code and the body both carry
the outcome: 201 when the analysis spent a model call, 200 when every candidate
was already settled, with `performed_calls` and full telemetry in the body for
clients that ignore status codes.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Response, status

from intelligence_engine.api.dependencies import IntelligenceServiceDep
from intelligence_engine.intelligence.schemas import (
    AnalysisResponse,
    EntityIntelligence,
    IntelligenceRelationRead,
    RelationReject,
)

router = APIRouter(prefix="/entities/{entity_id}", tags=["intelligence"])
relation_router = APIRouter(prefix="/intelligence-relations/{relation_id}", tags=["intelligence"])


@router.post(
    "/analyze-relations",
    response_model=AnalysisResponse,
    summary="Derive relations among one entity's grounded claims",
    responses={
        200: {"description": "Every candidate pair was already settled; no model call was made"},
        201: {"description": "New relations were derived"},
        404: {"description": "Entity not found"},
        502: {"description": "Classifier failed or returned unusable output"},
    },
)
async def analyze_relations(
    entity_id: uuid.UUID, service: IntelligenceServiceDep, response: Response
) -> AnalysisResponse:
    """201 when a model call happened, 200 when everything was reused.

    `telemetry` reports what narrowing achieved and what the analysis cost:
    theoretical pairs, candidates after narrowing, pairs reused, pairs actually
    classified, batches, external calls and tokens.
    """
    result = await service.analyze(entity_id)
    response.status_code = status.HTTP_201_CREATED if result.performed_calls else status.HTTP_200_OK
    return result


@router.get(
    "/intelligence",
    response_model=EntityIntelligence,
    summary="Grounded claims and the relations derived from them",
    responses={404: {"description": "Entity not found"}},
)
async def get_intelligence(
    entity_id: uuid.UUID, service: IntelligenceServiceDep
) -> EntityIntelligence:
    """A read model, assembled per request. Nothing here is stored.

    Derived relations are kept structurally separate from the grounded claims
    they were derived from, and each one carries both source items with their
    verbatim quotes, evidence ids, run ids and resolution ids.
    """
    return await service.get_intelligence(entity_id)


@relation_router.post(
    "/reject",
    response_model=IntelligenceRelationRead,
    summary="Mark a derived relation wrong",
    responses={
        200: {"description": "The relation is no longer served"},
        404: {"description": "Relation not found"},
        409: {"description": "The relation was already rejected"},
    },
)
async def reject_relation(
    relation_id: uuid.UUID, payload: RelationReject, service: IntelligenceServiceDep
) -> IntelligenceRelationRead:
    """Withdraw a relation without deleting it.

    It stops appearing as intelligence, stays in the table as a record of what
    was asserted and when, and keeps satisfying the reuse lookup — so the next
    analysis does not pay to regenerate the same wrong answer. Bumping the
    classifier version is what re-opens the question.
    """
    return await service.reject_relation(relation_id, payload)
