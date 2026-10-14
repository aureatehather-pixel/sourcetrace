"""Liveness and readiness.

Two endpoints, no framework. /health answers "is this process up"; /ready
answers "can it actually reach PostgreSQL", which is the only dependency M1 has.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from intelligence_engine.api.dependencies import EngineDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# A readiness probe must answer within a bounded time. Hanging is a failure,
# not a pending success.
READINESS_TIMEOUT_SECONDS = 3.0


class HealthStatus(BaseModel):
    status: str


class ReadyStatus(BaseModel):
    status: str
    database: str


@router.get("/health", response_model=HealthStatus, summary="Process liveness")
async def health() -> HealthStatus:
    return HealthStatus(status="ok")


@router.get("/ready", response_model=ReadyStatus, summary="Dependency readiness")
async def ready(engine: EngineDep, response: Response) -> ReadyStatus:
    try:
        await asyncio.wait_for(_ping(engine), timeout=READINESS_TIMEOUT_SECONDS)
    except (SQLAlchemyError, TimeoutError, OSError):
        logger.exception("readiness.database_unavailable")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyStatus(status="unavailable", database="unavailable")
    return ReadyStatus(status="ok", database="ok")


async def _ping(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
