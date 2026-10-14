"""Request-scoped dependencies.

The engine and session factory live on ``app.state`` (created at startup,
disposed at shutdown). Nothing here is a module-level global, so tests can run
a fully independent application against a disposable database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from intelligence_engine.core.config import Settings
from intelligence_engine.db.session import session_scope
from intelligence_engine.entities.service import EntityService
from intelligence_engine.evidence.service import EvidenceService
from intelligence_engine.extraction.provider import ExtractionProvider
from intelligence_engine.extraction.service import ExtractionService
from intelligence_engine.intelligence.service import IntelligenceService


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_engine(request: Request) -> AsyncEngine:
    engine: AsyncEngine = request.app.state.engine
    return engine


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async for session in session_scope(factory):
        yield session


def get_evidence_service(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> EvidenceService:
    return EvidenceService(session)


def get_entity_service(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> EntityService:
    return EntityService(session)


def get_extraction_provider(request: Request) -> ExtractionProvider:
    """The provider built once at startup and held on app.state.

    One instance per process: the Anthropic client owns a connection pool, and
    rebuilding it per request would discard it.
    """
    provider: ExtractionProvider = request.app.state.extraction_provider
    return provider


def get_extraction_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    provider: Annotated[ExtractionProvider, Depends(get_extraction_provider)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> ExtractionService:
    return ExtractionService(
        session, provider, window_max_chars=settings.extraction_window_max_chars
    )


def get_relation_provider(request: Request) -> ExtractionProvider:
    """The relation classifier's provider, built once at startup.

    Separate from the extraction provider because it is configured with a
    different (cheaper) model, and because each owns its own connection pool.
    """
    provider: ExtractionProvider = request.app.state.relation_provider
    return provider


def get_intelligence_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    provider: Annotated[ExtractionProvider, Depends(get_relation_provider)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> IntelligenceService:
    return IntelligenceService(
        session,
        provider,
        max_candidate_pairs=settings.relation_max_candidate_pairs,
        batch_size=settings.relation_batch_size,
    )


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
EngineDep = Annotated[AsyncEngine, Depends(get_engine)]
EvidenceServiceDep = Annotated[EvidenceService, Depends(get_evidence_service)]
ExtractionServiceDep = Annotated[ExtractionService, Depends(get_extraction_service)]
EntityServiceDep = Annotated[EntityService, Depends(get_entity_service)]
IntelligenceServiceDep = Annotated[IntelligenceService, Depends(get_intelligence_service)]
