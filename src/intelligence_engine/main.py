"""Application assembly: lifespan, middleware, error translation, routes."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from intelligence_engine import health
from intelligence_engine.api.router import api_router
from intelligence_engine.core.config import Settings, get_settings
from intelligence_engine.core.errors import EvidenceNotFound, InvalidCursor, PersistenceError
from intelligence_engine.core.logging import configure_logging, request_id_var
from intelligence_engine.db.session import create_engine, create_session_factory
from intelligence_engine.entities.errors import (
    EntityError,
    EntityNotFound,
    ExtractionItemNotFound,
    IncompatibleResolutionRole,
    ProfileAssemblyError,
    ResolutionAlreadyExists,
    ResolutionAlreadySuperseded,
    ResolutionNotFound,
)
from intelligence_engine.extraction.errors import ExtractionError, ProviderError
from intelligence_engine.extraction.factory import build_provider
from intelligence_engine.intelligence.errors import (
    IntelligenceAssemblyError,
    IntelligenceError,
    InvalidRelationOutput,
    RelationAlreadyRejected,
    RelationNotFound,
)
from intelligence_engine.intelligence.factory import build_relation_provider

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


def _error(
    status_code: int,
    message: str,
    code: str,
    details: list[dict[str, str]] | None = None,
) -> JSONResponse:
    """Uniform error envelope. Never carries SQL, stack traces or internals."""
    error: dict[str, object] = {"code": code, "message": message}
    request_id = request_id_var.get()
    if request_id is not None:
        error["request_id"] = request_id
    if details:
        error["details"] = details
    return JSONResponse(status_code=status_code, content={"error": error})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    engine = create_engine(settings.database_url_str)
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    # One provider per process each - they own HTTP connection pools, and they
    # are configured with different models.
    app.state.extraction_provider = build_provider(settings)
    app.state.relation_provider = build_relation_provider(settings)
    logger.info(
        "application.startup",
        extra={
            "environment": settings.environment,
            "extraction_provider_configured": settings.has_extraction_provider,
            "extraction_model": settings.extraction_model,
            "relation_model": settings.relation_model,
        },
    )
    try:
        yield
    finally:
        await engine.dispose()
        logger.info("application.shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an application instance.

    Tests pass their own settings so a disposable database can be targeted
    without mutating process-wide state.
    """
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    app = FastAPI(
        title="Personal Intelligence Engine",
        version="0.1.0",
        summary="Durable ingestion and retrieval of raw evidence.",
        lifespan=lifespan,
    )
    app.state.settings = resolved

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[object]]
    ) -> object:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            request_id_var.reset(token)

        status_code = getattr(response, "status_code", 0)
        logger.info(
            "http.request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers[REQUEST_ID_HEADER] = request_id  # type: ignore[attr-defined]
        return response

    @app.exception_handler(EvidenceNotFound)
    async def _not_found(_: Request, exc: EvidenceNotFound) -> JSONResponse:
        return _error(status.HTTP_404_NOT_FOUND, str(exc), "evidence_not_found")

    @app.exception_handler(InvalidCursor)
    async def _bad_cursor(_: Request, exc: InvalidCursor) -> JSONResponse:
        return _error(status.HTTP_400_BAD_REQUEST, str(exc), "invalid_cursor")

    @app.exception_handler(EntityNotFound)
    @app.exception_handler(ExtractionItemNotFound)
    @app.exception_handler(ResolutionNotFound)
    async def _entity_not_found(_: Request, exc: EntityError) -> JSONResponse:
        return _error(status.HTTP_404_NOT_FOUND, str(exc), exc.error_code)

    @app.exception_handler(ResolutionAlreadyExists)
    @app.exception_handler(ResolutionAlreadySuperseded)
    async def _resolution_conflict(_: Request, exc: EntityError) -> JSONResponse:
        # 409: the request was well formed and the state says no. Overwriting an
        # existing assertion, or forking an already-corrected one, would destroy
        # the audit trail M3 exists to keep.
        return _error(status.HTTP_409_CONFLICT, str(exc), exc.error_code)

    @app.exception_handler(IncompatibleResolutionRole)
    async def _incompatible_role(_: Request, exc: EntityError) -> JSONResponse:
        return _error(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc), exc.error_code)

    @app.exception_handler(RelationNotFound)
    async def _relation_not_found(_: Request, exc: IntelligenceError) -> JSONResponse:
        return _error(status.HTTP_404_NOT_FOUND, str(exc), exc.error_code)

    @app.exception_handler(RelationAlreadyRejected)
    async def _relation_conflict(_: Request, exc: IntelligenceError) -> JSONResponse:
        # 409: the request was well formed and the state says no. Rejecting twice
        # would overwrite the first rejection's note and timestamp.
        return _error(status.HTTP_409_CONFLICT, str(exc), exc.error_code)

    @app.exception_handler(InvalidRelationOutput)
    async def _invalid_relation_output(_: Request, exc: IntelligenceError) -> JSONResponse:
        # 502, as M2 does for unusable extraction output: the upstream returned
        # something we refuse to store. The failed analysis run is already recorded.
        logger.warning("intelligence.rejected", extra={"error_code": exc.error_code})
        return _error(
            status.HTTP_502_BAD_GATEWAY,
            "The relation classifier returned output that failed validation",
            exc.error_code,
        )

    @app.exception_handler(IntelligenceAssemblyError)
    async def _intelligence_assembly_failed(
        _: Request, exc: IntelligenceAssemblyError
    ) -> JSONResponse:
        # Our own stored payload could not be read back. Fails loudly rather than
        # quietly dropping grounded material out of a derived assertion.
        logger.error("intelligence.assembly_failed", exc_info=exc)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error", "internal_error"
        )

    @app.exception_handler(ProfileAssemblyError)
    async def _profile_failed(_: Request, exc: ProfileAssemblyError) -> JSONResponse:
        # Our own stored payload could not be read back. That is an internal
        # fault, and it fails loudly rather than quietly dropping a profile line.
        logger.error("entity.profile.assembly_failed", exc_info=exc)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error", "internal_error"
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's per-field detail is useful to the caller and leaks nothing
        # internal, but the raw input echo is dropped.
        details = [
            {"field": ".".join(str(part) for part in err["loc"][1:]), "message": err["msg"]}
            for err in exc.errors()
        ]
        return _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Request validation failed",
            "invalid_request",
            details=details,
        )

    @app.exception_handler(ProviderError)
    async def _provider(_: Request, exc: ProviderError) -> JSONResponse:
        # 502: the upstream failed, not us. The code is bounded and stable;
        # the message is ours, never the provider's exception text.
        logger.warning("extraction.provider_error", extra={"error_code": exc.error_code})
        return _error(
            status.HTTP_502_BAD_GATEWAY,
            "The extraction provider could not complete the request",
            exc.error_code,
        )

    @app.exception_handler(ExtractionError)
    async def _extraction(_: Request, exc: ExtractionError) -> JSONResponse:
        # Invalid or ungrounded model output. Also 502 - the upstream returned
        # something unusable. The failed run is already recorded.
        logger.warning("extraction.rejected", extra={"error_code": exc.error_code})
        return _error(
            status.HTTP_502_BAD_GATEWAY,
            "The extraction provider returned output that failed validation",
            exc.error_code,
        )

    @app.exception_handler(PersistenceError)
    async def _persistence(_: Request, exc: PersistenceError) -> JSONResponse:
        logger.error("persistence.error", exc_info=exc)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal server error",
            "internal_error",
        )

    @app.exception_handler(SQLAlchemyError)
    async def _database(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        # Full detail goes to the log; the client gets nothing actionable.
        logger.error("database.error", exc_info=exc)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal server error",
            "internal_error",
        )

    app.include_router(health.router)
    app.include_router(api_router)
    return app
