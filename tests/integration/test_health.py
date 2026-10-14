"""Health and readiness, including the case that actually matters: failure.

A probe that only ever returns 200 proves nothing. These tests assert that
readiness reports 503 when PostgreSQL is unreachable, and that it does so within
a bounded time rather than hanging.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from intelligence_engine.core.config import Settings
from intelligence_engine.health import READINESS_TIMEOUT_SECONDS
from intelligence_engine.main import create_app

pytestmark = pytest.mark.integration


async def _client_for(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as http:
            yield http


async def test_ready_returns_200_when_the_database_is_reachable(settings: Settings) -> None:
    async for http in _client_for(settings):
        response = await http.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


async def test_ready_returns_503_when_the_database_is_unreachable(settings: Settings) -> None:
    """Prove the gate rejects. A port with nothing behind it is 'down'."""
    from sqlalchemy.engine import make_url

    dead_url = make_url(settings.database_url_str).set(port=1)
    broken = settings.model_copy(
        update={"database_url": dead_url.render_as_string(hide_password=False)}
    )

    async for http in _client_for(broken):
        response = await http.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": "unavailable"}


async def test_ready_answers_within_its_timeout_budget(settings: Settings) -> None:
    """A hanging probe is a failed probe: the answer must be bounded."""
    from sqlalchemy.engine import make_url

    dead_url = make_url(settings.database_url_str).set(port=1)
    broken = settings.model_copy(
        update={"database_url": dead_url.render_as_string(hide_password=False)}
    )

    started = time.perf_counter()
    async for http in _client_for(broken):
        response = await http.get("/ready")
    elapsed = time.perf_counter() - started

    assert response.status_code == 503
    assert elapsed < READINESS_TIMEOUT_SECONDS + 2.0, f"readiness took {elapsed:.1f}s"


async def test_health_stays_200_when_the_database_is_unreachable(settings: Settings) -> None:
    """Liveness is about this process, not its dependencies."""
    from sqlalchemy.engine import make_url

    dead_url = make_url(settings.database_url_str).set(port=1)
    broken = settings.model_copy(
        update={"database_url": dead_url.render_as_string(hide_password=False)}
    )

    async for http in _client_for(broken):
        response = await http.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
