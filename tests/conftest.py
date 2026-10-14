"""Shared fixtures.

Integration tests run against a real, disposable PostgreSQL 16 container and
apply the real migrations. SQLite is never substituted: it does not share
PostgreSQL's JSONB, TIMESTAMPTZ, regex CHECK or row-comparison semantics, so a
green SQLite suite would prove nothing about production behaviour.
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from intelligence_engine.core.config import Settings
from intelligence_engine.core.platform import psycopg_loop_factory
from intelligence_engine.db.session import create_engine, create_session_factory
from tests.integration_gate import MARKER, integration_gate_message
from tests.loop_factories import LoopFactory, required_loop_factories

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSTGRES_IMAGE = "postgres:16-alpine"


_LOOP_FACTORIES = required_loop_factories(psycopg_loop_factory())

if _LOOP_FACTORIES is not None:
    # Defined only where psycopg needs a different loop; see tests/loop_factories.py
    # for why returning None from this hook is not an option.
    _REQUIRED_LOOP_FACTORIES: dict[str, LoopFactory] = _LOOP_FACTORIES

    def pytest_asyncio_loop_factories(
        config: pytest.Config, item: pytest.Item
    ) -> dict[str, LoopFactory]:
        return _REQUIRED_LOOP_FACTORIES


def _docker_available() -> bool:
    try:
        import docker  # type: ignore[import-untyped]

        docker.from_env().ping()
    except Exception:
        return False
    return True


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Fail the run rather than let selected integration tests skip themselves.

    `trylast` is load-bearing, not tidiness. `-m` deselection is itself a
    `pytest_collection_modifyitems` implementation, and without `trylast` this
    one runs first and sees the integration items that are about to be dropped —
    so `pytest -m "not integration"`, the documented no-Docker path, failed on
    Docker being missing. Running last means `items` is the final selection.

    Only pays for the Docker probe when integration tests survive selection.
    `pytest.UsageError` is what makes this one loud line instead of 222
    identical skip reasons.
    """
    selected = sum(1 for item in items if item.get_closest_marker(MARKER) is not None)
    if selected == 0:
        return
    message = integration_gate_message(
        docker_available=_docker_available(), selected_integration_tests=selected
    )
    if message:
        raise pytest.UsageError(message)


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """A disposable PostgreSQL instance for the whole test session.

    Deliberately does not skip when Docker is missing: `pytest_collection_modifyitems`
    has already failed the run by this point. A skip here would be a second,
    quieter answer to the same question and could only ever hide one.
    """
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer(POSTGRES_IMAGE, driver="psycopg") as container:
        yield container.get_connection_url()


def run_migrations(database_url: str, revision: str = "head", *, downgrade: bool = False) -> None:
    """Apply migrations to ``database_url`` exactly as an operator would.

    Direction is explicit because a target revision alone does not imply it:
    "0001" is an upgrade from an empty database and a downgrade from 0002.
    "base" can only ever be a downgrade.
    """
    from alembic import command
    from alembic.config import Config

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    if downgrade or revision == "base":
        command.downgrade(config, revision)
    else:
        command.upgrade(config, revision)


@pytest.fixture(scope="session")
def migrated_url(postgres_url: str) -> str:
    run_migrations(postgres_url)
    return postgres_url


@pytest.fixture(scope="session")
def settings(migrated_url: str) -> Settings:
    return Settings(
        environment="test",
        database_url=migrated_url,
        log_level="WARNING",
    )


@pytest.fixture
async def engine(migrated_url: str) -> AsyncIterator[AsyncEngine]:
    created = create_engine(migrated_url)
    try:
        yield created
    finally:
        await created.dispose()


@pytest.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return create_session_factory(engine)


@pytest.fixture
async def session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as active:
        yield active


def make_empty_database(admin_url: str) -> str:
    """Create a brand-new empty database and return its URL.

    Used to prove `alembic upgrade head` builds the schema from nothing.
    """
    import psycopg
    from sqlalchemy.engine import make_url

    url = make_url(admin_url)
    name = f"pie_empty_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(
        host=url.host,
        port=url.port,
        user=url.username,
        password=url.password,
        dbname=url.database,
        autocommit=True,
    ) as connection:
        connection.execute(f'CREATE DATABASE "{name}"')
    # render_as_string(hide_password=False): str(URL) masks the password as '***'.
    return url.set(database=name).render_as_string(hide_password=False)


async def apply_migrations(
    database_url: str, revision: str = "head", *, downgrade: bool = False
) -> None:
    """``run_migrations`` from async test code.

    Alembic's env.py opens its own event loop, so it must run off the test's.
    """
    await asyncio.to_thread(
        functools.partial(run_migrations, database_url, revision, downgrade=downgrade)
    )


async def create_empty_database(admin_url: str) -> str:
    """``make_empty_database`` from async test code (psycopg sync connection)."""
    return await asyncio.to_thread(make_empty_database, admin_url)
