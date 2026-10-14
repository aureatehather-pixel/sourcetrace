"""Alembic environment.

The database URL is resolved from typed application settings (DATABASE_URL),
never from alembic.ini - a committed connection string is a committed secret.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

from intelligence_engine.core.config import get_settings
from intelligence_engine.core.platform import run_async
from intelligence_engine.db.base import Base
from intelligence_engine.entities import models as entity_models  # noqa: F401

# Importing the models registers them on Base.metadata for autogenerate.
# Every model module must be listed here or autogenerate will propose
# dropping its table.
from intelligence_engine.evidence import models as evidence_models  # noqa: F401
from intelligence_engine.extraction import models as extraction_models  # noqa: F401
from intelligence_engine.intelligence import models as intelligence_models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Prefer an explicit override (tests, CI), otherwise typed settings."""
    override = config.get_main_option("sqlalchemy.url", None) or os.getenv("ALEMBIC_DATABASE_URL")
    if override:
        return override
    return get_settings().database_url_str


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    engine = async_engine_from_config(section, prefix="sqlalchemy.", poolclass=NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_async(run_migrations_online())
