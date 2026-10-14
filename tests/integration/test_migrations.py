"""An empty PostgreSQL database must reach the current schema through migrations."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.conftest import apply_migrations, create_empty_database

pytestmark = pytest.mark.integration

CURRENT_HEAD = "0009"

EXPECTED_COLUMNS = {
    "id": ("uuid", "NO"),
    "content": ("text", "NO"),
    "content_hash": ("character varying", "NO"),
    "source_type": ("character varying", "YES"),
    "source_label": ("text", "YES"),
    "source_url": ("text", "YES"),
    "captured_at": ("timestamp with time zone", "YES"),
    "created_at": ("timestamp with time zone", "NO"),
    "metadata": ("jsonb", "NO"),
}


async def _inspect(url: str) -> dict[str, object]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            columns = (
                await connection.execute(
                    text(
                        "SELECT column_name, data_type, is_nullable "
                        "FROM information_schema.columns "
                        "WHERE table_name = 'evidence_items'"
                    )
                )
            ).all()
            indexes = (
                await connection.execute(
                    text(
                        "SELECT indexname, indexdef FROM pg_indexes "
                        "WHERE tablename = 'evidence_items'"
                    )
                )
            ).all()
            checks = (
                (
                    await connection.execute(
                        text(
                            "SELECT conname FROM pg_constraint "
                            "WHERE conrelid = 'evidence_items'::regclass AND contype = 'c'"
                        )
                    )
                )
                .scalars()
                .all()
            )
            revision = (
                await connection.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one()
    finally:
        await engine.dispose()

    return {
        "columns": {row[0]: (row[1], row[2]) for row in columns},
        "indexes": {row[0]: row[1] for row in indexes},
        "checks": set(checks),
        "revision": revision,
    }


async def test_upgrade_head_builds_the_schema_from_an_empty_database(
    postgres_url: str,
) -> None:
    empty_url = await create_empty_database(postgres_url)

    await apply_migrations(empty_url)
    schema = await _inspect(empty_url)

    assert schema["columns"] == EXPECTED_COLUMNS
    # Pinned deliberately: this must be updated with each new migration, which
    # forces a conscious check that head is what the schema assertions expect.
    assert schema["revision"] == CURRENT_HEAD


async def test_hash_index_exists_and_is_deliberately_not_unique(postgres_url: str) -> None:
    """Identical text from different sources must be able to coexist."""
    empty_url = await create_empty_database(postgres_url)
    await apply_migrations(empty_url)

    indexes: dict[str, str] = (await _inspect(empty_url))["indexes"]  # type: ignore[assignment]

    definition = indexes["ix_evidence_items_content_hash"]
    assert "UNIQUE" not in definition.upper()
    assert not any(
        "UNIQUE" in other.upper() and "content_hash" in other
        for name, other in indexes.items()
        if name != "pk_evidence_items"
    )


async def test_pagination_index_matches_the_list_ordering(postgres_url: str) -> None:
    empty_url = await create_empty_database(postgres_url)
    await apply_migrations(empty_url)

    indexes: dict[str, str] = (await _inspect(empty_url))["indexes"]  # type: ignore[assignment]

    definition = indexes["ix_evidence_items_created_at_id"].lower()
    assert "created_at desc" in definition
    assert "id desc" in definition


async def test_check_constraints_are_created(postgres_url: str) -> None:
    empty_url = await create_empty_database(postgres_url)
    await apply_migrations(empty_url)

    checks = (await _inspect(empty_url))["checks"]

    assert checks == {
        "ck_evidence_items_content_not_empty",
        "ck_evidence_items_content_hash_is_sha256_hex",
        "ck_evidence_items_metadata_is_object",
    }


async def test_downgrade_then_upgrade_round_trips(postgres_url: str) -> None:
    """A migration that cannot be reversed is a migration that cannot be reviewed."""
    empty_url = await create_empty_database(postgres_url)

    await apply_migrations(empty_url)
    await apply_migrations(empty_url, revision="base")

    engine = create_async_engine(empty_url)
    try:
        async with engine.connect() as connection:
            exists = (
                await connection.execute(text("SELECT to_regclass('evidence_items')"))
            ).scalar_one()
    finally:
        await engine.dispose()
    assert exists is None

    await apply_migrations(empty_url)
    assert (await _inspect(empty_url))["columns"] == EXPECTED_COLUMNS
