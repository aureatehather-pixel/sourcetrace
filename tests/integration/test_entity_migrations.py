"""The M3 migration, against a genuinely empty PostgreSQL database.

0003 -> 0004 and back. Beyond shape, these assert the constraints that carry
M3's epistemic guarantees, because a constraint that exists in the ORM and not
in the database is not a guarantee at all:

- names are NOT unique (many John Smiths must coexist);
- one item can hold at most one *active* resolution per role;
- a role can only describe the item kind it belongs to;
- nothing about M1 or M2 changed except the redundant unique constraint that
  gives the composite foreign key a legal target.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.conftest import apply_migrations, create_empty_database
from tests.integration.test_extraction_migrations import (
    EXPECTED_ITEM_COLUMNS,
    EXPECTED_RUN_COLUMNS_BEFORE_WINDOWS,
)
from tests.integration.test_migrations import EXPECTED_COLUMNS

pytestmark = pytest.mark.integration

CURRENT_HEAD = "0009"
M2_HEAD = "0003"
#: The last revision M3 itself created. This milestone's "changes nothing
#: earlier" claim is about its own migrations, so it is measured against this
#: rather than against whatever the newest global head happens to be.
M3_HEAD = "0004"

EXPECTED_ENTITY_COLUMNS = {
    "id": ("uuid", "NO"),
    "entity_type": ("character varying", "NO"),
    "canonical_name": ("text", "NO"),
    "normalized_name": ("text", "NO"),
    "disambiguator": ("text", "YES"),
    "created_at": ("timestamp with time zone", "NO"),
}

EXPECTED_RESOLUTION_COLUMNS = {
    "id": ("uuid", "NO"),
    "extraction_item_id": ("uuid", "NO"),
    "source_item_kind": ("character varying", "NO"),
    "entity_id": ("uuid", "NO"),
    "role": ("character varying", "NO"),
    "method": ("character varying", "NO"),
    "is_active": ("boolean", "NO"),
    "superseded_by_id": ("uuid", "YES"),
    "superseded_at": ("timestamp with time zone", "YES"),
    "note": ("character varying", "YES"),
    "created_at": ("timestamp with time zone", "NO"),
}


async def _columns(url: str, table: str) -> dict[str, tuple[str, str]]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        "SELECT column_name, data_type, is_nullable "
                        "FROM information_schema.columns WHERE table_name = :t"
                    ),
                    {"t": table},
                )
            ).all()
    finally:
        await engine.dispose()
    return {row[0]: (row[1], row[2]) for row in rows}


async def _scalar(url: str, sql: str) -> object:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            return (await connection.execute(text(sql))).scalar()
    finally:
        await engine.dispose()


async def _rows(url: str, sql: str) -> list[tuple[object, ...]]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            return [tuple(row) for row in (await connection.execute(text(sql))).all()]
    finally:
        await engine.dispose()


# --- shape ----------------------------------------------------------------


async def test_empty_database_upgrades_to_the_m3_head(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)

    await apply_migrations(url)

    assert await _columns(url, "entities") == EXPECTED_ENTITY_COLUMNS
    assert await _columns(url, "entity_resolutions") == EXPECTED_RESOLUTION_COLUMNS
    assert await _scalar(url, "SELECT version_num FROM alembic_version") == CURRENT_HEAD


async def test_m3_leaves_the_m1_and_m2_schemas_intact(postgres_url: str) -> None:
    """M3 adds tables. The M1/M2 columns must be identical before and after."""
    url = await create_empty_database(postgres_url)

    await apply_migrations(url, revision=M2_HEAD)
    before = {
        table: await _columns(url, table)
        for table in ("evidence_items", "extraction_runs", "extraction_items")
    }

    # Upgraded to the end of M3, not to the global head. Later milestones are
    # free to change M2's own tables — 0007 does exactly that — and folding
    # those changes in here would turn a statement about M3 into a statement
    # about every migration ever written after it.
    await apply_migrations(url, revision=M3_HEAD)
    after = {
        table: await _columns(url, table)
        for table in ("evidence_items", "extraction_runs", "extraction_items")
    }

    assert before["evidence_items"] == EXPECTED_COLUMNS
    assert before["extraction_runs"] == EXPECTED_RUN_COLUMNS_BEFORE_WINDOWS
    assert before["extraction_items"] == EXPECTED_ITEM_COLUMNS
    assert after == before


async def test_the_only_change_to_m2_is_the_composite_fk_target(postgres_url: str) -> None:
    """`unique_id_kind` is the single addition, and it is redundant by design."""
    url = await create_empty_database(postgres_url)

    await apply_migrations(url, revision=M2_HEAD)
    before = set(
        await _rows(
            url,
            "SELECT conname FROM pg_constraint WHERE conrelid = 'extraction_items'::regclass",
        )
    )
    await apply_migrations(url, revision=CURRENT_HEAD)
    after = set(
        await _rows(
            url,
            "SELECT conname FROM pg_constraint WHERE conrelid = 'extraction_items'::regclass",
        )
    )

    assert after - before == {("unique_id_kind",)}
    assert before - after == set()


# --- constraints that carry the guarantees --------------------------------


async def test_entity_names_are_deliberately_not_unique(postgres_url: str) -> None:
    """Two people are both called John Smith. Neither may be rejected."""
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    uniques = await _rows(
        url,
        "SELECT indexdef FROM pg_indexes "
        "WHERE tablename = 'entities' AND indexdef ILIKE '%UNIQUE%'",
    )

    # The primary key is the only unique index on the table.
    assert all("pk_entities" in str(row[0]) for row in uniques)
    assert not any("canonical_name" in str(row[0]) for row in uniques)
    assert not any("normalized_name" in str(row[0]) for row in uniques)


async def test_the_match_key_index_exists_and_is_type_aware(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    definition = await _scalar(
        url,
        "SELECT indexdef FROM pg_indexes "
        "WHERE indexname = 'ix_entities_entity_type_normalized_name'",
    )

    assert isinstance(definition, str)
    assert "entity_type" in definition
    assert "normalized_name" in definition
    assert "UNIQUE" not in definition.upper()


async def test_active_resolutions_are_unique_per_item_and_role(postgres_url: str) -> None:
    """A mention cannot currently refer to two entities. History is unconstrained."""
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    definition = await _scalar(
        url,
        "SELECT indexdef FROM pg_indexes "
        "WHERE indexname = 'uq_entity_resolutions_active_item_role'",
    )

    assert isinstance(definition, str)
    assert "UNIQUE" in definition.upper()
    assert "extraction_item_id" in definition
    assert "role" in definition
    assert "is_active" in definition  # partial: superseded rows are exempt


async def test_the_expected_check_constraints_exist(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    checks = {
        str(row[0])
        for row in await _rows(
            url,
            "SELECT conname FROM pg_constraint WHERE contype = 'c' AND conrelid::regclass::text "
            "IN ('entities', 'entity_resolutions')",
        )
    }

    assert checks == {
        "ck_entities_canonical_name_not_empty",
        "ck_entities_normalized_name_not_empty",
        "ck_entities_disambiguator_not_empty",
        "ck_entity_resolutions_role_matches_source_item_kind",
        "ck_entity_resolutions_superseded_state_is_consistent",
        "ck_entity_resolutions_cannot_supersede_itself",
        "ck_entity_resolutions_note_not_empty",
    }


async def test_all_resolution_foreign_keys_restrict(postgres_url: str) -> None:
    """Nothing a resolution depends on can be deleted out from under it."""
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    rules = {
        str(row[0]): str(row[1])
        for row in await _rows(
            url,
            "SELECT conname, confdeltype FROM pg_constraint "
            "WHERE contype = 'f' AND conrelid::regclass::text = 'entity_resolutions'",
        )
    }

    # 'r' = RESTRICT
    assert rules == {
        "fk_entity_resolutions_extraction_item": "r",
        "fk_entity_resolutions_entity_id_entities": "r",
        "fk_entity_resolutions_superseded_by_id_entity_resolutions": "r",
    }


async def test_the_composite_fk_references_both_id_and_kind(postgres_url: str) -> None:
    """This is what makes the role/kind CHECK trustworthy rather than decorative."""
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    definition = await _scalar(
        url,
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conname = 'fk_entity_resolutions_extraction_item'",
    )

    assert isinstance(definition, str)
    assert "extraction_item_id" in definition
    assert "source_item_kind" in definition
    assert "extraction_items(id, kind)" in definition


# --- round trip -----------------------------------------------------------


async def test_downgrade_to_0003_removes_only_the_m3_tables(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    await apply_migrations(url, revision=M2_HEAD, downgrade=True)

    assert await _scalar(url, "SELECT to_regclass('entity_resolutions')") is None
    assert await _scalar(url, "SELECT to_regclass('entities')") is None
    assert await _columns(url, "extraction_items") == EXPECTED_ITEM_COLUMNS
    # At 0003 the window columns do not exist yet; 0007 is far above this point.
    assert await _columns(url, "extraction_runs") == EXPECTED_RUN_COLUMNS_BEFORE_WINDOWS
    assert await _columns(url, "evidence_items") == EXPECTED_COLUMNS
    # The composite FK target goes with the tables that needed it.
    assert (
        await _scalar(
            url,
            "SELECT to_regclass('unique_id_kind')",
        )
        is None
    )
    assert await _scalar(url, "SELECT version_num FROM alembic_version") == M2_HEAD


async def test_upgrade_downgrade_upgrade_round_trips(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)

    await apply_migrations(url, revision=CURRENT_HEAD)
    await apply_migrations(url, revision=M2_HEAD, downgrade=True)
    await apply_migrations(url, revision=CURRENT_HEAD)

    assert await _columns(url, "entities") == EXPECTED_ENTITY_COLUMNS
    assert await _columns(url, "entity_resolutions") == EXPECTED_RESOLUTION_COLUMNS
    assert await _columns(url, "extraction_items") == EXPECTED_ITEM_COLUMNS
    assert await _columns(url, "evidence_items") == EXPECTED_COLUMNS
    assert await _scalar(url, "SELECT version_num FROM alembic_version") == CURRENT_HEAD


async def test_downgrade_to_base_removes_everything(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    await apply_migrations(url, revision="base")

    for table in (
        "entity_resolutions",
        "entities",
        "extraction_items",
        "extraction_runs",
        "evidence_items",
    ):
        assert await _scalar(url, f"SELECT to_regclass('{table}')") is None
