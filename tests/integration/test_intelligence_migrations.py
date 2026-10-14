"""The M4 migration, against a genuinely empty PostgreSQL database.

0004 -> 0005 and back. Beyond shape, these assert the constraints that carry
M4's epistemic guarantees, because a constraint that exists in the ORM and not
in the database is not a guarantee at all:

- a pair has exactly one row, in canonical order;
- only assertion-bearing item kinds can stand in a relation;
- only `qualifies` may state a direction, and it must;
- a meaningful relation cannot be stored without an explanation;
- one settled answer per pair per classifier version, rejected ones included;
- nothing about M1, M2 or M3 changed at all.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.conftest import apply_migrations, create_empty_database
from tests.integration.test_entity_migrations import (
    EXPECTED_ENTITY_COLUMNS,
    EXPECTED_RESOLUTION_COLUMNS,
)
from tests.integration.test_extraction_migrations import (
    EXPECTED_ITEM_COLUMNS,
    EXPECTED_RUN_COLUMNS_BEFORE_WINDOWS,
)
from tests.integration.test_migrations import EXPECTED_COLUMNS

pytestmark = pytest.mark.integration

CURRENT_HEAD = "0009"
M3_HEAD = "0004"
#: The last revision M4 itself created. This milestone's "touches nothing"
#: claim is about M4's own migrations; 0007 is a later M2 change and is not
#: evidence against it.
M4_HEAD = "0005"

EXPECTED_ANALYSIS_RUN_COLUMNS = {
    "id": ("uuid", "NO"),
    "entity_id": ("uuid", "NO"),
    "classifier_name": ("character varying", "NO"),
    "classifier_version": ("character varying", "NO"),
    "schema_version": ("integer", "NO"),
    "status": ("character varying", "NO"),
    # Still nullable, unlike the relation columns: an analysis run is a record
    # of an attempt, and a run that ended before a provider was recorded is
    # real history. Only `intelligence_relations` needs them NOT NULL, because
    # only there are they part of a unique constraint.
    "provider": ("character varying", "YES"),
    "model": ("character varying", "YES"),
    "started_at": ("timestamp with time zone", "NO"),
    "completed_at": ("timestamp with time zone", "YES"),
    "error_code": ("character varying", "YES"),
    "error_message": ("character varying", "YES"),
    "theoretical_pairs": ("integer", "NO"),
    "candidate_pairs": ("integer", "NO"),
    "reused_pairs": ("integer", "NO"),
    "classified_pairs": ("integer", "NO"),
    "batch_count": ("integer", "NO"),
    "external_call_count": ("integer", "NO"),
    "input_tokens": ("integer", "YES"),
    "output_tokens": ("integer", "YES"),
    "created_at": ("timestamp with time zone", "NO"),
}

EXPECTED_RELATION_COLUMNS = {
    "id": ("uuid", "NO"),
    "analysis_run_id": ("uuid", "NO"),
    "entity_id": ("uuid", "NO"),
    "left_extraction_item_id": ("uuid", "NO"),
    "right_extraction_item_id": ("uuid", "NO"),
    "left_item_kind": ("character varying", "NO"),
    "right_item_kind": ("character varying", "NO"),
    "relation_type": ("character varying", "NO"),
    "direction": ("character varying", "NO"),
    "explanation": ("text", "YES"),
    "classifier_name": ("character varying", "NO"),
    "classifier_version": ("character varying", "NO"),
    "schema_version": ("integer", "NO"),
    "provider": ("character varying", "NO"),
    "model": ("character varying", "NO"),
    "input_fingerprint": ("character varying", "NO"),
    "status": ("character varying", "NO"),
    "rejected_at": ("timestamp with time zone", "YES"),
    "rejection_note": ("character varying", "YES"),
    "created_at": ("timestamp with time zone", "NO"),
}

M1_M2_M3_TABLES = (
    "evidence_items",
    "extraction_runs",
    "extraction_items",
    "entities",
    "entity_resolutions",
)


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


async def _constraint_names(url: str, table: str) -> set[str]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    # cast(... AS regclass) rather than `:t::regclass`: the `::`
                    # cast operator collides with SQLAlchemy's bind-parameter syntax.
                    text("SELECT conname FROM pg_constraint WHERE conrelid = cast(:t AS regclass)"),
                    {"t": table},
                )
            ).all()
    finally:
        await engine.dispose()
    return {row[0] for row in rows}


# --- shape ----------------------------------------------------------------


async def test_empty_database_upgrades_to_the_m4_head(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)

    await apply_migrations(url)

    assert await _columns(url, "relation_analysis_runs") == EXPECTED_ANALYSIS_RUN_COLUMNS
    assert await _columns(url, "intelligence_relations") == EXPECTED_RELATION_COLUMNS
    assert await _scalar(url, "SELECT version_num FROM alembic_version") == CURRENT_HEAD


async def test_m4_changes_no_earlier_table_at_all(postgres_url: str) -> None:
    """M4 adds two tables and touches nothing.

    Stronger than the equivalent M3 assertion, which had to allow one added
    unique constraint. M4 needs no schema change beneath it because
    `unique_id_kind` already exists and is exactly the target both composite
    foreign keys need.
    """
    url = await create_empty_database(postgres_url)

    await apply_migrations(url, revision=M3_HEAD)
    columns_before = {table: await _columns(url, table) for table in M1_M2_M3_TABLES}
    constraints_before = {table: await _constraint_names(url, table) for table in M1_M2_M3_TABLES}

    # Ends at M4's own head. Running to the global head would fold 0007 — an
    # M2 change that deliberately adds four columns to `extraction_runs` — into
    # a claim that is only about what M4 did.
    await apply_migrations(url, revision=M4_HEAD)
    columns_after = {table: await _columns(url, table) for table in M1_M2_M3_TABLES}
    constraints_after = {table: await _constraint_names(url, table) for table in M1_M2_M3_TABLES}

    assert columns_before["evidence_items"] == EXPECTED_COLUMNS
    assert columns_before["extraction_runs"] == EXPECTED_RUN_COLUMNS_BEFORE_WINDOWS
    assert columns_before["extraction_items"] == EXPECTED_ITEM_COLUMNS
    assert columns_before["entities"] == EXPECTED_ENTITY_COLUMNS
    assert columns_before["entity_resolutions"] == EXPECTED_RESOLUTION_COLUMNS

    assert columns_after == columns_before
    assert constraints_after == constraints_before


async def test_the_composite_fks_reuse_the_existing_m3_target(postgres_url: str) -> None:
    """Both item foreign keys point at `unique_id_kind`, added by 0004."""
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    names = await _constraint_names(url, "intelligence_relations")
    assert "fk_intelligence_relations_left_item" in names
    assert "fk_intelligence_relations_right_item" in names
    assert "unique_id_kind" in await _constraint_names(url, "extraction_items")


async def test_every_foreign_key_out_of_m4_is_restrict(postgres_url: str) -> None:
    """Derived intelligence never lets its source material be deleted."""
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        "SELECT conname, confdeltype FROM pg_constraint "
                        "WHERE contype = 'f' AND conrelid IN "
                        "('intelligence_relations'::regclass, "
                        "'relation_analysis_runs'::regclass)"
                    )
                )
            ).all()
    finally:
        await engine.dispose()

    assert rows
    # 'r' is RESTRICT; 'a' would be NO ACTION, 'c' CASCADE.
    assert all(row[1] == "r" for row in rows), rows


# --- the constraints that carry the epistemics ----------------------------


async def test_a_pair_must_be_canonically_ordered(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)
    assert "ck_intelligence_relations_pair_is_canonically_ordered" in await _constraint_names(
        url, "intelligence_relations"
    )


async def test_only_assertion_bearing_kinds_may_relate(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    definition = await _scalar(
        url,
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conname = 'ck_intelligence_relations_left_item_kind_is_relatable'",
    )
    assert definition is not None
    assert "reported_claim" in str(definition)
    assert "observation" in str(definition)
    # A mention names a thing; naming a thing cannot qualify or contradict.
    assert "entity_mention" not in str(definition)


async def test_direction_is_tied_to_relation_type_in_the_database(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    definition = await _scalar(
        url,
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conname = 'ck_intelligence_relations_direction_matches_relation_type'",
    )
    assert definition is not None
    assert "qualifies" in str(definition)
    assert "symmetric" in str(definition)


async def test_a_meaningful_relation_must_be_explained(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    definition = await _scalar(
        url,
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conname = 'ck_intelligence_relations_meaningful_relation_is_explained'",
    )
    assert definition is not None
    assert "no_meaningful_relation" in str(definition)


async def test_one_settled_answer_per_pair_per_classifier_version(postgres_url: str) -> None:
    """The reuse mechanism, as a database constraint rather than a convention."""
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    definition = await _scalar(
        url,
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conname = 'unique_settled_pair_per_classifier'",
    )
    assert definition is not None
    rendered = str(definition)
    for column in (
        "left_extraction_item_id",
        "right_extraction_item_id",
        "classifier_name",
        "classifier_version",
        "schema_version",
    ):
        assert column in rendered
    # Deliberately NOT partial on status: a rejected relation is a settled
    # question, and regenerating it would pay for the same wrong answer again.
    assert "WHERE" not in rendered.upper()


async def test_narrowing_can_only_reduce(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    names = await _constraint_names(url, "relation_analysis_runs")
    assert "ck_relation_analysis_runs_candidates_within_theoretical" in names
    assert "ck_relation_analysis_runs_classified_within_candidates" in names


# --- round trip -----------------------------------------------------------


async def test_downgrade_removes_only_the_m4_tables(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    await apply_migrations(url, revision=M3_HEAD, downgrade=True)

    assert await _scalar(url, "SELECT to_regclass('intelligence_relations')") is None
    assert await _scalar(url, "SELECT to_regclass('relation_analysis_runs')") is None
    for table in M1_M2_M3_TABLES:
        assert await _scalar(url, f"SELECT to_regclass('{table}')") is not None
    assert await _scalar(url, "SELECT version_num FROM alembic_version") == M3_HEAD


async def test_the_migration_round_trips(postgres_url: str) -> None:
    """0004 -> 0005 -> 0004 -> 0005, as an operator would run it."""
    url = await create_empty_database(postgres_url)
    await apply_migrations(url, revision=M3_HEAD)

    await apply_migrations(url, revision=CURRENT_HEAD)
    await apply_migrations(url, revision=M3_HEAD, downgrade=True)
    await apply_migrations(url, revision=CURRENT_HEAD)

    assert await _columns(url, "intelligence_relations") == EXPECTED_RELATION_COLUMNS
    assert await _scalar(url, "SELECT version_num FROM alembic_version") == CURRENT_HEAD


async def test_the_full_chain_downgrades_to_nothing(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    await apply_migrations(url, revision="base")

    for table in (*M1_M2_M3_TABLES, "intelligence_relations", "relation_analysis_runs"):
        assert await _scalar(url, f"SELECT to_regclass('{table}')") is None
