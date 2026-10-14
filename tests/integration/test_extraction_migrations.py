"""The M2 migration chain, against a genuinely empty PostgreSQL database.

0001 -> 0002 -> 0003 and back. The downgrade path matters as much as the
upgrade: a migration that cannot be reversed cannot be safely deployed. These
also assert that M2 leaves the M1 schema exactly as 0001 built it.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.conftest import apply_migrations, create_empty_database
from tests.integration.test_migrations import EXPECTED_COLUMNS

pytestmark = pytest.mark.integration

EXPECTED_RUN_COLUMNS = {
    "id": ("uuid", "NO"),
    "evidence_id": ("uuid", "NO"),
    "input_content_hash": ("character varying", "NO"),
    "extractor_name": ("character varying", "NO"),
    "extractor_version": ("character varying", "NO"),
    "schema_version": ("integer", "NO"),
    "status": ("character varying", "NO"),
    "provider": ("character varying", "YES"),
    "model": ("character varying", "YES"),
    "started_at": ("timestamp with time zone", "NO"),
    "completed_at": ("timestamp with time zone", "YES"),
    "error_code": ("character varying", "YES"),
    "error_message": ("character varying", "YES"),
    "input_tokens": ("integer", "YES"),
    "output_tokens": ("integer", "YES"),
    "created_at": ("timestamp with time zone", "NO"),
    # 0007, narrowed by 0008. NOT NULL both: a run that cannot say which slice
    # of its source it read cannot prove its items are grounded in that source.
    # 0008 dropped `window_index` and `window_count` — an ordinal and a plan
    # size are only meaningful while the set of intervals is fixed, and adaptive
    # subdivision makes it dynamic. The interval itself is the identity.
    "window_start": ("integer", "NO"),
    "window_end": ("integer", "NO"),
}

#: The interval columns, added by 0007 and narrowed to two by 0008. Named
#: separately so milestones that predate them can assert the shape that existed
#: when they landed, instead of importing a set that silently grows every time
#: M2 changes.
WINDOW_COLUMNS = ("window_start", "window_end")

#: Columns 0007 introduced and 0008 removed. Kept named so a test can assert
#: their absence rather than merely not mentioning them.
DROPPED_ORDINAL_COLUMNS = ("window_index", "window_count")

#: `extraction_runs` as it stood before 0007. What M3 and M4 must leave alone.
EXPECTED_RUN_COLUMNS_BEFORE_WINDOWS = {
    name: spec for name, spec in EXPECTED_RUN_COLUMNS.items() if name not in WINDOW_COLUMNS
}

CURRENT_HEAD = "0009"

EXPECTED_ITEM_COLUMNS = {
    "id": ("uuid", "NO"),
    "extraction_run_id": ("uuid", "NO"),
    "kind": ("character varying", "NO"),
    "payload": ("jsonb", "NO"),
    "grounding_quote": ("text", "NO"),
    "position": ("integer", "NO"),
    "created_at": ("timestamp with time zone", "NO"),
}

# 0002 additionally carried model_confidence, dropped by 0003 as uncalibrated.
EXPECTED_ITEM_COLUMNS_AT_0002 = EXPECTED_ITEM_COLUMNS | {
    "model_confidence": ("character varying", "NO"),
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


async def test_empty_database_upgrades_to_head(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)

    await apply_migrations(url)

    assert await _columns(url, "extraction_runs") == EXPECTED_RUN_COLUMNS
    assert await _columns(url, "extraction_items") == EXPECTED_ITEM_COLUMNS
    assert await _scalar(url, "SELECT version_num FROM alembic_version") == CURRENT_HEAD


async def test_0003_drops_model_confidence(postgres_url: str) -> None:
    """The column exists at 0002 and is gone at 0003 - both asserted."""
    url = await create_empty_database(postgres_url)

    await apply_migrations(url, revision="0002")
    assert await _columns(url, "extraction_items") == EXPECTED_ITEM_COLUMNS_AT_0002

    await apply_migrations(url, revision="0003")
    assert await _columns(url, "extraction_items") == EXPECTED_ITEM_COLUMNS
    assert "model_confidence" not in await _columns(url, "extraction_items")


async def test_downgrading_0003_restores_the_0002_column_shape(postgres_url: str) -> None:
    """Shape, not data: the ratings are deliberately unrecoverable."""
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    await apply_migrations(url, revision="0002", downgrade=True)

    assert await _columns(url, "extraction_items") == EXPECTED_ITEM_COLUMNS_AT_0002
    # 0002 declared no server default; the downgrade must not leave one behind.
    default = await _scalar(
        url,
        "SELECT column_default FROM information_schema.columns "
        "WHERE table_name = 'extraction_items' AND column_name = 'model_confidence'",
    )
    assert default is None


async def test_m2_leaves_the_m1_evidence_schema_untouched(postgres_url: str) -> None:
    """M2 adds tables. It must not alter evidence_items in any way."""
    url = await create_empty_database(postgres_url)

    await apply_migrations(url, revision="0001")
    m1_columns = await _columns(url, "evidence_items")
    await apply_migrations(url, revision=CURRENT_HEAD)
    after_m2 = await _columns(url, "evidence_items")

    assert m1_columns == EXPECTED_COLUMNS
    assert after_m2 == m1_columns


async def test_downgrade_to_0001_removes_only_the_m2_tables(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    await apply_migrations(url, revision="0001", downgrade=True)

    assert await _scalar(url, "SELECT to_regclass('extraction_items')") is None
    assert await _scalar(url, "SELECT to_regclass('extraction_runs')") is None
    assert await _scalar(url, "SELECT to_regclass('evidence_items')") is not None
    assert await _columns(url, "evidence_items") == EXPECTED_COLUMNS
    assert await _scalar(url, "SELECT version_num FROM alembic_version") == "0001"


async def test_upgrade_downgrade_upgrade_round_trips(postgres_url: str) -> None:
    """The whole M2 chain, down to M1 and back up."""
    url = await create_empty_database(postgres_url)

    await apply_migrations(url, revision=CURRENT_HEAD)
    await apply_migrations(url, revision="0001", downgrade=True)
    await apply_migrations(url, revision=CURRENT_HEAD)

    assert await _columns(url, "extraction_runs") == EXPECTED_RUN_COLUMNS
    assert await _columns(url, "extraction_items") == EXPECTED_ITEM_COLUMNS
    assert await _columns(url, "evidence_items") == EXPECTED_COLUMNS
    assert await _scalar(url, "SELECT version_num FROM alembic_version") == CURRENT_HEAD


async def test_downgrade_to_base_removes_everything(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    await apply_migrations(url, revision="base")

    for table in ("extraction_items", "extraction_runs", "evidence_items"):
        assert await _scalar(url, f"SELECT to_regclass('{table}')") is None


async def test_reuse_index_is_partial_over_successful_runs(postgres_url: str) -> None:
    """The cache lookup must be an index scan, not a sequential scan."""
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    definition = await _scalar(
        url,
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_extraction_runs_reuse'",
    )

    assert isinstance(definition, str)
    assert "succeeded" in definition
    assert "UNIQUE" not in definition.upper()
    for column in (
        "evidence_id",
        "window_start",
        "window_end",
        "input_content_hash",
        "extractor_name",
        "extractor_version",
        "schema_version",
        # The generator is part of the reuse key, so it is part of the index
        # that serves it. Without these two, a run extracted by one vendor
        # answered a lookup made while configured for another.
        "provider",
        "model",
    ):
        assert column in definition
    # The ordinal must be gone from the key: a subdivided interval's child would
    # otherwise inherit a number an untouched sibling holds and be served that
    # sibling's stored items.
    assert "window_index" not in definition


async def test_no_unique_constraint_on_the_extraction_version_tuple(
    postgres_url: str,
) -> None:
    """Retries are history. A unique key here would silently discard attempts."""
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            uniques = (
                (
                    await connection.execute(
                        text(
                            "SELECT indexdef FROM pg_indexes "
                            "WHERE tablename = 'extraction_runs' AND indexdef ILIKE '%UNIQUE%'"
                        )
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await engine.dispose()

    # The primary key index is the only unique index on the table.
    assert all("pk_extraction_runs" in definition for definition in uniques)


async def test_foreign_keys_have_the_intended_delete_rules(postgres_url: str) -> None:
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            rules: dict[str, str] = dict(
                (
                    await connection.execute(
                        text(
                            "SELECT conname, confdeltype FROM pg_constraint "
                            "WHERE contype = 'f' AND conrelid::regclass::text "
                            "IN ('extraction_runs', 'extraction_items')"
                        )
                    )
                ).all()  # type: ignore[arg-type]
            )
    finally:
        await engine.dispose()

    # 'r' = RESTRICT, 'c' = CASCADE
    assert rules["fk_extraction_runs_evidence_id_evidence_items"] == "r"
    assert rules["fk_extraction_items_extraction_run_id_extraction_runs"] == "c"


# --- 0008: the interval becomes the durable identity -----------------------


async def test_0008_drops_the_ordinal_columns(postgres_url: str) -> None:
    """The ordinal and the plan size are gone, and the offsets remain.

    Not tidying. An ordinal is only meaningful while the set of intervals is
    fixed, and adaptive subdivision makes it dynamic: a child reusing a number
    an untouched sibling holds would be served that sibling's items by the reuse
    lookup and would evict it from every packet.
    """
    url = await create_empty_database(postgres_url)
    await apply_migrations(url)

    columns = await _columns(url, "extraction_runs")

    for dropped in DROPPED_ORDINAL_COLUMNS:
        assert dropped not in columns
    for kept in WINDOW_COLUMNS:
        assert columns[kept] == ("integer", "NO")


async def test_0008_preserves_the_intervals_recorded_under_0007(
    postgres_url: str,
) -> None:
    """The data that matters survives. Only the columns nobody may read go.

    A run's `[window_start, window_end)` is the audit trail for which text its
    items are grounded in, so an upgrade that disturbed it would break the
    grounding argument for every stored run at once.
    """
    url = await create_empty_database(postgres_url)
    await apply_migrations(url, revision="0007")

    evidence_id = uuid.uuid4()
    run_id = uuid.uuid4()
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO evidence_items (id, content, content_hash, created_at) "
                    "VALUES (:id, :content, :hash, now())"
                ),
                {"id": evidence_id, "content": "x" * 500, "hash": "b" * 64},
            )
            await connection.execute(
                text(
                    "INSERT INTO extraction_runs (id, evidence_id, input_content_hash, "
                    "extractor_name, extractor_version, schema_version, status, started_at, "
                    "completed_at, window_index, window_start, window_end, window_count) "
                    "VALUES (:id, :ev, :h, 'x', '1', 1, 'succeeded', now(), now(), 3, 120, 260, 6)"
                ),
                {"id": run_id, "ev": evidence_id, "h": "c" * 64},
            )
    finally:
        await engine.dispose()

    await apply_migrations(url, revision="0008")

    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            row = (
                await connection.execute(
                    text(
                        "SELECT window_start, window_end, status FROM extraction_runs "
                        "WHERE id = :id"
                    ),
                    {"id": run_id},
                )
            ).one()
    finally:
        await engine.dispose()

    assert (row[0], row[1], row[2]) == (120, 260, "succeeded")


async def test_0008_round_trips_without_losing_a_run(postgres_url: str) -> None:
    """Down to 0007 and back up. Ordinals are reconstructed, runs are not lost.

    The downgrade is faithful but approximate and says so: numbering is rebuilt
    from the offsets, so it need not match what 0007 originally stored. What it
    must never do is drop or duplicate a run, because those are extraction
    history.
    """
    url = await create_empty_database(postgres_url)
    await apply_migrations(url, revision="0007")

    evidence_id = uuid.uuid4()
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO evidence_items (id, content, content_hash, created_at) "
                    "VALUES (:id, :content, :hash, now())"
                ),
                {"id": evidence_id, "content": "y" * 300, "hash": "d" * 64},
            )
            for index, (start, end) in enumerate(((0, 100), (100, 200), (200, 300))):
                await connection.execute(
                    text(
                        "INSERT INTO extraction_runs (id, evidence_id, input_content_hash, "
                        "extractor_name, extractor_version, schema_version, status, started_at, "
                        "completed_at, window_index, window_start, window_end, window_count) "
                        "VALUES (:id, :ev, :h, 'x', '1', 1, 'succeeded', now(), now(), "
                        ":i, :s, :e, 3)"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "ev": evidence_id,
                        "h": f"{index}" * 64,
                        "i": index,
                        "s": start,
                        "e": end,
                    },
                )
    finally:
        await engine.dispose()

    await apply_migrations(url, revision="0008")
    await apply_migrations(url, revision="0007", downgrade=True)

    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        "SELECT window_index, window_count, window_start, window_end "
                        "FROM extraction_runs ORDER BY window_start"
                    )
                )
            ).all()
    finally:
        await engine.dispose()

    assert [(r[2], r[3]) for r in rows] == [(0, 100), (100, 200), (200, 300)]
    # A valid dense numbering, which is what the recreated CHECK requires.
    assert [r[0] for r in rows] == [0, 1, 2]
    assert {r[1] for r in rows} == {3}
