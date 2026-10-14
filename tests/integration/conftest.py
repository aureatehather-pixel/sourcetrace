"""Integration-only fixtures.

Confined to this package so the unit suite never touches Docker.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text

from intelligence_engine.db.session import create_engine

# All seven together, in one statement: every foreign key in the chain is
# ON DELETE RESTRICT, so truncating any table alone is (correctly) refused.
TRUNCATE_ALL = text(
    "TRUNCATE TABLE intelligence_relations, relation_analysis_runs, entity_resolutions, "
    "entities, extraction_items, extraction_runs, evidence_items"
)


@pytest.fixture(autouse=True)
async def clean_evidence(migrated_url: str) -> AsyncIterator[None]:
    """Truncate before each test so ordering and count assertions are exact."""
    engine = create_engine(migrated_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(TRUNCATE_ALL)
    finally:
        await engine.dispose()
    yield
