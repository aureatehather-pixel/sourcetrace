"""The synthetic demo does what it says it does.

The demo narrates outcomes - a paraphrase rejected, an ambiguous name declined, a
rerun that costs nothing, a citation the gate cannot see through. A narration
that drifted from the system would be worse than no demo, so each outcome it
prints is asserted here, against a fresh database of its own: the packet is
corpus-wide, and sharing the suite's database would make both sides
order-dependent.
"""

from __future__ import annotations

import pytest
from demo.run_demo import walkthrough

from intelligence_engine.db.session import create_engine, create_session_factory
from tests.conftest import apply_migrations, create_empty_database

pytestmark = pytest.mark.integration


async def test_the_demo_narrates_what_the_system_actually_did(postgres_url: str) -> None:
    database_url = await create_empty_database(postgres_url)
    await apply_migrations(database_url)
    engine = create_engine(database_url)
    try:
        result = await walkthrough(create_session_factory(engine))
    finally:
        await engine.dispose()

    assert result.paraphrase_rejected
    assert result.verbatim_accepted
    assert result.dana_claims_resolved == 4
    assert result.ambiguous_declined
    assert result.correction_history_rows == 2
    assert sorted(result.relation_types) == [
        "no_meaningful_relation",
        "no_meaningful_relation",
        "qualifies",
        "reinforces",
    ]
    assert result.rerun_model_calls == 0
    assert result.gate_accepted_verbatim
    assert result.gate_rejected_paraphrase
    assert result.gate_rejected_ambiguous
    assert result.gate_accepted_unsupported_claim, "the gate's limit is part of the demo"
