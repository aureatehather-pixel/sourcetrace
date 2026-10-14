"""The window budget is declared in two places. They must agree.

`Settings.extraction_window_max_chars` is what the CLI and the API pass in;
`ExtractionService.DEFAULT_WINDOW_MAX_CHARS` is what the service falls back to
when constructed without one, which is how most tests build it. A silent
disagreement would mean tests exercising a different division of a document than
production performs — the tests would pass and the real corpus would still fail.
"""

from __future__ import annotations

import pytest

from intelligence_engine.core.config import Settings
from intelligence_engine.extraction.service import DEFAULT_WINDOW_MAX_CHARS


@pytest.fixture
def default_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings carrying the declared defaults, and nothing from this machine.

    A developer's `.env` or exported `EXTRACTION_WINDOW_MAX_CHARS` would otherwise
    stand in for the default these tests are about, and a clean checkout with
    neither would fail on the unrelated required `DATABASE_URL`.
    """
    monkeypatch.delenv("EXTRACTION_WINDOW_MAX_CHARS", raising=False)
    return Settings(
        _env_file=None,
        environment="test",
        database_url="postgresql+psycopg://user:pw@localhost:5433/pie",
    )


def test_the_service_default_matches_the_configured_default(default_settings: Settings) -> None:
    assert default_settings.extraction_window_max_chars == DEFAULT_WINDOW_MAX_CHARS


def test_the_budget_is_well_under_what_broke_extraction(default_settings: Settings) -> None:
    """A sanity floor and ceiling, not a tuning knob.

    The document that reproduced the defect was 60,241 characters. The budget
    must divide it into several windows, and must not be so small that the fixed
    prompt overhead dominates every call.
    """
    budget = default_settings.extraction_window_max_chars

    assert 2_000 <= budget <= 20_000
    assert 60_241 // budget >= 4, "the reproducing document must divide several ways"
