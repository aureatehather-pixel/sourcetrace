"""Which provider the configuration actually builds.

One function with one `match`, so there is little to test — but the two things
worth pinning are the two that fail silently. A provider selected by accident
extracts a whole corpus with the wrong model and nothing on the run rows says
so; a fake selected by accident stores confident, deterministic fiction.
"""

from __future__ import annotations

import pytest

from intelligence_engine.core.config import Settings
from intelligence_engine.extraction.anthropic_provider import AnthropicExtractionProvider
from intelligence_engine.extraction.deepseek_provider import DeepSeekExtractionProvider
from intelligence_engine.extraction.factory import build_provider
from intelligence_engine.extraction.fake_provider import FakeExtractionProvider

VALID_URL = "postgresql+psycopg://user:pw@localhost:5433/pie"


def _settings(**overrides: object) -> Settings:
    """Settings built from the arguments alone.

    `_env_file=None` so a developer's real `.env` cannot leak a key into a test
    that is asserting one is absent — which would make the test pass or fail
    depending on whose machine it ran on.
    """
    values: dict[str, object] = {"environment": "test", "database_url": VALID_URL}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_anthropic_is_built_by_default() -> None:
    provider = build_provider(_settings(anthropic_api_key="key"))

    assert isinstance(provider, AnthropicExtractionProvider)


def test_deepseek_is_built_when_selected() -> None:
    provider = build_provider(_settings(extraction_provider="deepseek", deepseek_api_key="key"))

    assert isinstance(provider, DeepSeekExtractionProvider)


def test_selecting_deepseek_never_falls_back_to_anthropic() -> None:
    """No fallback chain, deliberately.

    Swapping vendors when one is unconfigured would mean a source extracted
    partly by one model and partly by another, with nothing on the run rows
    recording the switch. The fake is the honest outcome: it is loud, it is
    refused in production, and it cannot be mistaken for a real extraction.
    """
    provider = build_provider(
        _settings(extraction_provider="deepseek", anthropic_api_key="anthropic-key")
    )

    assert isinstance(provider, FakeExtractionProvider)


@pytest.mark.parametrize("provider_name", ["anthropic", "deepseek"])
def test_no_key_for_the_selected_provider_builds_the_fake(provider_name: str) -> None:
    provider = build_provider(_settings(extraction_provider=provider_name))

    assert isinstance(provider, FakeExtractionProvider)


def test_the_deepseek_provider_is_configured_from_settings() -> None:
    """The configured model and ceiling reach the adapter, not defaults."""
    provider = build_provider(
        _settings(
            extraction_provider="deepseek",
            deepseek_api_key="key",
            deepseek_extraction_model="deepseek-v4-pro",
            extraction_max_output_tokens=4_000,
        )
    )

    assert isinstance(provider, DeepSeekExtractionProvider)
    assert provider._model == "deepseek-v4-pro"
    assert provider._max_tokens == 4_000
