"""Configuration fails fast and loudly when it is wrong."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intelligence_engine.core.config import Settings

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


def test_valid_settings_load() -> None:
    settings = _settings()
    assert settings.database_url_str == VALID_URL
    assert settings.is_production is False


def test_missing_database_url_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://user:pw@localhost:5433/pie",  # sync driver
        "postgresql+asyncpg://user:pw@localhost:5433/pie",  # wrong driver
        "sqlite+aiosqlite:///./pie.db",  # not PostgreSQL at all
        "not-a-url",
    ],
)
def test_wrong_database_url_is_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        _settings(database_url=url)


def test_unknown_environment_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(environment="staging")


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(log_level="TRACE")


def test_settings_are_immutable() -> None:
    settings = _settings()
    with pytest.raises(ValidationError):
        settings.environment = "production"  # type: ignore[misc]


# --- provider selection ----------------------------------------------------
#
# The vendor is named, never inferred from whichever key happens to be set.
# With two keys configured, inference would silently pick one, and "which model
# produced this run" is exactly what this system refuses to leave implicit.


def test_anthropic_is_the_default_provider() -> None:
    """Adding a second vendor must not move anyone off the first one."""
    assert _settings().extraction_provider == "anthropic"


def test_an_unknown_provider_is_rejected() -> None:
    """A typo in the environment fails at startup, not at the first extraction."""
    with pytest.raises(ValidationError):
        _settings(extraction_provider="deepsek")


@pytest.mark.parametrize(
    ("provider", "key_field", "expected"),
    [
        ("anthropic", "anthropic_api_key", "anthropic-key"),
        ("deepseek", "deepseek_api_key", "deepseek-key"),
    ],
)
def test_the_selected_provider_decides_which_key_is_read(
    provider: str, key_field: str, expected: str
) -> None:
    settings = _settings(
        extraction_provider=provider,
        anthropic_api_key="anthropic-key",
        deepseek_api_key="deepseek-key",
    )

    assert settings.extraction_api_key is not None
    assert settings.extraction_api_key.get_secret_value() == expected
    assert settings.has_extraction_provider is True


def test_holding_the_other_vendors_key_is_not_a_configured_provider() -> None:
    """The failure this guards against is a quiet fallback to the fake.

    An Anthropic key with `EXTRACTION_PROVIDER=deepseek` is a misconfiguration,
    not a configuration. Reading "some key is present" would let it start up and
    extract with the deterministic fake.
    """
    settings = _settings(extraction_provider="deepseek", anthropic_api_key="anthropic-key")

    assert settings.extraction_api_key is None
    assert settings.has_extraction_provider is False


def test_production_requires_the_selected_providers_key() -> None:
    with pytest.raises(ValidationError, match="DEEPSEEK_API_KEY"):
        _settings(
            environment="production",
            extraction_provider="deepseek",
            anthropic_api_key="anthropic-key",
        )


def test_production_starts_when_the_selected_provider_is_configured() -> None:
    settings = _settings(
        environment="production",
        extraction_provider="deepseek",
        deepseek_api_key="deepseek-key",
    )

    assert settings.is_production is True
    assert settings.has_extraction_provider is True


def test_the_deepseek_model_default_is_v4_flash() -> None:
    """The cheap routine model, named rather than mapped.

    DeepSeek's Anthropic-compatible endpoint would select this by translating a
    `claude-sonnet*` string; naming it directly is what makes the run row say
    what was actually billed.
    """
    assert _settings().deepseek_extraction_model == "deepseek-v4-flash"


def test_a_secret_key_is_not_printed_by_a_repr() -> None:
    """Both keys are SecretStr, so an accidental log of settings leaks neither."""
    settings = _settings(anthropic_api_key="anthropic-key", deepseek_api_key="deepseek-key")

    rendered = repr(settings)

    assert "anthropic-key" not in rendered
    assert "deepseek-key" not in rendered
