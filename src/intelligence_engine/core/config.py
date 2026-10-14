"""Typed application configuration.

Configuration is read from the process environment (optionally seeded from a
local ``.env``). Missing or malformed required values raise at import/startup
time rather than surfacing as a confusing runtime failure later.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
#: The extraction vendors M2 knows how to talk to. A closed set, so a typo in
#: the environment fails at startup rather than silently falling back.
ExtractionProvider = Literal["anthropic", "deepseek"]


class Settings(BaseSettings):
    """Application settings. Instantiating with invalid input raises ValidationError."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Environment = "local"
    database_url: PostgresDsn = Field(
        ...,
        description="Async SQLAlchemy URL, e.g. postgresql+psycopg://user:pass@host:5432/db",
    )
    log_level: LogLevel = "INFO"

    # Bounded pagination for the evidence list endpoint.
    default_page_size: int = Field(default=50, ge=1, le=200)
    max_page_size: int = Field(default=200, ge=1, le=500)

    # --- Extraction (M2) ---------------------------------------------------
    # Which vendor routine extraction goes to. Named explicitly rather than
    # inferred from whichever key happens to be set: with two keys configured,
    # inference would silently pick one, and "which model produced this run" is
    # exactly the kind of thing this system refuses to leave implicit.
    #
    # `anthropic` stays the default. Selecting `deepseek` changes the provider
    # and nothing else - same prompt, same schema, same strict parse, same
    # 50-item bound, same verbatim grounding check.
    extraction_provider: ExtractionProvider = "anthropic"

    # SecretStr so the key cannot be printed by an accidental repr/log of
    # settings. Unset means the deterministic fake provider is used, which is
    # fine locally and refused in production (see _require_provider_in_prod).
    anthropic_api_key: SecretStr | None = None
    extraction_model: str = Field(default="claude-opus-5", min_length=1, max_length=128)

    # --- Extraction via DeepSeek -------------------------------------------
    # A routine-cost alternative for the same work. V4 Flash is roughly a
    # twentieth of Claude Opus 5 per token, and M2's job is transcription
    # against a fixed schema rather than reasoning, so the question is whether
    # it transcribes as faithfully - which is measured, not assumed.
    deepseek_api_key: SecretStr | None = None
    deepseek_extraction_model: str = Field(
        default="deepseek-v4-flash", min_length=1, max_length=128
    )
    deepseek_base_url: str = Field(default="https://api.deepseek.com", min_length=1)
    # Whether to send the wire schema as a `json_schema` response format rather
    # than plain `json_object`. Configurable because DeepSeek's documented
    # response formats and what its endpoint actually accepts for a schema this
    # shape are two different questions, and the answer is a measurement. Either
    # way the output faces the same validation on arrival.
    deepseek_use_json_schema: bool = True
    extraction_timeout_seconds: float = Field(default=90.0, gt=0, le=600)
    # The entire retry policy: the SDK's bounded backoff on transient failures.
    # No loop of our own, and never a retry on invalid output.
    extraction_max_retries: int = Field(default=2, ge=0, le=5)
    extraction_max_output_tokens: int = Field(default=8_000, ge=1_000, le=64_000)

    # How much source text one extraction window may carry. Sized against
    # *output*, which is what actually failed: a 60,241-char export produced
    # `stop_reason: max_tokens` at exactly the 8,000-token ceiling, truncated
    # mid-item, having already emitted 69 items — so raising that ceiling would
    # only have surrendered the run to `MAX_ITEMS_PER_RUN = 50` instead.
    #
    # 6,000 characters, set from a measurement rather than an estimate. The
    # first value tried here was 12,000, reasoned from source density, and the
    # real corpus falsified it: window 0 of Event 2 returned 39 items from
    # 11,375 characters — 78% of the 50-item invariant, no useful headroom — and
    # window 1 exceeded 50 outright and was rejected. Observed density is
    # therefore ~3.4 items per 1,000 characters, rising above 4.3 on dense
    # passages, not the ~1.7 the estimate assumed.
    #
    # At 6,000 characters the densest observed passage yields roughly 26 items,
    # about half the cap, for roughly 3,000 output tokens against a ceiling of
    # 8,000. Event 2 divides into about eleven windows.
    #
    # The cost is named: the ~3,900-token fixed prompt and schema overhead is
    # paid once per window, so halving the window roughly doubles that overhead
    # for a document. Bounded input per call and headroom under a hard
    # invariant are worth more than the saving, because exceeding the invariant
    # loses the entire window rather than degrading.
    #
    # Both guards stay where they are. This budget exists to keep runs away
    # from them; they remain the loud per-window failure if it ever does not.
    extraction_window_max_chars: int = Field(default=6_000, ge=1_000, le=200_000)

    # --- Cross-evidence intelligence (M4) ----------------------------------
    # A separate model from extraction, on purpose. Relation classification
    # reads a few short statements under a flat schema and picks one of four
    # labels; extraction reads a whole evidence item under an eight-variant
    # schema.
    #
    # Chosen by running `evaluation/run_classifier_eval.py` over the 30-pair
    # hand-labelled fixture, not by inheriting the extraction default and not by
    # assuming bigger is better. The cheaper candidate was tried first and
    # failed, twice:
    #
    #   claude-haiku-4-5  run 1: 26/30, MISSED the flagship qualification,
    #                            1 false contradiction, 1 false relation
    #                     run 2: 19/30, flagship wrong again (called it
    #                            `reinforces`), 1 false contradiction,
    #                            1 false relation, and one whole batch rejected
    #                            for asserting `contradicts` with no explanation
    #   claude-sonnet-5   29/30, 0 false contradictions, 0 false relations,
    #                            0 missed relations, 0 malformed batches
    #
    # Both Haiku runs got the one pair this milestone exists to find wrong, and
    # both produced a false contradiction — the most expensive routine error. So
    # this is the cheapest model *demonstrated adequate*, which is the rule;
    # `claude-opus-5` was deliberately not tried, because Sonnet already has zero
    # failures in every category that matters and there is nothing left for a
    # 2.5x more expensive model to measurably improve on this fixture.
    #
    # Caveat worth carrying: Sonnet 5 runs adaptive thinking by default and Haiku
    # 4.5 does not, so part of what this comparison measured is configuration
    # rather than raw capability. n is small by design.
    relation_model: str = Field(default="claude-sonnet-5", min_length=1, max_length=128)
    relation_timeout_seconds: float = Field(default=90.0, gt=0, le=600)
    relation_max_retries: int = Field(default=2, ge=0, le=5)
    relation_max_output_tokens: int = Field(default=4_000, ge=500, le=32_000)
    # The two hard cost bounds. `relation_max_candidate_pairs` caps what one
    # analysis can cost regardless of how much evidence an entity accumulates;
    # `relation_batch_size` keeps prompts bounded in the other direction, since
    # an enormous batch trades a real quality risk for a small saving.
    relation_max_candidate_pairs: int = Field(default=60, ge=1, le=500)
    relation_batch_size: int = Field(default=8, ge=1, le=20)

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, value: PostgresDsn) -> PostgresDsn:
        """Reject sync URLs early: an async engine on a sync driver fails obscurely."""
        if value.scheme != "postgresql+psycopg":
            raise ValueError(
                "database_url must use the 'postgresql+psycopg' scheme "
                f"(psycopg 3, async-capable); got {value.scheme!r}"
            )
        return value

    @model_validator(mode="after")
    def _require_provider_in_prod(self) -> Settings:
        """Production must never silently serve fake extractions.

        Checks the key for the *selected* provider. Holding an Anthropic key
        while `EXTRACTION_PROVIDER=deepseek` is not a configured provider; it is
        a misconfiguration that would otherwise start up and quietly extract
        with the fake.
        """
        if self.environment == "production" and not self.has_extraction_provider:
            raise ValueError(
                f"{self.extraction_provider.upper()}_API_KEY is required when "
                f"ENVIRONMENT=production and EXTRACTION_PROVIDER={self.extraction_provider}; "
                "the fake extraction provider must never run in production"
            )
        return self

    @property
    def extraction_api_key(self) -> SecretStr | None:
        """The key for the selected provider, or None if it is not configured."""
        if self.extraction_provider == "deepseek":
            return self.deepseek_api_key
        return self.anthropic_api_key

    @property
    def has_extraction_provider(self) -> bool:
        return self.extraction_api_key is not None

    @property
    def database_url_str(self) -> str:
        return str(self.database_url)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Cached so validation happens exactly once."""
    return Settings()  # values are supplied by the environment / .env
