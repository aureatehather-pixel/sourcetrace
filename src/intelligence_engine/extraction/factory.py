"""Chooses the extraction provider from configuration.

One function, one `match`. Not a registry, not a plugin system, not a router:
there are two real implementations plus a fake, the choice is a config value,
and nothing here falls back from one vendor to another. A provider that fails
fails - swapping vendors mid-run would mean a source extracted half by one model
and half by another, with nothing on the run rows saying so.

When the selected provider has no API key the deterministic fake is used, so the
whole extraction path is exercisable locally without credentials. `Settings`
refuses that combination in production, so this cannot become a silent fake in a
real deployment.
"""

from __future__ import annotations

import logging

from intelligence_engine.core.config import Settings
from intelligence_engine.extraction.anthropic_provider import AnthropicExtractionProvider
from intelligence_engine.extraction.deepseek_provider import DeepSeekExtractionProvider
from intelligence_engine.extraction.fake_provider import FakeExtractionProvider
from intelligence_engine.extraction.provider import ExtractionProvider

logger = logging.getLogger(__name__)


def build_provider(settings: Settings) -> ExtractionProvider:
    key = settings.extraction_api_key
    if key is None:
        logger.warning(
            "extraction.provider.fake_selected",
            extra={
                "environment": settings.environment,
                "extraction_provider": settings.extraction_provider,
            },
        )
        return FakeExtractionProvider()

    if settings.extraction_provider == "deepseek":
        return DeepSeekExtractionProvider(
            api_key=key.get_secret_value(),
            model=settings.deepseek_extraction_model,
            timeout_seconds=settings.extraction_timeout_seconds,
            max_retries=settings.extraction_max_retries,
            max_tokens=settings.extraction_max_output_tokens,
            base_url=settings.deepseek_base_url,
            use_json_schema=settings.deepseek_use_json_schema,
        )

    return AnthropicExtractionProvider(
        api_key=key.get_secret_value(),
        model=settings.extraction_model,
        timeout_seconds=settings.extraction_timeout_seconds,
        max_retries=settings.extraction_max_retries,
        max_tokens=settings.extraction_max_output_tokens,
    )
