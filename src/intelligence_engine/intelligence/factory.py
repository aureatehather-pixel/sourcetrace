"""Chooses the relation-classification provider from configuration.

Deliberately reuses M2's provider boundary rather than cloning it. That boundary
takes a system prompt, a user prompt and a JSON schema and returns raw text plus
token counts — it is model-agnostic and task-agnostic, and the Anthropic
implementation is already parameterised by model, timeout, retries and output
cap. A second near-identical module would be duplication, and a second Protocol
with the same one method would be ceremony.

What is *not* shared is the model. Extraction reads a whole evidence item under a
large eight-variant schema; relation classification reads a handful of short
statements under a flat schema and picks one of four labels. The second is a much
smaller job, and `RELATION_MODEL` is configured separately so it can be — and is
— a cheaper model, chosen by evaluation rather than by inheriting whatever
extraction happened to use.

The one wart worth naming: the Protocol is called `ExtractionProvider`, which now
under-describes what it does. Renaming it is a mechanical change across M2 and its
tests, and it buys nothing at this size, so it is recorded rather than done.
"""

from __future__ import annotations

import logging

from intelligence_engine.core.config import Settings
from intelligence_engine.extraction.anthropic_provider import AnthropicExtractionProvider
from intelligence_engine.extraction.fake_provider import FakeExtractionProvider
from intelligence_engine.extraction.provider import ExtractionProvider

logger = logging.getLogger(__name__)


def build_relation_provider(settings: Settings) -> ExtractionProvider:
    """The provider used for relation classification.

    As with extraction, no API key means the deterministic fake, so the whole
    path is exercisable locally without credentials. `Settings` refuses that
    combination in production.
    """
    if settings.anthropic_api_key is None:
        logger.warning(
            "intelligence.provider.fake_selected",
            extra={"environment": settings.environment},
        )
        return FakeExtractionProvider()

    return AnthropicExtractionProvider(
        api_key=settings.anthropic_api_key.get_secret_value(),
        model=settings.relation_model,
        timeout_seconds=settings.relation_timeout_seconds,
        max_retries=settings.relation_max_retries,
        max_tokens=settings.relation_max_output_tokens,
    )
