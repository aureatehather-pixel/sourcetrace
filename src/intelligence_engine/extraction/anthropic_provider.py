"""Anthropic implementation of the extraction provider boundary.

The only module in the project that imports `anthropic`. It owns request
shaping, timeouts, retry policy and exception translation; it deliberately does
not parse or validate the model's JSON — that happens in the application layer
where malformed output is tested against.

Retries: the SDK already retries connection errors, 408/409/429 and 5xx with
exponential backoff, bounded by `max_retries`. That is the whole retry policy.
No loop of our own, no retry on invalid output — re-asking a model that just
returned garbage usually returns more garbage, at full price.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from intelligence_engine.extraction.errors import ProviderTimeout, ProviderUnavailable
from intelligence_engine.extraction.provider import (
    ProviderIdentity,
    ProviderResult,
    StopReason,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "anthropic"

#: Anthropic's `stop_reason` vocabulary, narrowed to what the application acts
#: on. Only `max_tokens` changes behaviour — it is the truncation signal that
#: lets an overflowing interval be told apart from a model that emitted
#: garbage. `refusal` never reaches here: it is raised as `ProviderUnavailable`
#: before the mapping runs, exactly as it always was. Anything unlisted,
#: including `None`, maps to `OTHER` and is handled as a normal completion, so
#: a new SDK value is never a crash and never a silent truncation.
_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": StopReason.COMPLETE,
    "stop_sequence": StopReason.COMPLETE,
    "max_tokens": StopReason.MAX_OUTPUT,
}


def _stop_reason(raw: str | None) -> StopReason:
    """Translate the SDK's reason into the application's three-value enum."""
    if raw is None:
        return StopReason.OTHER
    return _STOP_REASONS.get(raw, StopReason.OTHER)


class AnthropicExtractionProvider:
    """Calls the Messages API with a constrained JSON output schema."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        max_tokens: int = 8_000,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self._model = model
        self._identity = ProviderIdentity(provider=PROVIDER_NAME, model=model)
        self._max_tokens = max_tokens

    @property
    def identity(self) -> ProviderIdentity:
        """The model this client was configured with, not the one a reply names.

        `ProviderResult.model` carries `response.model`, which is what actually
        answered and is the right thing to record about a finished call. This is
        the requested identity, and it is what reuse compares — see
        `ProviderIdentity`.
        """
        return self._identity

    async def extract(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> ProviderResult:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                # Constrains the response to our schema. It narrows the failure
                # surface; it does not replace validation, because the run must
                # still fail safely if the guarantee ever does not hold.
                output_config={"format": {"type": "json_schema", "schema": json_schema}},
            )
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeout("Extraction provider timed out") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderUnavailable("Could not reach the extraction provider") from exc
        except anthropic.AuthenticationError as exc:
            raise ProviderUnavailable("Extraction provider rejected the credentials") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderUnavailable("Extraction provider rate limit exceeded") from exc
        except anthropic.APIStatusError as exc:
            # Status and request id are useful operationally and leak nothing.
            logger.warning(
                "extraction.provider.api_error",
                extra={"status_code": exc.status_code, "provider": PROVIDER_NAME},
            )
            raise ProviderUnavailable(
                f"Extraction provider returned status {exc.status_code}"
            ) from exc

        if response.stop_reason == "refusal":
            raise ProviderUnavailable("Extraction provider declined the request")

        # Narrowed on the discriminator: thinking and tool blocks have no .text.
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text.strip():
            raise ProviderUnavailable("Extraction provider returned an empty response")

        return ProviderResult(
            raw_output=text,
            provider=PROVIDER_NAME,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            # Carried, not interpreted. Deciding what a truncated response
            # means is the application's job; this boundary only reports it in
            # terms that do not name a vendor.
            stop_reason=_stop_reason(response.stop_reason),
        )
