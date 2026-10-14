"""DeepSeek implementation of the extraction provider boundary.

The second implementation of `ExtractionProvider`, and it exists for one reason:
DeepSeek V4 Flash costs roughly a twentieth of Claude Opus 5 per token, and M2's
routine work is transcription against a fixed schema rather than reasoning. If
it extracts as faithfully, routine extraction should not be paid for at
frontier-model rates.

**Nothing about M2 changes to accommodate it.** The protocol is unchanged, the
prompt is unchanged, `ProviderResult` and `StopReason` are unchanged, and the
output faces exactly the same strict parse, the same 50-item bound and the same
verbatim grounding check. This module only translates one vendor's HTTP shape
into that contract, and it is the only module in the project that knows DeepSeek
exists.

## Why the native API and not DeepSeek's Anthropic-compatible endpoint

DeepSeek publishes an Anthropic-shaped endpoint, and reusing
`AnthropicExtractionProvider` against it with a `base_url` override would have
been a smaller diff. It was rejected on the documented behaviour of that
compatibility layer, not on taste:

- **It ignores the schema.** Of `output_config` it supports only `effort`. Our
  extraction sends `output_config.format.json_schema`, so the constraint would
  be accepted and silently dropped, and the model would be left to guess the
  shape from a prompt that says "matching the provided schema" while no schema
  was provided.
- **It selects the model by name mapping.** `claude-opus*` maps to
  `deepseek-v4-pro`, `claude-sonnet*`/`claude-haiku*` to `deepseek-v4-flash`.
  Asking for V4 Flash would mean sending a `claude-sonnet` string and trusting a
  mapping table, rather than naming the model we intend to be billed for.
- **Thinking is on by default there and `budget_tokens` is ignored**, so
  non-thinking mode is not reliably selectable.

The native endpoint gives all three explicitly, so it is the honest integration
even though it is a slightly larger file.

## Non-thinking mode, deliberately

`thinking: {"type": "disabled"}`. Extraction is transcription: every rule in the
prompt is about *not* inferring - not upgrading a reported claim into a fact,
not inventing biography, not inferring causation. Reasoning tokens buy nothing
against that and are billed as output, which is the expensive half. Thinking is
on by default at DeepSeek, so this must be sent explicitly.

## Retries

The Anthropic SDK retries connection errors, 429 and 5xx with bounded
exponential backoff. `httpx2` has no equivalent, so the same policy is written
out here - and only that policy. Nothing retries on invalid output: a model that
just returned garbage returns more garbage, at full price. Without this the two
providers would have materially different reliability under load, and a
comparison between them would be measuring the retry gap rather than the models.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx2

from intelligence_engine.extraction.errors import ProviderTimeout, ProviderUnavailable
from intelligence_engine.extraction.provider import (
    ProviderIdentity,
    ProviderResult,
    StopReason,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "deepseek"

DEFAULT_BASE_URL = "https://api.deepseek.com"
_COMPLETIONS_PATH = "/chat/completions"

#: DeepSeek's `finish_reason` vocabulary, narrowed to what the application acts
#: on. Only `length` changes behaviour - it is the truncation signal that lets
#: an overflowing interval be told apart from a model that emitted garbage.
#:
#: `content_filter` and `insufficient_system_resource` are deliberately absent:
#: they are raised as `ProviderUnavailable` before this mapping runs. Neither is
#: an overflow — a filtered or resource-starved response is not evidence that
#: the interval was too large, and mapping either onto `MAX_OUTPUT` would make
#: the service subdivide, and pay repeatedly, for something no smaller prompt
#: can fix. Anything unlisted maps to `OTHER` and is handled as a normal
#: completion, so a new vendor value is never a crash and never a silent
#: truncation.
_FINISH_REASONS: dict[str, StopReason] = {
    "stop": StopReason.COMPLETE,
    "length": StopReason.MAX_OUTPUT,
}

#: Finish reasons that mean the response is unusable for reasons unrelated to
#: how much text was sent. Raised, never mapped.
_UNUSABLE_FINISH_REASONS = {
    "content_filter": "Extraction provider filtered the response",
    "insufficient_system_resource": "Extraction provider ran out of capacity mid-response",
}

#: Status codes worth trying again: rate limiting and server-side faults. A 400
#: or a 401 will fail identically on every attempt, so retrying one only spends
#: time before reporting the same thing.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _finish_reason(raw: str | None) -> StopReason:
    """Translate DeepSeek's reason into the application's three-value enum."""
    if raw is None:
        return StopReason.OTHER
    return _FINISH_REASONS.get(raw, StopReason.OTHER)


class DeepSeekExtractionProvider:
    """Calls DeepSeek's Chat Completions API with a constrained JSON output."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        max_tokens: int = 8_000,
        base_url: str = DEFAULT_BASE_URL,
        use_json_schema: bool = True,
    ) -> None:
        self._client = httpx2.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
            headers={
                # The key is set once, on the client, so it cannot be logged by
                # a request-shaping helper that dumps its own arguments.
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self._model = model
        self._identity = ProviderIdentity(provider=PROVIDER_NAME, model=model)
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._use_json_schema = use_json_schema

    @property
    def identity(self) -> ProviderIdentity:
        """The configured model, never the one the response reports.

        This client already tolerates a response that omits `model` by falling
        back to `self._model`, so the reported value is not reliably the vendor's
        own word for what ran. The requested identity has no such ambiguity.
        """
        return self._identity

    async def extract(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> ProviderResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Extraction transcribes; it does not reason. Sent explicitly
            # because DeepSeek's default is thinking-enabled.
            "thinking": {"type": "disabled"},
            "response_format": self._response_format(json_schema),
            # Determinism is not on offer from any of these models, but nothing
            # here wants variety either.
            "temperature": 0.0,
        }

        data = await self._post(payload)
        return self._to_result(data)

    async def aclose(self) -> None:
        """Release the connection pool. Safe to call more than once."""
        await self._client.aclose()

    # -- internals ---------------------------------------------------------

    def _response_format(self, json_schema: dict[str, Any]) -> dict[str, Any]:
        """The strongest output constraint this endpoint will accept.

        `json_schema` constrains the response to the same schema the Anthropic
        path sends, which is what keeps the two providers answering the same
        question. `json_object` guarantees only that the response parses as
        JSON, leaving the shape to the prompt.

        Both are a hint, never a guarantee: output is validated on arrival by
        `validation.parse_output` either way, and the item bound and grounding
        check are applied identically. The constraint narrows the failure
        surface; it does not replace the trust boundary.
        """
        if self._use_json_schema:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_output",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        return {"type": "json_object"}

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One request, with the bounded backoff the Anthropic SDK gives us free.

        `max_retries` attempts *after* the first, matching the SDK's meaning of
        the setting, so the same configured number means the same thing whichever
        provider is selected.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(_COMPLETIONS_PATH, json=payload)
            except httpx2.TimeoutException as exc:
                last_error = ProviderTimeout("Extraction provider timed out")
                exc_to_raise: Exception = last_error
                exc_to_raise.__cause__ = exc
            except httpx2.RequestError as exc:
                last_error = ProviderUnavailable("Could not reach the extraction provider")
                last_error.__cause__ = exc
            else:
                if response.status_code < 400:
                    return self._decode(response)
                last_error = self._status_error(response)
                if response.status_code not in _RETRYABLE_STATUS:
                    raise last_error

            if attempt < self._max_retries:
                await asyncio.sleep(_backoff_seconds(attempt))

        assert last_error is not None  # noqa: S101 - the loop always sets it
        raise last_error

    def _status_error(self, response: httpx2.Response) -> ProviderUnavailable:
        """Translate a non-2xx into a bounded, storable error.

        The provider's response body is never included. It can be long, it can
        echo the evidence text back, and `error_message` is a 500-character
        database column that ends up in logs.
        """
        status = response.status_code
        logger.warning(
            "extraction.provider.api_error",
            extra={"status_code": status, "provider": PROVIDER_NAME},
        )
        if status in {401, 403}:
            return ProviderUnavailable("Extraction provider rejected the credentials")
        if status == 429:
            return ProviderUnavailable("Extraction provider rate limit exceeded")
        return ProviderUnavailable(f"Extraction provider returned status {status}")

    def _decode(self, response: httpx2.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderUnavailable("Extraction provider returned a non-JSON envelope") from exc
        if not isinstance(data, dict):
            raise ProviderUnavailable("Extraction provider returned an unexpected envelope")
        return data

    def _to_result(self, data: dict[str, Any]) -> ProviderResult:
        """Turn one response envelope into a `ProviderResult`.

        Everything is read defensively. This is the trust boundary for a vendor's
        HTTP shape, and a missing `choices` list must be a bounded provider error
        rather than a `KeyError` surfacing from inside the service.
        """
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderUnavailable("Extraction provider returned no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ProviderUnavailable("Extraction provider returned an unexpected choice")

        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason in _UNUSABLE_FINISH_REASONS:
            # Neither is a truncation. Raised before the mapping so that
            # subdivision can never be triggered by one.
            raise ProviderUnavailable(_UNUSABLE_FINISH_REASONS[finish_reason])

        message = choice.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            # An empty body with `length` means the whole budget went to
            # something that is not the answer. Reported as the truncation it is,
            # so the interval subdivides rather than failing as "unavailable".
            if _finish_reason(finish_reason) is StopReason.MAX_OUTPUT:
                text = ""
            else:
                raise ProviderUnavailable("Extraction provider returned an empty response")

        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}

        return ProviderResult(
            raw_output=text,
            provider=PROVIDER_NAME,
            model=str(data.get("model") or self._model),
            input_tokens=_int_or_none(usage.get("prompt_tokens")),
            output_tokens=_int_or_none(usage.get("completion_tokens")),
            # Carried, not interpreted. Deciding what a truncated response means
            # is the application's job; this boundary only reports it in terms
            # that do not name a vendor.
            stop_reason=_finish_reason(finish_reason if isinstance(finish_reason, str) else None),
        )


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped. Mirrors the Anthropic SDK's shape."""
    jitter: float = random.uniform(0, 0.25)  # noqa: S311 - pacing, not cryptography
    # `2.0` rather than `2`: an int base makes `**` return `Any` under typeshed,
    # which would quietly widen this function's return type.
    return min(0.5 * (2.0**attempt) + jitter, 8.0)
