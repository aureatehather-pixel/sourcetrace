"""The DeepSeek adapter, exercised without a network.

Two responsibilities, the same two the Anthropic adapter has: turning a vendor
response into a provider-neutral `ProviderResult`, and translating transport and
status failures into our bounded error codes. The HTTP transport is replaced, so
nothing here needs credentials.

The tests that matter most are the ones about `finish_reason`. The whole
adaptive-extraction mechanism turns on being able to tell "this interval
demanded more than one bounded response could carry" from "this response is
unusable for some other reason", and getting that wrong in the cheap direction
means recursively re-paying for garbage.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest

from intelligence_engine.extraction.deepseek_provider import (
    PROVIDER_NAME,
    DeepSeekExtractionProvider,
)
from intelligence_engine.extraction.errors import ProviderTimeout, ProviderUnavailable
from intelligence_engine.extraction.provider import StopReason
from intelligence_engine.extraction.schemas import wire_schema

VALID_BODY = json.dumps({"items": []})


def _envelope(
    *,
    content: str | None = VALID_BODY,
    finish_reason: str | None = "stop",
    usage: dict[str, Any] | None = None,
    model: str = "deepseek-v4-flash",
) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": usage if usage is not None else {"prompt_tokens": 100, "completion_tokens": 50},
    }


def _provider(handler: Any, *, max_retries: int = 0, **kwargs: Any) -> DeepSeekExtractionProvider:
    """A provider whose transport is `handler`, capturing the requests it sees."""
    provider = DeepSeekExtractionProvider(
        api_key="test-key-not-real",
        model="deepseek-v4-flash",
        timeout_seconds=5.0,
        max_retries=max_retries,
        **kwargs,
    )
    # Swapping the transport is the point: the adapter is exercised whole,
    # request shaping included, without a network.
    provider._client = httpx2.AsyncClient(
        base_url="https://api.deepseek.test",
        transport=httpx2.MockTransport(handler),
        headers={"Authorization": "Bearer test-key-not-real"},
    )
    return provider


def _responder(*responses: httpx2.Response) -> Any:
    """Serve each response in turn, recording every request body."""
    seen: list[dict[str, Any]] = []
    remaining = list(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


async def _extract(provider: DeepSeekExtractionProvider) -> Any:
    try:
        return await provider.extract(
            system_prompt="SYSTEM",
            user_prompt="USER",
            json_schema=wire_schema(),
        )
    finally:
        await provider.aclose()


# --- the request shape -----------------------------------------------------


async def test_thinking_is_disabled_explicitly() -> None:
    """Extraction transcribes; it does not reason.

    DeepSeek defaults to thinking-enabled, and reasoning tokens are billed as
    output — the expensive half — for work whose every prompt rule is about not
    inferring. So the request must say so rather than relying on a default.
    """
    handler = _responder(httpx2.Response(200, json=_envelope()))
    await _extract(_provider(handler))

    assert handler.seen[0]["thinking"] == {"type": "disabled"}


async def test_the_model_and_ceiling_are_sent_explicitly() -> None:
    handler = _responder(httpx2.Response(200, json=_envelope()))
    await _extract(_provider(handler, max_tokens=8_000))

    assert handler.seen[0]["model"] == "deepseek-v4-flash"
    assert handler.seen[0]["max_tokens"] == 8_000


async def test_the_prompt_is_sent_as_system_and_user_messages_unchanged() -> None:
    """The prompt is M2's, not this adapter's. It must arrive verbatim.

    Any rewriting here would mean the two providers answer different questions,
    which would make the comparison between them meaningless — and would change
    what `EXTRACTOR_VERSION` refers to without the version gate noticing.
    """
    handler = _responder(httpx2.Response(200, json=_envelope()))
    await _extract(_provider(handler))

    assert handler.seen[0]["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
    ]


async def test_the_wire_schema_is_sent_as_a_json_schema_constraint() -> None:
    """The same schema the Anthropic path sends, in this vendor's envelope."""
    handler = _responder(httpx2.Response(200, json=_envelope()))
    await _extract(_provider(handler, use_json_schema=True))

    fmt = handler.seen[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"] == wire_schema()
    assert fmt["json_schema"]["strict"] is True


async def test_json_object_is_used_when_schema_mode_is_off() -> None:
    """The weaker constraint remains available, and changes nothing downstream.

    Validation on arrival is what actually enforces the contract; the response
    format only narrows the failure surface.
    """
    handler = _responder(httpx2.Response(200, json=_envelope()))
    await _extract(_provider(handler, use_json_schema=False))

    assert handler.seen[0]["response_format"] == {"type": "json_object"}


# --- the termination reason, which subdivision depends on ------------------


async def test_a_natural_stop_is_reported_as_complete() -> None:
    result = await _extract(_provider(_responder(httpx2.Response(200, json=_envelope()))))

    assert result.stop_reason is StopReason.COMPLETE
    assert result.raw_output == VALID_BODY
    assert result.provider == PROVIDER_NAME


async def test_hitting_the_output_ceiling_is_reported_as_max_output() -> None:
    """`length` is this vendor's truncation signal, and the only overflow one."""
    truncated = '{"items": [{"kind": "reported_claim", "data": {"claim": "unter'
    handler = _responder(
        httpx2.Response(200, json=_envelope(content=truncated, finish_reason="length"))
    )

    result = await _extract(_provider(handler))

    assert result.stop_reason is StopReason.MAX_OUTPUT
    assert result.raw_output == truncated


async def test_a_truncated_response_with_no_text_is_still_a_truncation() -> None:
    """The whole budget went somewhere that was not the answer.

    Reported as the truncation it is, so the interval subdivides rather than
    failing as "unavailable" — an empty body with `length` is evidence the
    interval was too large, which is exactly what overflow means.
    """
    handler = _responder(httpx2.Response(200, json=_envelope(content="", finish_reason="length")))

    result = await _extract(_provider(handler))

    assert result.stop_reason is StopReason.MAX_OUTPUT


@pytest.mark.parametrize("reason", ["tool_calls", "something_invented_later", None])
async def test_an_unmodelled_finish_reason_is_neither_a_crash_nor_a_truncation(
    reason: str | None,
) -> None:
    """A new vendor value must degrade to the safe reading.

    Mapping an unknown reason onto `MAX_OUTPUT` would subdivide responses that
    were never truncated; raising would take a run down over a vocabulary
    change. `OTHER` is handled exactly like a normal completion, so the output
    still faces the same strict parse and grounding check.
    """
    handler = _responder(httpx2.Response(200, json=_envelope(finish_reason=reason)))

    result = await _extract(_provider(handler))

    assert result.stop_reason is StopReason.OTHER


@pytest.mark.parametrize("reason", ["content_filter", "insufficient_system_resource"])
async def test_an_unusable_finish_reason_is_a_provider_failure_not_an_overflow(
    reason: str,
) -> None:
    """The expensive mistake this guards against.

    Both mean the response is unusable for reasons no smaller prompt can fix. If
    either reached the `StopReason` mapping and came out as `MAX_OUTPUT`, the
    service would subdivide and re-pay, exponentially, for a filter or a
    capacity problem.
    """
    handler = _responder(httpx2.Response(200, json=_envelope(finish_reason=reason)))

    with pytest.raises(ProviderUnavailable):
        await _extract(_provider(handler))


async def test_usage_tokens_are_carried_across() -> None:
    handler = _responder(
        httpx2.Response(
            200,
            json=_envelope(usage={"prompt_tokens": 7202, "completion_tokens": 2362}),
        )
    )

    result = await _extract(_provider(handler))

    assert (result.input_tokens, result.output_tokens) == (7202, 2362)


# --- failure translation ---------------------------------------------------


async def test_a_timeout_is_a_provider_timeout() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("too slow", request=request)

    with pytest.raises(ProviderTimeout):
        await _extract(_provider(handler))


async def test_a_connection_failure_is_a_provider_failure() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route", request=request)

    with pytest.raises(ProviderUnavailable):
        await _extract(_provider(handler))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_a_non_retryable_status_fails_immediately(status: int) -> None:
    """A 400 or a 401 fails identically on every attempt.

    Retrying one only spends time before reporting the same thing, so it must
    not consume the retry budget.
    """
    handler = _responder(httpx2.Response(status, json={"error": "nope"}))

    with pytest.raises(ProviderUnavailable):
        await _extract(_provider(handler, max_retries=2))

    assert len(handler.seen) == 1


async def test_the_error_never_carries_the_provider_response_body() -> None:
    """`error_message` is a bounded column that ends up in logs.

    A provider body can be long and can echo the evidence text back, so the
    message is authored here rather than taken from the response.
    """
    leaked = "the evidence said something private"
    handler = _responder(httpx2.Response(400, json={"error": {"message": leaked}}))

    with pytest.raises(ProviderUnavailable) as caught:
        await _extract(_provider(handler))

    assert leaked not in str(caught.value)
    assert "400" in str(caught.value)


async def test_a_rate_limit_is_retried_then_succeeds() -> None:
    """Parity with the Anthropic SDK, which retries 429 and 5xx for us.

    Without this the two providers would differ in reliability under load, and a
    comparison between them would measure the retry gap rather than the models.
    """
    handler = _responder(
        httpx2.Response(429, json={"error": "slow down"}),
        httpx2.Response(200, json=_envelope()),
    )

    result = await _extract(_provider(handler, max_retries=2))

    assert result.stop_reason is StopReason.COMPLETE
    assert len(handler.seen) == 2


async def test_retries_are_bounded_and_then_the_failure_is_raised() -> None:
    handler = _responder(httpx2.Response(503, json={"error": "down"}))

    with pytest.raises(ProviderUnavailable):
        await _extract(_provider(handler, max_retries=2))

    # The first attempt plus exactly `max_retries` more.
    assert len(handler.seen) == 3


async def test_invalid_output_is_never_retried() -> None:
    """A model that just returned garbage returns more garbage, at full price.

    The adapter hands the text up unvalidated; deciding it is garbage is the
    application's job, and it must reach that decision after exactly one call.
    """
    handler = _responder(httpx2.Response(200, json=_envelope(content="not json at all")))

    result = await _extract(_provider(handler, max_retries=2))

    assert result.raw_output == "not json at all"
    assert len(handler.seen) == 1


# --- envelope defensiveness ------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": "not-a-list"},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": "   "}}]},
    ],
)
async def test_a_malformed_envelope_is_a_bounded_provider_error(body: dict[str, Any]) -> None:
    """This is the trust boundary for a vendor's HTTP shape.

    A missing `choices` must surface as a provider error with an error code the
    run can store, never as a `KeyError` escaping from inside the service.
    """
    handler = _responder(httpx2.Response(200, json=body))

    with pytest.raises(ProviderUnavailable):
        await _extract(_provider(handler))


async def test_a_non_json_envelope_is_a_bounded_provider_error() -> None:
    handler = _responder(httpx2.Response(200, content=b"<html>gateway error</html>"))

    with pytest.raises(ProviderUnavailable):
        await _extract(_provider(handler))
