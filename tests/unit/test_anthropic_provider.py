"""The Anthropic adapter, exercised without a network.

Two responsibilities are tested: turning an SDK response into a provider-neutral
`ProviderResult`, and translating SDK exceptions into our bounded error codes.
The client's `messages.create` is replaced, so nothing here needs credentials.

What these tests cannot prove is that the real API accepts our request shape —
that needs credentials, and is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic
import httpx2
import pytest

from intelligence_engine.extraction.anthropic_provider import (
    PROVIDER_NAME,
    AnthropicExtractionProvider,
)
from intelligence_engine.extraction.errors import ProviderTimeout, ProviderUnavailable
from intelligence_engine.extraction.provider import ProviderResult, StopReason
from intelligence_engine.extraction.schemas import wire_schema


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Response:
    content: list[Any]
    usage: _Usage
    model: str = "claude-opus-5"
    stop_reason: str | None = "end_turn"


def _provider(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> AnthropicExtractionProvider:
    """Build a provider whose `messages.create` returns or raises `outcome`."""
    provider = AnthropicExtractionProvider(
        api_key="test-key-not-real",
        model="claude-opus-5",
        timeout_seconds=5.0,
        max_retries=0,
    )

    async def fake_create(**kwargs: Any) -> Any:
        fake_create.calls.append(kwargs)  # type: ignore[attr-defined]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    fake_create.calls = []  # type: ignore[attr-defined]
    monkeypatch.setattr(provider._client.messages, "create", fake_create)
    return provider


async def _extract(provider: AnthropicExtractionProvider) -> Any:
    return await provider.extract(
        system_prompt="SYSTEM",
        user_prompt="USER",
        json_schema=wire_schema(),
    )


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _response(status_code: int) -> httpx2.Response:
    return httpx2.Response(status_code, request=_request())


# --- response mapping -----------------------------------------------------


async def test_maps_a_successful_response_to_a_provider_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(
        monkeypatch,
        _Response(content=[_Block('{"items": []}')], usage=_Usage(1200, 340)),
    )

    result = await _extract(provider)

    assert result.raw_output == '{"items": []}'
    assert result.provider == PROVIDER_NAME
    assert result.model == "claude-opus-5"
    assert result.input_tokens == 1200
    assert result.output_tokens == 340


async def test_captures_token_usage_for_cost_awareness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch, _Response(content=[_Block("{}")], usage=_Usage(9999, 111)))

    result = await _extract(provider)

    assert (result.input_tokens, result.output_tokens) == (9999, 111)


async def test_concatenates_multiple_text_blocks_and_ignores_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(
        monkeypatch,
        _Response(
            content=[
                _Block("ignored", type="thinking"),
                _Block('{"items":'),
                _Block(" []}"),
            ],
            usage=_Usage(10, 10),
        ),
    )

    assert (await _extract(provider)).raw_output == '{"items": []}'


async def test_sends_the_schema_and_prompts_it_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch, _Response(content=[_Block("{}")], usage=_Usage(1, 1)))

    await _extract(provider)

    sent = provider._client.messages.create.calls[0]  # type: ignore[attr-defined]
    assert sent["system"] == "SYSTEM"
    assert sent["messages"] == [{"role": "user", "content": "USER"}]
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["format"]["schema"] == wire_schema()
    assert sent["model"] == "claude-opus-5"


# --- failure translation --------------------------------------------------


async def test_timeout_maps_to_provider_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider(monkeypatch, anthropic.APITimeoutError(request=_request()))

    with pytest.raises(ProviderTimeout) as caught:
        await _extract(provider)
    assert caught.value.error_code == "provider_timeout"


async def test_connection_error_maps_to_provider_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch, anthropic.APIConnectionError(request=_request()))

    with pytest.raises(ProviderUnavailable) as caught:
        await _extract(provider)
    assert caught.value.error_code == "provider_unavailable"


@pytest.mark.parametrize(
    ("exception_class", "status_code"),
    [
        (anthropic.AuthenticationError, 401),
        (anthropic.RateLimitError, 429),
        (anthropic.InternalServerError, 500),
        (anthropic.BadRequestError, 400),
    ],
)
async def test_api_errors_map_to_provider_unavailable(
    monkeypatch: pytest.MonkeyPatch, exception_class: type[Any], status_code: int
) -> None:
    error = exception_class("upstream detail", response=_response(status_code), body=None)
    provider = _provider(monkeypatch, error)

    with pytest.raises(ProviderUnavailable) as caught:
        await _extract(provider)

    assert caught.value.error_code == "provider_unavailable"


async def test_error_messages_do_not_echo_provider_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream text can be long and can contain the request payload."""
    provider = _provider(
        monkeypatch,
        anthropic.BadRequestError(
            "evidence content leaked into an upstream message",
            response=_response(400),
            body=None,
        ),
    )

    with pytest.raises(ProviderUnavailable) as caught:
        await _extract(provider)

    assert "leaked" not in str(caught.value)


async def test_a_refusal_is_a_provider_failure_not_an_empty_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal must never be stored as 'this evidence contained nothing'."""
    provider = _provider(
        monkeypatch,
        _Response(content=[], usage=_Usage(10, 0), stop_reason="refusal"),
    )

    with pytest.raises(ProviderUnavailable):
        await _extract(provider)


@pytest.mark.parametrize("content", [[], [_Block("")], [_Block("   \n ")]])
async def test_empty_output_is_a_provider_failure(
    monkeypatch: pytest.MonkeyPatch, content: list[Any]
) -> None:
    provider = _provider(monkeypatch, _Response(content=content, usage=_Usage(5, 0)))

    with pytest.raises(ProviderUnavailable):
        await _extract(provider)


# --- the termination reason, which subdivision depends on ------------------
#
# Without this signal a response truncated at the output ceiling and a response
# that is simply malformed both arrive as unparseable JSON and cannot be told
# apart. One of them is worth retrying in smaller pieces and the other is not.


@pytest.mark.parametrize("raw", ["end_turn", "stop_sequence"])
async def test_ordinary_termination_is_reported_as_complete(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """A response that ended on its own terms carries no truncation signal."""
    provider = _provider(
        monkeypatch,
        _Response(content=[_Block('{"items": []}')], usage=_Usage(10, 5), stop_reason=raw),
    )

    result = await _extract(provider)

    assert result.stop_reason is StopReason.COMPLETE


async def test_hitting_the_output_ceiling_is_reported_as_max_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The signal the whole adaptive mechanism turns on.

    This is the exact shape the real 60,241-character export produced: output
    stopped at the ceiling with the JSON cut off mid-string. The adapter must
    surface it in provider-independent terms so the application can tell "too
    much to say" from "said something broken".
    """
    truncated = '{"items": [{"kind": "reported_claim", "data": {"claim": "unter'
    provider = _provider(
        monkeypatch,
        _Response(
            content=[_Block(truncated)], usage=_Usage(6_374, 8_000), stop_reason="max_tokens"
        ),
    )

    result = await _extract(provider)

    assert result.stop_reason is StopReason.MAX_OUTPUT
    # Carried, not interpreted: the adapter does not parse or reject it.
    assert result.raw_output == truncated
    assert result.output_tokens == 8_000


@pytest.mark.parametrize("raw", ["tool_use", "pause_turn", "something_invented_later", None])
async def test_an_unmodelled_stop_reason_is_neither_a_crash_nor_a_truncation(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    """A new SDK value must degrade to the safe reading, not to overflow.

    Mapping an unknown reason onto `MAX_OUTPUT` would make the service subdivide
    on responses that were never truncated; raising would take the run down over
    a vocabulary change. `OTHER` is handled exactly like a normal completion, so
    the output still faces the same strict parse and grounding check — which is
    what actually decides whether it is usable.
    """
    provider = _provider(
        monkeypatch,
        _Response(content=[_Block('{"items": []}')], usage=_Usage(10, 5), stop_reason=raw),
    )

    result = await _extract(provider)

    assert result.stop_reason is StopReason.OTHER


async def test_a_refusal_still_raises_before_any_stop_reason_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusal handling is unchanged by the new field, and comes first.

    A refusal is a provider failure, never a truncation and never an empty
    extraction, so it must not reach the mapping at all.
    """
    provider = _provider(
        monkeypatch,
        _Response(content=[_Block("refused")], usage=_Usage(10, 1), stop_reason="refusal"),
    )

    with pytest.raises(ProviderUnavailable):
        await _extract(provider)


def test_the_default_result_reports_a_complete_response() -> None:
    """Every existing test double and the fake provider keep working unchanged.

    The default is also the safe reading: a provider that says nothing about how
    it stopped has not told us anything was truncated, and treating silence as
    overflow would subdivide sources that were read perfectly well.
    """
    result = ProviderResult(raw_output='{"items": []}', provider="fake", model="fake-1")

    assert result.stop_reason is StopReason.COMPLETE
