"""Deterministic providers for tests.

`FakeExtractionProvider` returns a scripted response with no network, no
credentials and no randomness, so the bulk of the extraction suite is fast and
repeatable. `ScriptedFailureProvider` raises a chosen provider error.

These live in `src/`, not `tests/`, for one reason: the fake is also what runs
when no credentials are configured, so the API is exercisable end to end on a
laptop without an API key. That is a deliberate, documented default for the
`local` and `test` environments — `Settings` refuses it in production.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from intelligence_engine.extraction.errors import ProviderError
from intelligence_engine.extraction.provider import (
    ProviderIdentity,
    ProviderResult,
    StopReason,
)

PROVIDER_NAME = "fake"
FAKE_MODEL = "fake-deterministic-1"


class FakeExtractionProvider:
    """Returns a fixed payload, recording the prompts it was given."""

    def __init__(
        self,
        *,
        raw_output: str | None = None,
        payload: dict[str, Any] | None = None,
        model: str = FAKE_MODEL,
        input_tokens: int | None = 100,
        output_tokens: int | None = 50,
        stop_reason: StopReason = StopReason.COMPLETE,
    ) -> None:
        if raw_output is not None and payload is not None:
            raise ValueError("pass raw_output or payload, not both")
        if raw_output is not None:
            self._raw_output = raw_output
        else:
            self._raw_output = json.dumps(payload if payload is not None else {"items": []})
        self._model = model
        self._identity = ProviderIdentity(provider=PROVIDER_NAME, model=model)
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._stop_reason = stop_reason

        self.calls: list[dict[str, Any]] = []

    @property
    def identity(self) -> ProviderIdentity:
        """Configurable through `model`, so a test can swap models without a vendor.

        Proving "a changed model must not reuse" needs two identities and no
        network. This is how a test gets them.
        """
        return self._identity

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def extract(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> ProviderResult:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "json_schema": json_schema,
            }
        )
        return ProviderResult(
            raw_output=self._raw_output,
            provider=PROVIDER_NAME,
            model=self._model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            stop_reason=self._stop_reason,
        )


class IntervalScriptedProvider:
    """Answers each call according to the evidence text that call carried.

    Adaptive extraction is a control flow over *intervals*: a parent overflows,
    its children do not, and a sibling is never touched. Testing that needs a
    double that can answer differently per interval rather than per call
    ordinal, because call ordinal is exactly the thing subdivision changes —
    a test written against ordinals would pass for the wrong reason as soon as
    the traversal order moved.

    `responder` receives the evidence slice the service actually sent, pulled
    back out of the user prompt, and returns a `ProviderResult` or raises. The
    slices it sees are the real ones, so a test can key on content and assert
    that only the interval it made overflow was ever subdivided.
    """

    def __init__(
        self,
        responder: Callable[[str], ProviderResult],
        *,
        model: str = FAKE_MODEL,
    ) -> None:
        self._responder = responder
        self._identity = ProviderIdentity(provider=PROVIDER_NAME, model=model)
        self.calls: list[dict[str, Any]] = []
        #: Every evidence slice sent, in call order. The subdivision tree as the
        #: provider saw it.
        self.intervals_seen: list[str] = []

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def extract(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> ProviderResult:
        content = extract_evidence_block(user_prompt)
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "json_schema": json_schema,
                "content": content,
            }
        )
        self.intervals_seen.append(content)
        return self._responder(content)


def extract_evidence_block(user_prompt: str) -> str:
    """The exact text between the prompt's evidence delimiters.

    Mirrors `prompt.USER_PROMPT_TEMPLATE` rather than re-deriving the interval
    from offsets, so a test double sees precisely what the model would.
    """
    opening = "<evidence>\n"
    closing = "\n</evidence>"
    start = user_prompt.index(opening) + len(opening)
    return user_prompt[start : user_prompt.index(closing, start)]


class ScriptedFailureProvider:
    """Always raises, to exercise failure persistence."""

    def __init__(self, error: ProviderError, *, model: str = FAKE_MODEL) -> None:
        self._error = error
        self._identity = ProviderIdentity(provider=PROVIDER_NAME, model=model)
        self.call_count = 0

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    async def extract(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> ProviderResult:
        self.call_count += 1
        raise self._error
