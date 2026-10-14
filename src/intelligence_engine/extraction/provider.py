"""The external provider boundary.

Everything provider-specific stops here. The protocol takes plain strings and
returns `ProviderResult` — no SDK types, no provider exceptions, no JSON
parsing. The application layer never imports `anthropic`.

Note what this deliberately is not: there is no registry, no router, no
fallback chain, no consensus. It is one method and one property, because that is
what M2 needs. Error types live in `errors.py` alongside the rest of the
extraction error model.

The property is `identity`, and it is here rather than on the result because of
what it is for: deciding whether stored output may be *served instead of* a call.
That decision happens before the call, so it cannot be made from anything the
call returns. See `ProviderIdentity`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Protocol


class StopReason(enum.StrEnum):
    """Why the provider stopped generating, in provider-independent terms.

    Deliberately three values, not a faithful mirror of any vendor's vocabulary.
    The application asks this enum exactly one question — *was the response cut
    off by the output ceiling?* — because that is the only distinction that
    changes what the application does next.

    Without it, a response truncated at `max_tokens` and a response that is
    simply malformed both arrive as unparseable JSON and are indistinguishable.
    The first means "this interval demanded more than one bounded response can
    carry", which is a reason to subdivide the interval and try again; the
    second means the model emitted garbage, which is a reason to stop. Guessing
    wrong in either direction is expensive: recursively re-paying for garbage,
    or refusing to read a source that is merely long.

    `OTHER` exists so a vendor adding a new reason is neither a crash nor a
    silent `COMPLETE`. It is treated exactly like `COMPLETE`: the output still
    faces the same strict parse and the same grounding check, which is what
    actually decides whether it is usable.
    """

    #: Generation ended on its own terms. The response is whole.
    COMPLETE = "complete"
    #: Generation hit the output-token ceiling. The response is truncated, and
    #: is therefore an incomplete answer for the interval that was sent.
    MAX_OUTPUT = "max_output"
    #: Anything this application does not model. Handled as `COMPLETE`.
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """Which generator a call will go to, known *before* the call is made.

    This exists because reuse is decided before any call. The application asks
    "is there stored output I can serve instead of paying for this?" while the
    only thing it holds is a configured provider — so any identity that arrives
    with the response is one round of spending too late to prevent the round
    that produced it.

    It is deliberately not `ProviderResult`'s `provider`/`model`. Those report
    *who answered*, which is the honest provenance of a finished call and can
    legitimately differ from what was asked for: a vendor may resolve an alias to
    a dated snapshot, and `deepseek_provider` already falls back to its own
    configured name when the response omits one. Keying reuse on a value that is
    only knowable afterwards is not possible; keying it on a value that is
    *sometimes* the same is worse, because it works until it quietly does not.

    So this is the requested identity: the vendor this client talks to, and the
    model string it will send. Both are fixed at construction and neither depends
    on a response.
    """

    #: The vendor. Matches `ProviderResult.provider` and each module's
    #: `PROVIDER_NAME`, so a stored row names the same thing either way.
    provider: str
    #: The model string this client sends. Not the one a response reports back.
    model: str


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """A raw, still-untrusted provider response.

    `raw_output` is text on purpose. Parsing and validating it is the
    application's job, so malformed output flows through one tested code path
    rather than through whatever each provider happens to do.
    """

    raw_output: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    #: Defaulted so every existing test double and the fake provider keep
    #: working unchanged, and so the safe reading — "nothing was truncated" —
    #: is what a provider that does not report one implies.
    stop_reason: StopReason = StopReason.COMPLETE


class ExtractionProvider(Protocol):
    """Turns evidence text into a raw structured-extraction response."""

    @property
    def identity(self) -> ProviderIdentity:
        """Who this provider will ask, before it asks.

        Part of the Protocol rather than an optional extra, and structurally so:
        a double that omits it stops satisfying `ExtractionProvider` and fails
        type-checking at the call site rather than silently reusing another
        model's output.
        """
        ...

    async def extract(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> ProviderResult: ...
