"""Extraction error model.

Every error carries a short `error_code`. That code is what gets stored in
`extraction_runs.error_code` and returned to HTTP clients, so it is a stable
identifier — never a message, never an exception dump. Messages are bounded and
authored here rather than taken from provider or library exception text, which
can be long and can contain request payloads.
"""

from __future__ import annotations

from intelligence_engine.core.errors import ApplicationError


class ExtractionError(ApplicationError):
    """Base for anything that can fail an extraction run."""

    error_code = "extraction_error"


class ProviderError(ExtractionError):
    """The provider call failed. Nothing was extracted."""

    error_code = "provider_error"


class ProviderTimeout(ProviderError):
    error_code = "provider_timeout"


class ProviderUnavailable(ProviderError):
    """Auth failure, rate limit, refusal, or a 5xx that survived SDK retries."""

    error_code = "provider_unavailable"


class ProviderNotConfigured(ProviderError):
    """No credentials configured, so no call was attempted."""

    error_code = "provider_not_configured"


class InvalidProviderOutput(ExtractionError):
    """Output did not survive strict validation.

    Covers non-JSON, unknown item kinds, missing or mistyped fields, and extra
    keys. Never retried automatically: a model that just emitted garbage
    usually emits more garbage, at full price.
    """

    error_code = "invalid_provider_output"


class UngroundedItem(ExtractionError):
    """An item's grounding quote is not verbatim in the evidence.

    The strongest hallucination signal available without another model call: if
    the quote cannot be found character-for-character in the source text, the
    model invented it, and the whole run is rejected.
    """

    error_code = "ungrounded_item"


class ExtractionOverflow(ExtractionError):
    """One interval demanded more than a single bounded response could carry.

    **The only failure that may cause subdivision**, and deliberately its own
    type rather than a flag on `InvalidProviderOutput`. Every other failure —
    a fabricated quote, a timeout, a rate limit, a refusal, ordinary malformed
    JSON — says something is wrong with *this attempt*, and answering it by
    splitting the interval and calling again would multiply the cost of a
    problem that has nothing to do with how much source text was sent.

    Raised on exactly two signals, both mechanical:

    1. the response carried more than `MAX_ITEMS_PER_RUN` items, which strict
       validation reports as a `too_long` error on `items`;
    2. the provider stopped at `StopReason.MAX_OUTPUT`, so the response is cut
       off and is an incomplete answer for the interval regardless of whether
       what arrived happens to parse.

    Distinct from `InvalidProviderOutput` in one further way that matters: a
    truncated response and a malformed one both fail to parse, and only the
    stop reason tells them apart. Without that signal this type would be
    guesswork.
    """

    error_code = "extraction_overflow"


class IrreducibleOverflow(ExtractionError):
    """An interval still overflows but can no longer be safely subdivided.

    The honest end of the recursion. Reaching it means a region of the source
    is denser than one bounded response can carry at the smallest unit this
    system is willing to split to — so that region is reported unread, by
    exact character range, and the source is incomplete.

    Everything this deliberately is not: it never truncates the item list to
    fit, never asks the model to paginate, never invents a second evidence
    item, and never recurses further. Work already succeeded on sibling
    intervals is committed and stays committed.
    """

    def __init__(self, start: int, end: int, reason: str) -> None:
        super().__init__(
            f"Interval [{start}:{end}) still overflows and cannot be subdivided further "
            f"({reason}). This region of the source is unread."
        )
        self.start = start
        self.end = end

    error_code = "irreducible_overflow"
