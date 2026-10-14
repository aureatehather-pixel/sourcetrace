"""The trust boundary for provider output.

Model output is untrusted input. It arrives as a string and gets two
independent checks before anything reaches the database:

1. **Structural** — strict Pydantic parse. Unknown kinds, missing fields, wrong
   types and extra keys all fail. Nothing is coerced, nothing is dropped.
2. **Semantic** — every item's grounding quote must appear verbatim in the
   evidence. This is the cheapest hallucination check that actually works: a
   model that invented a quote could not have found it in the source.

Either failure rejects the *entire* run. Partial acceptance would mean storing
items from a response we know to be unreliable.

One structural failure is singled out and given its own type. A response
carrying more than `MAX_ITEMS_PER_RUN` items is not a bad response — it is a
correct response to too large a question, and the answer is to ask a smaller
one. That is `ExtractionOverflow`, and it is the only validation outcome the
service is allowed to answer by subdividing the interval. Keeping it separate
here, on Pydantic's structured error record, is what stops the service from
having to guess from an error message.
"""

from __future__ import annotations

from pydantic import ValidationError

from intelligence_engine.extraction.errors import (
    ExtractionOverflow,
    InvalidProviderOutput,
    UngroundedItem,
)
from intelligence_engine.extraction.schemas import MAX_ITEMS_PER_RUN, ExtractionOutput

# Bounded so a 40-error Pydantic report cannot become a 20KB database row.
MAX_REPORTED_ERRORS = 3

#: Pydantic's own error type and location for "the items list is longer than
#: `MAX_ITEMS_PER_RUN`". Matched structurally, on the parsed error record,
#: never by searching the rendered message: the message is prose that Pydantic
#: is free to reword, while `type` and `loc` are its published contract. This
#: is the difference between a classification a release can silently break and
#: one it cannot.
_TOO_MANY_ITEMS_TYPE = "too_long"
_TOO_MANY_ITEMS_LOC: tuple[str, ...] = ("items",)


def parse_output(raw_output: str) -> ExtractionOutput:
    """Strictly parse a provider response.

    Raises `ExtractionOverflow` when the single reason validation failed is
    that the model returned more than `MAX_ITEMS_PER_RUN` items — the interval
    was too large for one bounded response, which is answerable by subdividing
    it. Every other validation failure raises `InvalidProviderOutput`, which is
    not.

    The distinction is narrow on purpose. A response that is both over-long and
    malformed in some other way is *not* treated as an overflow: something else
    is wrong with it, and subdividing would pay repeatedly to find that out.
    """
    try:
        return ExtractionOutput.model_validate_json(raw_output)
    except ValidationError as exc:
        if _is_only_too_many_items(exc):
            raise ExtractionOverflow(
                f"Response carried more than {MAX_ITEMS_PER_RUN} items"
            ) from exc
        raise InvalidProviderOutput(_summarise(exc)) from exc


def _is_only_too_many_items(exc: ValidationError) -> bool:
    """True when the item-count bound is the *sole* reason validation failed."""
    errors = exc.errors()
    return len(errors) == 1 and (
        errors[0]["type"] == _TOO_MANY_ITEMS_TYPE and tuple(errors[0]["loc"]) == _TOO_MANY_ITEMS_LOC
    )


def verify_grounding(output: ExtractionOutput, evidence_content: str) -> None:
    """Reject the run unless every quote is verbatim in the evidence.

    Exact substring matching, with no normalisation. Allowing "close enough"
    matches would defeat the point: the check exists precisely to catch text
    the model produced rather than read.
    """
    for index, item in enumerate(output.items):
        if item.grounding.quote not in evidence_content:
            raise UngroundedItem(
                f"Item {index} ({item.kind.value}) has a grounding quote "
                f"that does not appear verbatim in the evidence"
            )


def _summarise(exc: ValidationError) -> str:
    """A short, storable description of why validation failed.

    Pydantic's full report includes the offending input, which for a provider
    response can be large and can echo evidence content into logs and error
    columns. Only the field path and the error type are kept.
    """
    errors = exc.errors()
    parts = [
        f"{'.'.join(str(loc) for loc in error['loc']) or '<root>'}: {error['type']}"
        for error in errors[:MAX_REPORTED_ERRORS]
    ]
    summary = f"{len(errors)} validation error(s): " + "; ".join(parts)
    if len(errors) > MAX_REPORTED_ERRORS:
        summary += f"; and {len(errors) - MAX_REPORTED_ERRORS} more"
    return summary
