"""The trust boundary for classifier output.

Model output is untrusted input. It arrives as a string and passes three
independent checks before anything reaches the database:

1. **Structural** — strict Pydantic parse. Unknown relation types, missing
   fields, wrong types and extra keys all fail. Nothing is coerced or dropped.
2. **Batch identity** — every returned `pair_id` must be one that was supplied,
   every supplied id must come back, and none twice. This is M4's equivalent of
   M2's verbatim-quote check: it is the cheapest available guarantee that the
   explanation you are about to store is attached to the two claims it was
   actually written about. Without it, a reordered or hallucinated id would
   silently bind an assertion to the wrong pair of statements — a failure no
   reader could detect afterwards, because the stored row would look perfectly
   well-formed.
3. **Semantic shape** — direction present exactly when the relation is
   directional, explanation present exactly when the relation is meaningful.

Any failure rejects the **whole batch**. Partial acceptance would mean storing
assertions from a response already known to be unreliable.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from intelligence_engine.intelligence.candidates import CandidatePair
from intelligence_engine.intelligence.errors import InvalidRelationOutput
from intelligence_engine.intelligence.schemas import (
    DIRECTIONAL_RELATION_TYPES,
    MEANINGFUL_RELATION_TYPES,
    RelationDirection,
    RelationOutput,
    RelationResult,
)

# Bounded so a long Pydantic report cannot become a large database row or log line.
MAX_REPORTED_ERRORS = 3

_DIRECTION_BY_WIRE_VALUE = {
    "a_qualifies_b": RelationDirection.LEFT_TO_RIGHT,
    "b_qualifies_a": RelationDirection.RIGHT_TO_LEFT,
}


def parse_output(raw_output: str) -> RelationOutput:
    """Strictly parse a classifier response, or raise `InvalidRelationOutput`."""
    try:
        return RelationOutput.model_validate_json(raw_output)
    except ValidationError as exc:
        raise InvalidRelationOutput(_summarise(exc)) from exc


def bind_results(
    output: RelationOutput, batch: Mapping[str, CandidatePair]
) -> list[tuple[CandidatePair, RelationResult, RelationDirection]]:
    """Match every result to the pair it claims to describe, or reject the batch.

    Returns the pairs in the order they were *supplied*, not the order the model
    returned them, so persistence is deterministic regardless of how the model
    chose to order its answer.
    """
    seen: dict[str, RelationResult] = {}
    for result in output.results:
        if result.pair_id not in batch:
            raise InvalidRelationOutput(
                f"Result references pair id '{result.pair_id}', which was not supplied"
            )
        if result.pair_id in seen:
            raise InvalidRelationOutput(f"Pair id '{result.pair_id}' was answered more than once")
        seen[result.pair_id] = result

    missing = [pair_id for pair_id in batch if pair_id not in seen]
    if missing:
        raise InvalidRelationOutput(
            f"{len(missing)} of {len(batch)} supplied pairs were not answered "
            f"(first missing: '{missing[0]}')"
        )

    return [(batch[pair_id], seen[pair_id], _resolve_direction(seen[pair_id])) for pair_id in batch]


def _resolve_direction(result: RelationResult) -> RelationDirection:
    """Turn the wire's A/B direction into the stored canonical direction.

    Pairs are given to the model with A = the canonical left item and B = the
    canonical right item, so this mapping is a rename rather than a decision.
    """
    is_directional = result.relation_type in DIRECTIONAL_RELATION_TYPES

    if is_directional and result.direction is None:
        raise InvalidRelationOutput(
            f"Pair '{result.pair_id}' is '{result.relation_type.value}' but states no direction"
        )
    if not is_directional and result.direction is not None:
        raise InvalidRelationOutput(
            f"Pair '{result.pair_id}' is '{result.relation_type.value}', "
            "which is symmetric, but states a direction"
        )

    is_meaningful = result.relation_type in MEANINGFUL_RELATION_TYPES
    if is_meaningful and not result.explanation:
        raise InvalidRelationOutput(
            f"Pair '{result.pair_id}' asserts '{result.relation_type.value}' with no explanation"
        )
    if not is_meaningful and result.explanation:
        # Not pedantry: an explanation attached to "no relation" is the model
        # arguing for a relationship it declined to assert, and storing it would
        # put unlabelled speculation in front of a reader.
        raise InvalidRelationOutput(
            f"Pair '{result.pair_id}' asserts no meaningful relation but supplies an explanation"
        )

    if result.direction is None:
        return RelationDirection.SYMMETRIC
    return _DIRECTION_BY_WIRE_VALUE[result.direction]


def _summarise(exc: ValidationError) -> str:
    """A short, storable description of why validation failed.

    Pydantic's full report includes the offending input, which for a classifier
    response echoes claim text into logs and error columns. Only the field path
    and the error type are kept.
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
