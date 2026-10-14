"""The provider trust boundary. Prove it rejects, not just that it accepts.

Every test here is a shape a real model could plausibly emit. If any of these
were accepted, the database would hold derived records that look valid and are
not.
"""

from __future__ import annotations

import json

import pytest

from intelligence_engine.extraction.errors import (
    ExtractionOverflow,
    InvalidProviderOutput,
    UngroundedItem,
)
from intelligence_engine.extraction.schemas import (
    MAX_ITEMS_PER_RUN,
    MAX_QUOTE_LENGTH,
    MIN_QUOTE_LENGTH,
    ItemKind,
)
from intelligence_engine.extraction.validation import parse_output, verify_grounding
from tests.extraction_fixtures import (
    EVIDENCE_CONTENT,
    QUOTE_CLAIM,
    claim_item,
    payload_with_items,
    valid_output_json,
    valid_payload,
)


def test_valid_output_parses_into_typed_items() -> None:
    output = parse_output(valid_output_json())

    assert len(output.items) == 4
    kinds = [item.kind for item in output.items]
    assert kinds == [
        ItemKind.REPORTED_CLAIM,
        ItemKind.REPORTED_CLAIM,
        ItemKind.ENTITY_MENTION,
        ItemKind.OBSERVATION,
    ]


def test_empty_item_list_is_valid() -> None:
    """Evidence that supports no structured items is a correct extraction."""
    output = parse_output(json.dumps({"items": []}))
    assert output.items == []


def test_reported_claim_keeps_the_attribution() -> None:
    """Epistemics: who said it must survive parsing."""
    output = parse_output(valid_output_json())
    first = output.items[0]

    assert first.kind is ItemKind.REPORTED_CLAIM
    assert first.data.claimed_by == "An engineer at DevGuild"


# --- structural rejection -------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json at all",
        "{",
        "null",
        "[]",
        '{"items": "not-a-list"}',
        '{"items": null}',
        '```json\n{"items": []}\n```',  # markdown fence the prompt forbids
    ],
)
def test_non_conforming_json_is_rejected(raw: str) -> None:
    with pytest.raises(InvalidProviderOutput):
        parse_output(raw)


def test_missing_items_key_is_rejected() -> None:
    with pytest.raises(InvalidProviderOutput):
        parse_output(json.dumps({}))


def test_unknown_item_kind_is_rejected() -> None:
    """A kind we never defined must fail, not be stored as an unknown string."""
    payload = payload_with_items(
        {
            "kind": "personality_assessment",
            "data": {"summary": "confident"},
            "grounding": {"quote": QUOTE_CLAIM},
        }
    )
    with pytest.raises(InvalidProviderOutput):
        parse_output(json.dumps(payload))


def test_missing_required_field_is_rejected() -> None:
    payload = payload_with_items(
        {
            "kind": "reported_claim",
            "data": {"claimed_by": "An engineer"},  # no `claim`
            "grounding": {"quote": QUOTE_CLAIM},
        }
    )
    with pytest.raises(InvalidProviderOutput):
        parse_output(json.dumps(payload))


def test_missing_grounding_is_rejected() -> None:
    payload = payload_with_items(
        {
            "kind": "observation",
            "data": {"observation": "Something happened."},
        }
    )
    with pytest.raises(InvalidProviderOutput):
        parse_output(json.dumps(payload))


def test_wrong_field_type_is_rejected() -> None:
    payload = payload_with_items(
        {
            "kind": "reported_claim",
            "data": {"claim": ["a", "list", "not", "a", "string"]},
            "grounding": {"quote": QUOTE_CLAIM},
        }
    )
    with pytest.raises(InvalidProviderOutput):
        parse_output(json.dumps(payload))


def test_extra_keys_are_rejected_not_silently_dropped() -> None:
    """Model garbage must fail loudly; a dropped field is a silent data loss."""
    item = claim_item()
    item["data"]["invented_field"] = "please store me"
    with pytest.raises(InvalidProviderOutput):
        parse_output(json.dumps(payload_with_items(item)))


def test_extra_top_level_key_is_rejected() -> None:
    payload = valid_payload()
    payload["commentary"] = "Here is what I extracted!"
    with pytest.raises(InvalidProviderOutput):
        parse_output(json.dumps(payload))


def test_unknown_enum_value_is_rejected() -> None:
    with pytest.raises(InvalidProviderOutput):
        parse_output(json.dumps(payload_with_items(claim_item(stance="probably-true"))))


def test_a_confidence_field_is_rejected_as_an_extra_key() -> None:
    """The field is gone, and re-adding it by model output must not work."""
    item = claim_item()
    item["model_confidence"] = "high"
    with pytest.raises(InvalidProviderOutput):
        parse_output(json.dumps(payload_with_items(item)))


def test_unknown_entity_type_is_rejected() -> None:
    payload = payload_with_items(
        {
            "kind": "entity_mention",
            "data": {"name": "DevGuild", "entity_type": "sentient_being"},
            "grounding": {"quote": QUOTE_CLAIM},
        }
    )
    with pytest.raises(InvalidProviderOutput):
        parse_output(json.dumps(payload))


@pytest.mark.parametrize("quote", ["", "short", "x" * (MAX_QUOTE_LENGTH + 1)])
def test_out_of_bounds_quote_length_is_rejected(quote: str) -> None:
    """Too short is not grounding; too long duplicates the evidence body."""
    with pytest.raises(InvalidProviderOutput):
        parse_output(json.dumps(payload_with_items(claim_item(quote=quote))))


def test_minimum_length_quote_is_accepted() -> None:
    quote = EVIDENCE_CONTENT[:MIN_QUOTE_LENGTH]
    output = parse_output(json.dumps(payload_with_items(claim_item(quote=quote))))
    verify_grounding(output, EVIDENCE_CONTENT)


def test_too_many_items_is_rejected_as_an_overflow() -> None:
    """A bound on runaway output, enforced before anything touches the database.

    Rejected as `ExtractionOverflow` specifically, not as generic bad output.
    The response is not malformed — it is a correct answer to too large a
    question — and that distinction is what licenses the service to subdivide
    the interval and ask again rather than giving up.
    """
    items = [claim_item() for _ in range(MAX_ITEMS_PER_RUN + 1)]
    with pytest.raises(ExtractionOverflow):
        parse_output(json.dumps(payload_with_items(*items)))


def test_a_well_formed_but_over_long_response_reports_only_the_count() -> None:
    """Pydantic short-circuits: too long is reported *instead of* member errors.

    Measured, not assumed. A 52-item list containing one malformed item yields
    exactly one error — `too_long` on `items` — because the length check runs
    before the members are validated. So `_is_only_too_many_items` sees a lone
    `too_long` and classifies the response as an overflow even though something
    else is also wrong with it.

    That is safe, and worth stating rather than hiding. The interval subdivides,
    the children are smaller, and the malformed item — if it recurs at all —
    then fails as ordinary invalid output and stops the run loudly. The cost of
    the misclassification is bounded by `MAX_SUBDIVISION_DEPTH`; the alternative,
    validating every member of a list we have already decided is too long,
    would buy a better error message for a response that is being rejected
    either way.
    """
    items = [claim_item() for _ in range(MAX_ITEMS_PER_RUN + 1)]
    items.append({"kind": "reported_claim", "data": {}, "grounding": {"quote": "12345678"}})

    with pytest.raises(ExtractionOverflow):
        parse_output(json.dumps(payload_with_items(*items)))


def test_a_correctly_sized_but_malformed_response_is_not_an_overflow() -> None:
    """The negative case the subdivision trigger depends on.

    Ordinary bad output must never be answered by splitting the interval and
    calling again: the model did not run out of room, it emitted something
    wrong, and asking it twice more costs twice as much to learn the same thing.
    """
    items = [claim_item() for _ in range(3)]
    items.append({"kind": "reported_claim", "data": {}, "grounding": {"quote": "12345678"}})

    with pytest.raises(InvalidProviderOutput) as caught:
        parse_output(json.dumps(payload_with_items(*items)))
    assert not isinstance(caught.value, ExtractionOverflow)


def test_max_items_is_accepted() -> None:
    items = [claim_item() for _ in range(MAX_ITEMS_PER_RUN)]
    assert len(parse_output(json.dumps(payload_with_items(*items))).items) == MAX_ITEMS_PER_RUN


# --- error message hygiene ------------------------------------------------


def test_validation_error_is_bounded_and_does_not_echo_the_payload() -> None:
    """The message lands in a DB column and a log line, so it must stay small.

    Pydantic's own report includes the offending input; ours must not, or
    evidence text would leak into logs through a validation failure.
    """
    private_text = "PRIVATE-EVIDENCE-abc123"
    items = [claim_item(claim=private_text, stance="nope") for _ in range(10)]

    with pytest.raises(InvalidProviderOutput) as caught:
        parse_output(json.dumps(payload_with_items(*items)))

    message = str(caught.value)
    assert private_text not in message
    assert len(message) <= 500
    assert "and 7 more" in message


# --- grounding ------------------------------------------------------------


def test_grounding_accepts_verbatim_quotes() -> None:
    verify_grounding(parse_output(valid_output_json()), EVIDENCE_CONTENT)


def test_grounding_rejects_an_invented_quote() -> None:
    """The cheapest hallucination check available: was this text really there?"""
    fabricated = "He told me he was the best engineer in Kestrel Bay"
    with pytest.raises(UngroundedItem):
        verify_grounding(
            parse_output(json.dumps(payload_with_items(claim_item(quote=fabricated)))),
            EVIDENCE_CONTENT,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        QUOTE_CLAIM.upper(),
        QUOTE_CLAIM.replace(" ", "  "),
        QUOTE_CLAIM.replace("LeetCode", "leetcode"),
        QUOTE_CLAIM + " and more",
        QUOTE_CLAIM.replace("serious", "very serious"),
    ],
)
def test_grounding_rejects_near_misses(mutation: str) -> None:
    """No fuzzy matching. "Close enough" is exactly what a hallucination looks like."""
    with pytest.raises(UngroundedItem):
        verify_grounding(
            parse_output(json.dumps(payload_with_items(claim_item(quote=mutation)))),
            EVIDENCE_CONTENT,
        )


def test_grounding_failure_names_the_offending_item() -> None:
    payload = payload_with_items(claim_item(), claim_item(quote="totally invented text here"))

    with pytest.raises(UngroundedItem, match="Item 1"):
        verify_grounding(parse_output(json.dumps(payload)), EVIDENCE_CONTENT)


def test_one_bad_quote_rejects_the_whole_run() -> None:
    """No partial acceptance: a response we distrust is distrusted entirely."""
    payload = payload_with_items(claim_item(), claim_item(quote="not in the evidence at all"))
    output = parse_output(json.dumps(payload))

    with pytest.raises(UngroundedItem):
        verify_grounding(output, EVIDENCE_CONTENT)
