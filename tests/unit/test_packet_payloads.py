"""Reading stored payloads back, for every kind M2 can write.

The property this file protects is **totality**. A reader covering a subset of
`ItemKind` pushes "which kinds are allowed here?" into its callers, and that
question has different answers in different places — which is precisely how a
filter and a dispatch drift apart. `test_every_item_kind_can_be_read` is the
guard: add a kind to M2 without teaching this module to read it and the suite
fails, rather than an operator meeting a `KeyError`.
"""

from __future__ import annotations

import pytest

from intelligence_engine.extraction.payloads import (
    ItemStatement,
    UnreadablePayload,
    read_statement,
)
from intelligence_engine.extraction.schemas import ItemKind

_MINIMAL: dict[ItemKind, dict[str, object]] = {
    ItemKind.REPORTED_CLAIM: {"claim": "a claim"},
    ItemKind.OBSERVATION: {"observation": "an observation"},
    ItemKind.ENTITY_MENTION: {"name": "Aisyah", "entity_type": "person"},
    ItemKind.PROJECT: {"name": "Fernvault"},
    ItemKind.RESOURCE: {"resource_type": "article"},
    ItemKind.EVENT: {"name": "DevGuild Lab 1"},
    ItemKind.OPPORTUNITY: {"summary": "take the settlement module"},
    ItemKind.IDEA: {"idea": "start from a real project"},
}


def _read(kind: ItemKind, payload: dict[str, object]) -> ItemStatement:
    return read_statement(kind, payload, item_label="Extraction item X")


def test_every_item_kind_can_be_read() -> None:
    """Totality over `ItemKind`, asserted against the enum rather than a list."""
    assert set(_MINIMAL) == set(ItemKind)
    for kind, payload in _MINIMAL.items():
        assert _read(kind, payload).text


def test_a_reported_claim_keeps_its_speaker_and_never_gains_a_subject() -> None:
    statement = _read(
        ItemKind.REPORTED_CLAIM,
        {"claim": "Judging weighted the demo.", "claimed_by": "Aisyah", "stance": "asserted"},
    )

    assert statement.text == "Judging weighted the demo."
    assert statement.claimed_by == "Aisyah"
    assert statement.stance == "asserted"
    assert statement.subject is None


def test_an_observation_keeps_its_subject_and_never_gains_a_speaker() -> None:
    """The invariant the whole reported-claim design exists to protect.

    An observation about someone is not something they said. Collapsing the two
    turns "her profile lists Solidity" into an assertion she made.
    """
    statement = _read(
        ItemKind.OBSERVATION,
        {"observation": "The backend was scaffolded in one commit.", "subject": "a teammate"},
    )

    assert statement.subject == "a teammate"
    assert statement.claimed_by is None


def test_an_observation_is_never_given_a_stance() -> None:
    """It was not offered as advice or as a position, so inventing one lies."""
    assert _read(ItemKind.OBSERVATION, {"observation": "x happened"}).stance is None


def test_an_event_date_stays_as_the_evidence_worded_it() -> None:
    """Parsing "last spring" into a timestamp would invent precision."""
    statement = _read(ItemKind.EVENT, {"name": "DevGuild Lab 1", "occurred_at_text": "last spring"})

    assert statement.text == "DevGuild Lab 1"
    assert statement.attributes == (("occurred_at_text", "last spring"),)


def test_fields_the_evidence_did_not_supply_are_absent_not_defaulted() -> None:
    """Absent means "the evidence did not say", never "unknown value"."""
    statement = _read(ItemKind.PROJECT, {"name": "Fernvault"})

    assert statement.text == "Fernvault"
    assert statement.attributes == ()


def test_supplied_fields_appear_in_the_models_declaration_order() -> None:
    """Deterministic rendering starts here, not in the renderer."""
    statement = _read(
        ItemKind.ENTITY_MENTION,
        {"name": "Aisyah", "entity_type": "person", "described_as": "a teammate"},
    )

    assert statement.attributes == (("entity_type", "person"), ("described_as", "a teammate"))


def test_a_resource_without_a_title_falls_back_to_what_it_actually_has() -> None:
    """Its type is the only always-present field; a title is not manufactured."""
    statement = _read(
        ItemKind.RESOURCE, {"resource_type": "tool", "url": "https://example.invalid/t"}
    )

    assert statement.text == "https://example.invalid/t"


def test_a_payload_that_does_not_match_its_kind_fails_loudly() -> None:
    """Silently skipping would drop grounded material with no reader able to tell."""
    with pytest.raises(UnreadablePayload, match="readable observation"):
        _read(ItemKind.OBSERVATION, {"claim": "wrong shape for this kind"})


def test_the_failure_message_names_the_row_that_caused_it() -> None:
    with pytest.raises(UnreadablePayload, match="Extraction item X"):
        _read(ItemKind.IDEA, {"not_an_idea": True})


def test_an_extra_key_is_a_failure_rather_than_being_dropped() -> None:
    """Strict parse: stored JSONB that grew a field is a schema drift signal."""
    with pytest.raises(UnreadablePayload):
        _read(ItemKind.IDEA, {"idea": "fine", "confidence": 0.9})
