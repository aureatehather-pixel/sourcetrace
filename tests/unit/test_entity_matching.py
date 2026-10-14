"""The deterministic matcher, tested as pure functions.

These are the tests that decide whether M3 is safe. Entity resolution's only
irreversible failure is the false merge - two real people fused into one
identity - and the matcher is the thing that can cause it. So the negative
cases here matter more than the positive ones: most of this file proves the
matcher *declines*.
"""

from __future__ import annotations

import uuid

import pytest

from intelligence_engine.entities.matching import (
    ROLE_SOURCE_KINDS,
    decide,
    matchable_name,
    normalize_name,
)
from intelligence_engine.entities.schemas import ResolutionRole, UnresolvedReason
from intelligence_engine.extraction.schemas import ItemKind

# Written as escapes rather than literal glyphs: the point of the NFKC tests
# is that these are *different code points* rendering as "DevGuild", and a
# literal would be indistinguishable from ASCII when reading the file.
FULLWIDTH_DEVGUILD = "\uff24\uff45\uff56\uff27\uff55\uff49\uff4c\uff44"
# The same word twice: precomposed U+00E9, then "e" plus a combining acute.
PRECOMPOSED_CAFE = "Caf\u00e9"
DECOMPOSED_CAFE = "Cafe\u0301"

# --- normalisation: what it does ------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("DevGuild", "devguild"),  # case is not identity
        ("DevGuild", "DEVGUILD"),
        ("  Zilan  ", "Zilan"),  # surrounding whitespace is typography
        ("Zi  Lan", "Zi Lan"),  # a repeated space is a stray keystroke
        ("Zi\tLan", "Zi Lan"),
        ("Zi\nLan", "Zi Lan"),
        (FULLWIDTH_DEVGUILD, "DevGuild"),  # NFKC: an encoding, not another name
        (PRECOMPOSED_CAFE, DECOMPOSED_CAFE),  # NFKC: one string written two ways
    ],
)
def test_normalisation_treats_these_as_the_same_key(left: str, right: str) -> None:
    assert normalize_name(left) == normalize_name(right)


# --- normalisation: what it deliberately does NOT do ----------------------


@pytest.mark.parametrize(
    ("left", "right", "why"),
    [
        ("Zi Lan", "Zilan", "removing internal spaces guesses at a naming convention"),
        ("Bob", "Robert", "nicknames are claims, not normalisation"),
        ("OpenAI Inc.", "OpenAI", "a legal suffix can be the only distinguishing part"),
        ("Alex Lee", "A. Lee", "expanding initials invents information"),
        ("Cafe", PRECOMPOSED_CAFE, "accent folding merges different words"),
        ("Zilan Orr", "Tan Zilan", "reordering assumes a naming convention"),
        ("DevGuild", "Dev League", "inserting a word boundary is a guess"),
    ],
)
def test_normalisation_keeps_these_apart(left: str, right: str, why: str) -> None:
    assert normalize_name(left) != normalize_name(right), why


def test_whitespace_only_names_produce_an_unmatchable_key() -> None:
    """An empty key must never be a key that matches other empty keys."""
    assert normalize_name("   ") == ""
    assert normalize_name("\t\n") == ""


def test_normalisation_is_idempotent() -> None:
    once = normalize_name(f"  {FULLWIDTH_DEVGUILD}  ")
    assert normalize_name(once) == once


# --- the decision ---------------------------------------------------------


def test_exactly_one_candidate_resolves() -> None:
    only = uuid.uuid4()

    decision = decide("zilan", [only])

    assert decision.entity_id == only
    assert decision.reason is None


def test_no_candidates_stays_unresolved() -> None:
    decision = decide("zilan", [])

    assert decision.entity_id is None
    assert decision.reason is UnresolvedReason.NO_CANDIDATE


def test_two_candidates_are_never_force_merged() -> None:
    """The false-merge guard. Two real people can share a name."""
    decision = decide("alex lee", [uuid.uuid4(), uuid.uuid4()])

    assert decision.entity_id is None
    assert decision.reason is UnresolvedReason.AMBIGUOUS_CANDIDATES


def test_many_candidates_are_never_force_merged() -> None:
    decision = decide("john smith", [uuid.uuid4() for _ in range(9)])

    assert decision.entity_id is None
    assert decision.reason is UnresolvedReason.AMBIGUOUS_CANDIDATES


def test_an_empty_key_never_resolves_even_with_a_candidate() -> None:
    """Otherwise a blank name would silently attach to whatever it found."""
    decision = decide("", [uuid.uuid4()])

    assert decision.entity_id is None
    assert decision.reason is UnresolvedReason.NO_MATCHABLE_NAME


def test_a_decision_never_carries_both_an_entity_and_a_reason() -> None:
    for candidates in ([], [uuid.uuid4()], [uuid.uuid4(), uuid.uuid4()]):
        decision = decide("name", candidates)
        assert (decision.entity_id is None) != (decision.reason is None)


# --- which name an item offers -------------------------------------------


def test_entity_mention_offers_its_name() -> None:
    payload = {"name": "Zilan", "entity_type": "person", "described_as": None}

    assert matchable_name(ItemKind.ENTITY_MENTION, payload) == "Zilan"


def test_reported_claim_offers_its_attribution() -> None:
    payload = {"claim": "Ship things.", "claimed_by": "Zilan", "stance": "recommended"}

    assert matchable_name(ItemKind.REPORTED_CLAIM, payload) == "Zilan"


def test_an_unattributed_claim_offers_nothing() -> None:
    payload = {"claim": "Ship things.", "claimed_by": None, "stance": None}

    assert matchable_name(ItemKind.REPORTED_CLAIM, payload) is None


def test_a_role_attribution_is_offered_but_will_not_match_a_person_name() -> None:
    """The real M2 finding: `Engineer at DevGuild` identifies nobody.

    It is offered to the matcher honestly and then fails to match anything,
    which is how an unnamed person stays permanently unresolved.
    """
    payload = {
        "claim": "Serious projects matter more than LeetCode.",
        "claimed_by": "Engineer at DevGuild",
        "stance": "asserted",
    }

    offered = matchable_name(ItemKind.REPORTED_CLAIM, payload)

    assert offered == "Engineer at DevGuild"
    assert normalize_name(offered or "") != normalize_name("DevGuild")


@pytest.mark.parametrize(
    "kind",
    [
        ItemKind.PROJECT,
        ItemKind.RESOURCE,
        ItemKind.EVENT,
        ItemKind.OPPORTUNITY,
        ItemKind.IDEA,
    ],
)
def test_other_kinds_offer_no_name_at_all(kind: ItemKind) -> None:
    """M3 resolves mentions, attributions and subjects. Nothing else."""
    assert matchable_name(kind, {"anything": "here"}) is None


def test_an_observation_offers_its_subject() -> None:
    payload = {"observation": "listed Solidity and Rust", "subject": "Zi Lan"}
    assert matchable_name(ItemKind.OBSERVATION, payload) == "Zi Lan"


def test_an_observation_without_a_subject_offers_no_name() -> None:
    """Omitting the subject is the correct extraction for evidence that does not
    say who it is about. It must read as "no name offered", never as a name that
    matched nothing - the two lead to different operator decisions."""
    assert matchable_name(ItemKind.OBSERVATION, {"observation": "the room was loud"}) is None


def test_a_first_person_subject_is_carried_through_unchanged() -> None:
    """ "I" is an honest subject that names nobody on file. It must reach the
    matcher as written so the decision stays NO_CANDIDATE, rather than being
    silently rewritten into the author's name - author modelling is not built."""
    payload = {"observation": "I shipped the parser in one night", "subject": "I"}
    assert matchable_name(ItemKind.OBSERVATION, payload) == "I"


def test_a_malformed_payload_fails_loudly_rather_than_yielding_a_wrong_name() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        matchable_name(ItemKind.ENTITY_MENTION, {"nombre": "Zilan"})


# --- role/kind pairing ----------------------------------------------------


def test_every_role_pins_exactly_one_item_kind() -> None:
    assert ROLE_SOURCE_KINDS == {
        ResolutionRole.MENTION: ItemKind.ENTITY_MENTION,
        ResolutionRole.CLAIMED_BY: ItemKind.REPORTED_CLAIM,
        ResolutionRole.ABOUT: ItemKind.OBSERVATION,
    }
    assert set(ROLE_SOURCE_KINDS) == set(ResolutionRole)
