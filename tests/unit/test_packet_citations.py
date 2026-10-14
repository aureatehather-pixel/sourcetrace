"""The citation validator, including both failures it was built from.

Two observed failures, two checks, and this file is organised around them:

- **Fabrication** — the transposed UUID from the fair-context synthesis
  experiment. Caught by id membership.
- **Misattribution** — a *real* id bound to a claim its item does not make,
  from the first closed-loop run. Not caught by membership, which is why quote
  binding exists. Reproduced here with the actual ids from that run.

Most of the rest is about what the validator must *not* do. Rejection is only
useful if nothing quietly repairs, drops or reinterprets a bad citation, and if
the binding cannot be bypassed by simply omitting it — so both are asserted
directly. The strictness tests matter as much as the acceptance tests: a
validator that accepts an altered quote is not checking anything.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest

from intelligence_engine.extraction.schemas import ItemKind
from intelligence_engine.packet.citations import (
    MIN_CITATION_QUOTE_LENGTH,
    extract_bound_citations,
    extract_cited_ids,
    packet_handles,
    unbindable_items,
    validate_bound_answer,
    validate_citations,
)
from intelligence_engine.packet.errors import InvalidCitation
from intelligence_engine.packet.render import render_packet
from intelligence_engine.packet.schemas import EvidencePacket, PacketItem, PacketSource

# --- the fabrication case, from the synthesis experiment ------------------

# `REAL` is what the packet held; `TRANSPOSED` is what the model wrote —
# `94b3` became `9b4b`.
REAL_ID = uuid.UUID("5995bdaf-94b3-4e9f-9a9b-4605a0e2caac")
TRANSPOSED_ID = "5995bdaf-9b4b-4e9f-9a9b-4605a0e2caac"

# --- the misattribution case, from the first closed-loop run --------------

# The model cited FIRST_HACKATHON_ID and described it as the "effectively zero"
# observation. That content is ZERO_CONTRIBUTION_ID.
FIRST_HACKATHON_ID = uuid.UUID("e11a3fd5-b403-4134-b6b0-519e0a91292c")
FIRST_HACKATHON_QUOTE = "This was my first hackathon, Emberline DevGuild Lab 1."
ZERO_CONTRIBUTION_ID = uuid.UUID("8609f6d6-c976-4bca-abe5-6f58715d6c47")
ZERO_CONTRIBUTION_QUOTE = "My direct engineering contribution was effectively zero."

OTHER_ID = uuid.UUID("f3e88dfd-4e54-4049-8f4a-af0f34d5778d")
EVIDENCE_ID = uuid.UUID("0df5980f-cbe6-4e99-881e-8e135f77c970")

DEFAULT_QUOTE = "a verbatim quote from the stored note"


def _item(item_id: uuid.UUID, quote: str = DEFAULT_QUOTE) -> PacketItem:
    return PacketItem(
        item_id=item_id,
        kind=ItemKind.IDEA,
        text="the item's statement",
        stance=None,
        claimed_by=None,
        subject=None,
        attributes=(),
        grounding_quote=quote,
        extraction_item_position=0,
        extraction_run_id=uuid.UUID("11111111-1111-4111-8111-111111111111"),
        evidence_id=EVIDENCE_ID,
        source_label="a note",
        captured_at=None,
        entities=(),
    )


def _packet(*items: PacketItem) -> EvidencePacket:
    return EvidencePacket(
        items=items,
        sources=(
            PacketSource(
                evidence_id=EVIDENCE_ID,
                source_type=None,
                source_label="a note",
                source_url=None,
                captured_at=None,
                created_at=dt.datetime(2026, 8, 22, tzinfo=dt.UTC),
                item_count=len(items),
            ),
        ),
        total_items_available=len(items),
        truncated=False,
    )


def _cite(item_id: object, quote: str) -> str:
    return f'[{item_id} "{quote}"]'


def _h(packet: EvidencePacket, item_id: uuid.UUID) -> str:
    """The handle this packet gives an item. See `_cite_in`."""
    for handle, resolved in packet_handles(packet).items():
        if resolved == item_id:
            return handle
    raise AssertionError(f"{item_id} is not in this packet, so it has no handle")


def _cite_in(packet: EvidencePacket, item_id: uuid.UUID, quote: str) -> str:
    """Cite an item by whatever handle *this* packet gives it.

    These tests reason about items by id, because that is what the assertions
    are about. An answer can only name them by handle, because a handle is all
    the packet prints. This keeps the two in step without hard-coding a position
    into every test, and it fails loudly rather than silently citing the wrong
    row if an item is not in the packet at all.
    """
    for handle, resolved in packet_handles(packet).items():
        if resolved == item_id:
            return _cite(handle, quote)
    raise AssertionError(f"{item_id} is not in this packet, so it has no handle")


# --- failure 1: fabrication ------------------------------------------------


def test_transposed_uuid_from_the_experiment_is_rejected() -> None:
    """A well-formed id for an item that does not exist. The laundering case.

    **This now rejects one step earlier than it was written to.** A uuid is not
    a citable identifier at all since the packet stopped printing uuids, so this
    never reaches the membership check it originally exercised. The fabrication
    guarantee itself is unchanged and is exercised through the identifier an
    answer can actually use — see
    `test_packet_handles.py::test_3_a_handle_that_names_no_item_is_rejected`.
    Kept here because the string is the real one from the experiment, and
    because it pins that a fabricated uuid never becomes acceptable.
    """
    packet = _packet(_item(REAL_ID))

    with pytest.raises(InvalidCitation) as caught:
        validate_bound_answer(_cite(TRANSPOSED_ID, DEFAULT_QUOTE), packet)

    assert TRANSPOSED_ID in str(caught.value)


def test_the_transposed_id_is_not_repaired_to_the_real_one() -> None:
    """No nearest-match, no edit distance, no inference of intent.

    The two ids differ by one transposed hex group out of thirty-two characters.
    Any similarity heuristic loose enough to be useful would accept this, and
    accepting it is exactly how a fabricated citation becomes a trusted one.
    """
    packet = _packet(_item(REAL_ID))

    with pytest.raises(InvalidCitation) as caught:
        validate_bound_answer(_cite(TRANSPOSED_ID, DEFAULT_QUOTE), packet)

    assert str(REAL_ID) not in str(caught.value)


def test_a_whole_answer_is_rejected_when_one_citation_is_invalid() -> None:
    """Whole-answer rejection, exactly as `bind_results` rejects a whole batch.

    Keeping the valid citations and dropping the invalid one would leave the
    sentence it supported still standing, reading as evidence-backed with its
    support silently removed.
    """
    packet = _packet(_item(REAL_ID), _item(OTHER_ID))
    answer = (
        f"Ownership was zero {_cite_in(packet, REAL_ID, DEFAULT_QUOTE)} and the fix "
        f"follows {_cite('E900', DEFAULT_QUOTE)} on the record "
        f"{_cite_in(packet, OTHER_ID, DEFAULT_QUOTE)}."
    )

    with pytest.raises(InvalidCitation):
        validate_bound_answer(answer, packet)


# --- failure 2: misattribution, the reason binding exists ------------------


def test_the_closed_loop_misattribution_is_now_caught() -> None:
    """The exact failure from the first real run, reproduced with its own ids.

    Both ids are genuine and both items are in the packet, so id membership
    passes. The quote is what fails: the model bound the "effectively zero"
    content to the id of "this was my first hackathon".
    """
    packet = _packet(
        _item(FIRST_HACKATHON_ID, FIRST_HACKATHON_QUOTE),
        _item(ZERO_CONTRIBUTION_ID, ZERO_CONTRIBUTION_QUOTE),
    )
    answer = _cite_in(
        packet, FIRST_HACKATHON_ID, "My direct engineering contribution was effectively zero."
    )

    with pytest.raises(InvalidCitation) as caught:
        validate_bound_answer(answer, packet)

    assert "does not appear in that item's grounding quote" in str(caught.value)


def test_membership_alone_would_have_passed_that_same_answer() -> None:
    """Why binding was needed. Not a hypothetical: this is the earlier gate.

    The old check saw two real ids and accepted. The regression is that this
    stays true — the weaker gate is still exactly as weak, so nobody can mistake
    it for the strong one.
    """
    packet = _packet(
        _item(FIRST_HACKATHON_ID, FIRST_HACKATHON_QUOTE),
        _item(ZERO_CONTRIBUTION_ID, ZERO_CONTRIBUTION_QUOTE),
    )

    audit = validate_citations([str(FIRST_HACKATHON_ID)], packet)

    assert audit.cited == (FIRST_HACKATHON_ID,)


def test_the_rejection_names_the_item_the_quote_actually_belongs_to() -> None:
    """A diagnostic, not a repair: it never turns a rejection into acceptance."""
    packet = _packet(
        _item(FIRST_HACKATHON_ID, FIRST_HACKATHON_QUOTE),
        _item(ZERO_CONTRIBUTION_ID, ZERO_CONTRIBUTION_QUOTE),
    )
    answer = _cite_in(
        packet, FIRST_HACKATHON_ID, "direct engineering contribution was effectively zero"
    )

    with pytest.raises(InvalidCitation, match=str(ZERO_CONTRIBUTION_ID)):
        validate_bound_answer(answer, packet)


def test_a_quote_belonging_to_no_item_says_so() -> None:
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    with pytest.raises(InvalidCitation, match="No item in the packet has that quote"):
        validate_bound_answer(_cite_in(packet, REAL_ID, "something nobody ever wrote down"), packet)


# --- the bypass is closed --------------------------------------------------


def test_a_bare_id_is_rejected() -> None:
    """If a loose id were tolerated the binding would be optional.

    An optional check is not a check: the next answer simply omits the quote.
    """
    packet = _packet(_item(REAL_ID))

    with pytest.raises(InvalidCitation, match="without a bound quote"):
        validate_bound_answer(f"A finding [{REAL_ID}].", packet)


def test_an_id_in_brackets_with_no_quote_is_still_bare() -> None:
    """The packet's own item headers look like `[id]`. That is not a citation."""
    packet = _packet(_item(REAL_ID))

    with pytest.raises(InvalidCitation, match="without a bound quote"):
        validate_bound_answer(f"See [{REAL_ID}] above.", packet)


def test_one_bound_citation_does_not_license_a_bare_one_elsewhere() -> None:
    packet = _packet(_item(REAL_ID), _item(OTHER_ID))
    answer = f"First {_cite_in(packet, REAL_ID, DEFAULT_QUOTE)}, then also {OTHER_ID}."

    with pytest.raises(InvalidCitation, match=str(OTHER_ID)):
        validate_bound_answer(answer, packet)


# --- what binding accepts --------------------------------------------------


def test_the_full_grounding_quote_binds() -> None:
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    audit = validate_bound_answer(_cite_in(packet, REAL_ID, FIRST_HACKATHON_QUOTE), packet)

    assert audit.cited == (REAL_ID,)
    assert audit.bound[0].quote == FIRST_HACKATHON_QUOTE


def test_an_unbroken_fragment_of_the_quote_binds() -> None:
    """A model should not have to reproduce a 300-character quote to cite it."""
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    audit = validate_bound_answer(_cite_in(packet, REAL_ID, "Emberline DevGuild Lab 1"), packet)

    assert audit.cited == (REAL_ID,)


def test_two_items_sharing_one_grounding_quote_are_neither_citable() -> None:
    """The reversal, stated as a test rather than left in a docstring.

    One sentence can yield both a `project` and an `observation`, and in the live
    corpus it does — `d62679b1` and `595ab32d` share "Our team built Fernvault
    and placed 5th." verbatim. This used to be *accepted*, on the reasoning that
    a uniqueness requirement would make both permanently uncitable.

    Both halves of that are true; the conclusion was not. Accepting the citation
    never made the two items distinguishable — it only declined to check, so a
    fragment naming either one passed under the other's id. Uncitable and
    reported beats citable and meaningless, so both are now rejected.
    """
    shared = "Our team built Fernvault and placed 5th."
    packet = _packet(_item(REAL_ID, shared), _item(OTHER_ID, shared))

    with pytest.raises(InvalidCitation, match="cannot be uniquely cited"):
        validate_bound_answer(_cite_in(packet, REAL_ID, shared), packet)
    with pytest.raises(InvalidCitation, match="cannot be uniquely cited"):
        validate_bound_answer(_cite_in(packet, OTHER_ID, shared), packet)


# --- failure 3: a real quote that identifies nothing -----------------------

# The collision shape measured on the Event 2 corpus: many items quote a line
# beginning "**Qualification:**", so the 8-character fragment "**Qualif" is
# contained in the grounding quote of each of them and identifies none.
QUALIFICATION_A = "**Qualification:** Details of employer and role were not given."
QUALIFICATION_B = "**Qualification:** User also questioned that some firms require LeetCode."


def test_a_fragment_shared_with_another_item_is_rejected() -> None:
    """The measured collision, reproduced.

    Containment holds — "**Qualif" really is inside the cited item's own quote —
    and the citation is still worthless, because it is inside the other item's
    quote too. This is what the containment-only check passed.
    """
    packet = _packet(_item(REAL_ID, QUALIFICATION_A), _item(OTHER_ID, QUALIFICATION_B))

    with pytest.raises(InvalidCitation, match="also appears in 1 other item"):
        validate_bound_answer(_cite_in(packet, REAL_ID, "**Qualif"), packet)


def test_the_misattribution_survives_a_shared_fragment_and_is_now_caught() -> None:
    """The live misattribution the falsifier constructed, end to end.

    Prose describing item B, carrying item A's id, bound to a fragment that
    appears in both quotes. Every earlier check passes: the id is real, the id is
    in the packet, the fragment is long enough, and the fragment is verbatim
    inside A's own grounding quote. Only uniqueness rejects it.
    """
    packet = _packet(_item(REAL_ID, QUALIFICATION_A), _item(OTHER_ID, QUALIFICATION_B))
    answer = (
        "The user questioned whether some firms require LeetCode "
        f'[{_h(packet, REAL_ID)} "**Qualification:**"].'
    )

    with pytest.raises(InvalidCitation, match="also appears in"):
        validate_bound_answer(answer, packet)


def test_containment_alone_would_have_passed_that_same_answer() -> None:
    """The counterfactual, so the test above is known to be testing uniqueness.

    Without this, a later refactor could make the rejection come from the
    containment check and nothing would notice that the new guarantee had gone.
    """
    fragment = "**Qualification:**"

    assert fragment in QUALIFICATION_A
    assert fragment in QUALIFICATION_B
    assert len(fragment) >= MIN_CITATION_QUOTE_LENGTH


def test_a_longer_fragment_that_is_unique_still_binds() -> None:
    """The repair the rejection names actually works.

    A rule that rejected the collision but left no way to cite either item would
    be a rule nobody could satisfy.
    """
    packet = _packet(_item(REAL_ID, QUALIFICATION_A), _item(OTHER_ID, QUALIFICATION_B))

    audit = validate_bound_answer(_cite_in(packet, REAL_ID, "Details of employer and role"), packet)

    assert audit.cited == (REAL_ID,)


def test_a_nested_quote_leaves_the_containing_item_citable() -> None:
    """Containment is one-way, and the rejection is too.

    `OTHER_ID`'s quote sits inside `REAL_ID`'s, so `OTHER_ID` can never be
    identified. `REAL_ID` still can, using any stretch of its quote that falls
    outside the nested one — and rejecting it as well would lose a citable item
    to its neighbour's ambiguity.
    """
    inner = "placed 5th in the finals"
    outer = f"Our team built Fernvault and {inner} after a single weekend."
    packet = _packet(_item(REAL_ID, outer), _item(OTHER_ID, inner))

    audit = validate_bound_answer(_cite_in(packet, REAL_ID, "Our team built Fernvault"), packet)
    assert audit.cited == (REAL_ID,)

    with pytest.raises(InvalidCitation, match="cannot be uniquely cited"):
        validate_bound_answer(_cite_in(packet, OTHER_ID, inner), packet)


def test_the_rejection_distinguishes_a_fixable_collision_from_an_unfixable_one() -> None:
    """Two different findings, and the human reading them needs to tell them apart.

    One says "quote more of it"; the other says "this item cannot be cited at
    all, and that is the evidence's doing, not yours". Collapsing them into one
    message would send a reader looking for a longer fragment that does not
    exist.
    """
    fixable = _packet(_item(REAL_ID, QUALIFICATION_A), _item(OTHER_ID, QUALIFICATION_B))
    with pytest.raises(InvalidCitation, match="quote a longer stretch"):
        validate_bound_answer(_cite_in(fixable, REAL_ID, "**Qualif"), fixable)

    shared = "Our team built Fernvault and placed 5th."
    unfixable = _packet(_item(REAL_ID, shared), _item(OTHER_ID, shared))
    with pytest.raises(InvalidCitation, match="property of the stored evidence"):
        validate_bound_answer(_cite_in(unfixable, REAL_ID, shared), unfixable)


def test_uniqueness_is_asked_against_the_same_normalised_text_as_containment() -> None:
    """Our own line wrapping must not turn an ambiguous citation into a clean one.

    The renderer wraps long quotes, so a copied fragment arrives with inserted
    whitespace. `_squeeze` is applied to both sides for containment; if it were
    not applied on the uniqueness side, a wrapped fragment would silently stop
    colliding and pass.
    """
    packet = _packet(_item(REAL_ID, QUALIFICATION_A), _item(OTHER_ID, QUALIFICATION_B))
    wrapped = "**Qualification:**\n        "

    with pytest.raises(InvalidCitation, match="also appears in"):
        validate_bound_answer(_cite_in(packet, REAL_ID, wrapped), packet)


def test_unbindable_items_names_the_condition_without_being_asked_for_an_answer() -> None:
    """The inspection path: which items are ambiguous, before anything is written."""
    shared = "Our team built Fernvault and placed 5th."
    nested = "placed 5th"
    packet = _packet(
        _item(REAL_ID, shared),
        _item(OTHER_ID, shared),
        _item(FIRST_HACKATHON_ID, nested),
        _item(ZERO_CONTRIBUTION_ID, FIRST_HACKATHON_QUOTE),
    )

    assert unbindable_items(packet) == (REAL_ID, OTHER_ID, FIRST_HACKATHON_ID)


def test_a_packet_of_one_item_has_nothing_to_be_ambiguous_against() -> None:
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    assert unbindable_items(packet) == ()
    audit = validate_bound_answer(_cite_in(packet, REAL_ID, FIRST_HACKATHON_QUOTE), packet)
    assert audit.cited == (REAL_ID,)


def test_curly_quotation_marks_are_accepted_as_delimiters() -> None:
    """Syntax tolerance about our own punctuation, never about the content."""
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    audit = validate_bound_answer(f"[{_h(packet, REAL_ID)} “Emberline DevGuild Lab 1”]", packet)

    assert audit.cited == (REAL_ID,)


def test_angle_brackets_echoed_from_the_contract_are_accepted() -> None:
    """Observed on the first rerun, not anticipated.

    The packet states the shape as `[<id> "<verbatim quote>"]`, and a model
    copied the placeholder's angle brackets straight through. The id and the
    quote inside were both exact, so only our own notation had leaked; rejecting
    it would have failed a substantively correct citation on a defect of our
    instructions. This is the exact string that run produced.
    """
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    audit = validate_bound_answer(f'[<{_h(packet, REAL_ID)}> "Emberline DevGuild Lab 1"]', packet)

    assert audit.cited == (REAL_ID,)


def test_angle_brackets_do_not_loosen_the_binding() -> None:
    """The tolerance is about our notation, never about the quoted content."""
    packet = _packet(
        _item(FIRST_HACKATHON_ID, FIRST_HACKATHON_QUOTE),
        _item(ZERO_CONTRIBUTION_ID, ZERO_CONTRIBUTION_QUOTE),
    )

    with pytest.raises(InvalidCitation, match=str(ZERO_CONTRIBUTION_ID)):
        validate_bound_answer(
            f'[<{_h(packet, FIRST_HACKATHON_ID)}> "My direct engineering contribution"]', packet
        )


def test_a_separator_between_the_id_and_the_quote_is_accepted() -> None:
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    for separator in (":", ",", ";", "-", chr(0x2013), chr(0x2014)):
        audit = validate_bound_answer(
            f'[{_h(packet, REAL_ID)} {separator} "Emberline DevGuild Lab 1"]', packet
        )
        assert audit.cited == (REAL_ID,)


def test_a_quote_wrapped_by_our_own_renderer_still_binds() -> None:
    """The one tolerance, exercised against the renderer that motivates it.

    The quote is copied out of the *rendered* packet, line breaks and indentation
    included, exactly as a model reading the rendered text would produce it.
    Refusing this would fail a correct citation for a defect of our formatting.
    """
    long_quote = (
        "For future hackathons, placement alone should not be sufficient evidence "
        "of my own technical capability. My level of actual technical ownership "
        "should matter."
    )
    packet = _packet(_item(REAL_ID, long_quote))
    rendered = render_packet(packet)
    wrapped = rendered.split('quote: "', 1)[1].rsplit('"', 1)[0]

    assert "\n" in wrapped, "the renderer did not wrap; this test would prove nothing"

    audit = validate_bound_answer(_cite_in(packet, REAL_ID, wrapped), packet)

    assert audit.cited == (REAL_ID,)


# --- what binding refuses: strictness, proven ------------------------------


def test_a_single_altered_character_is_rejected() -> None:
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    with pytest.raises(InvalidCitation):
        validate_bound_answer(
            _cite_in(packet, REAL_ID, "This was my first hackathon, Emberline DevGuild Lab 2"),
            packet,
        )


def test_a_change_of_case_is_rejected() -> None:
    """Whitespace is normalised. Nothing else is."""
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    with pytest.raises(InvalidCitation):
        validate_bound_answer(_cite_in(packet, REAL_ID, "emberline devguild lab 1"), packet)


def test_dropped_punctuation_is_rejected() -> None:
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    with pytest.raises(InvalidCitation):
        validate_bound_answer(
            _cite_in(packet, REAL_ID, "This was my first hackathon Emberline"), packet
        )


def test_a_paraphrase_is_rejected() -> None:
    """The failure mode binding exists for: plausible words, not the evidence's."""
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    with pytest.raises(InvalidCitation):
        validate_bound_answer(_cite_in(packet, REAL_ID, "my very first hackathon event"), packet)


def test_a_fragment_too_short_to_identify_an_item_is_rejected() -> None:
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))
    short = FIRST_HACKATHON_QUOTE[: MIN_CITATION_QUOTE_LENGTH - 1]

    with pytest.raises(InvalidCitation, match="below the"):
        validate_bound_answer(_cite_in(packet, REAL_ID, short), packet)


def test_an_empty_quote_is_rejected() -> None:
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    with pytest.raises(InvalidCitation, match="below the"):
        validate_bound_answer(_cite_in(packet, REAL_ID, ""), packet)


# --- ordinary behaviour ----------------------------------------------------


def test_a_valid_answer_is_audited() -> None:
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE), _item(OTHER_ID))

    audit = validate_bound_answer(
        f"A finding {_cite_in(packet, REAL_ID, 'Emberline DevGuild Lab 1')}.", packet
    )

    assert audit.cited == (REAL_ID,)
    assert audit.uncited == (OTHER_ID,)
    assert audit.cited_count == 1
    assert len(audit.bound) == 1


def test_an_answer_citing_nothing_is_valid_by_default() -> None:
    """ "No evidence here bears on the question" is a correct answer.

    Requiring a citation would pressure the next answer into producing one.
    """
    packet = _packet(_item(REAL_ID))

    audit = validate_bound_answer("Nothing in this packet bears on the question.", packet)

    assert audit.cited == ()
    assert audit.uncited == (REAL_ID,)


def test_require_any_is_available_for_callers_that_demand_support() -> None:
    packet = _packet(_item(REAL_ID))

    with pytest.raises(InvalidCitation):
        validate_bound_answer("A confident conclusion, unsupported.", packet, require_any=True)


def test_citing_the_same_item_twice_is_ordinary_prose() -> None:
    """Unlike M4, where a pair answered twice means the model lost its batch."""
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))
    answer = (
        f"As {_cite_in(packet, REAL_ID, 'This was my first hackathon')} shows, "
        f"and again {_cite_in(packet, REAL_ID, 'Emberline DevGuild Lab 1')}."
    )

    audit = validate_bound_answer(answer, packet)

    assert audit.cited == (REAL_ID,)
    assert len(audit.bound) == 2


def test_both_bindings_must_hold_when_one_item_is_cited_twice() -> None:
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))
    answer = (
        f"{_cite_in(packet, REAL_ID, 'This was my first hackathon')} and "
        f"{_cite_in(packet, REAL_ID, 'a quote from somewhere else entirely')}"
    )

    with pytest.raises(InvalidCitation):
        validate_bound_answer(answer, packet)


def test_an_upper_cased_uuid_is_rejected_like_any_other_uuid() -> None:
    """Case tolerance used to matter because uuids were citable. They are not.

    A real item id, spelled perfectly, upper-cased or not, is still a uuid, and a
    uuid cannot have been read off a packet that prints none. The tolerance would
    only widen a door that is now closed.
    """
    packet = _packet(_item(REAL_ID, FIRST_HACKATHON_QUOTE))

    with pytest.raises(InvalidCitation, match="uuid instead of a packet handle"):
        validate_bound_answer(_cite(str(REAL_ID).upper(), "Emberline DevGuild Lab 1"), packet)


def test_an_empty_packet_accepts_no_citation_at_all() -> None:
    """No items means no handles, so every handle names nothing."""
    with pytest.raises(InvalidCitation, match="0 item"):
        validate_bound_answer(_cite("E1", DEFAULT_QUOTE), _packet())


# --- finding the citations in the first place ------------------------------


def test_bound_citations_are_extracted_in_order() -> None:
    text = f"{_cite(OTHER_ID, 'first quote here')} then {_cite(REAL_ID, 'second quote here')}"

    found = extract_bound_citations(text)

    assert [citation.item_id for citation in found] == [OTHER_ID, REAL_ID]
    assert [citation.quote for citation in found] == ["first quote here", "second quote here"]


def test_ids_are_found_in_prose_and_punctuation_and_kept_in_order() -> None:
    text = f"First [{OTHER_ID}], then ({REAL_ID}) and finally {OTHER_ID} again."

    assert extract_cited_ids(text) == (str(OTHER_ID), str(REAL_ID))


def test_a_longer_hex_run_containing_a_uuid_is_not_a_citation() -> None:
    """Word boundaries: a hash that happens to embed a UUID shape is not an id."""
    assert extract_cited_ids(f"{REAL_ID}deadbeef") == ()
