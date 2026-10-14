"""Packet-local handles, and the adversarial cases they were checked against.

Handles exist for one measured reason: answers corrupted UUIDs. A real id was
written with one extra character appended, and evidence ids were written where
item ids belonged. A handle has no interior to corrupt.

**Handles are ergonomics. They are not a guarantee, and this file is mostly
about proving they did not become one.** The design that was rejected —
resolve the handle, check the quote sits inside the resolved item, and stop —
is the pre-fix rule with a shorter identifier: nothing in it bounds how many
*other* items contain the fragment, so prose about item B carrying item A's
handle and a fragment both quotes share would pass, identifying neither.

So the handle is resolved to an item id *before* any check runs, and
containment and packet-wide uniqueness then apply to the resolved item exactly
as they did to a cited UUID. The tests below assert that equivalence directly,
including the two cases where uniqueness must still reject.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest

from intelligence_engine.extraction.schemas import ItemKind
from intelligence_engine.packet.citations import (
    handle_for_index,
    packet_handles,
    unbindable_items,
    validate_bound_answer,
)
from intelligence_engine.packet.errors import InvalidCitation
from intelligence_engine.packet.render import render_packet
from intelligence_engine.packet.schemas import EvidencePacket, PacketItem, PacketSource

EVIDENCE_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")
OTHER_EVIDENCE_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")


def _item(
    item_id: uuid.UUID,
    quote: str,
    *,
    kind: ItemKind = ItemKind.IDEA,
    evidence_id: uuid.UUID = EVIDENCE_ID,
) -> PacketItem:
    return PacketItem(
        item_id=item_id,
        kind=kind,
        text="the item's statement",
        stance=None,
        claimed_by=None,
        subject=None,
        attributes=(),
        grounding_quote=quote,
        extraction_item_position=0,
        extraction_run_id=uuid.UUID("11111111-1111-4111-8111-111111111111"),
        evidence_id=evidence_id,
        source_label="a note",
        captured_at=None,
        entities=(),
    )


def _packet(*items: PacketItem) -> EvidencePacket:
    evidence_ids: list[uuid.UUID] = []
    for item in items:
        if item.evidence_id not in evidence_ids:
            evidence_ids.append(item.evidence_id)
    return EvidencePacket(
        items=items,
        sources=tuple(
            PacketSource(
                evidence_id=evidence_id,
                source_type=None,
                source_label="a note",
                source_url=None,
                captured_at=None,
                created_at=dt.datetime(2026, 8, 22, tzinfo=dt.UTC),
                item_count=sum(1 for i in items if i.evidence_id == evidence_id),
            )
            for evidence_id in evidence_ids
        ),
        total_items_available=len(items),
        truncated=False,
    )


ID_A = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
ID_B = uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")

# The shared fragment is the whole point: "the reading part" sits in both
# quotes, so it identifies neither item.
QUOTE_A = "he spent six weeks on the reading part before writing anything"
QUOTE_B = "people who copy him quit during the reading part of the work"
SHARED = "the reading part"


# --- the mapping itself ---------------------------------------------------


def test_handles_are_assigned_by_packet_order_and_are_total() -> None:
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    assert packet_handles(packet) == {"E1": ID_A, "E2": ID_B}


def test_the_same_packet_yields_the_same_handles_every_time() -> None:
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    assert packet_handles(packet) == packet_handles(packet)


def test_handle_numbering_starts_at_one() -> None:
    """Zero-based internally, one-based to the reader. E0 never exists."""
    assert handle_for_index(0) == "E1"
    assert handle_for_index(16) == "E17"


def test_the_renderer_and_the_resolver_agree_on_what_a_handle_means() -> None:
    """Display groups items by source; handles follow packet order regardless.

    If these two ever disagreed, E2 would mean one row to the reader and another
    to the validator, and every citation of it would be judged against the wrong
    quote. Two sources, interleaved, so grouping actually reorders the display.
    """
    first = _item(ID_A, QUOTE_A)
    second = _item(ID_B, QUOTE_B, evidence_id=OTHER_EVIDENCE_ID)
    packet = _packet(first, second)

    rendered = render_packet(packet)
    handles = packet_handles(packet)

    for handle, item_id in handles.items():
        quote = next(i.grounding_quote for i in packet.items if i.item_id == item_id)
        block = rendered.split(f"[{handle}]")[1]
        assert quote[:30] in " ".join(block.split())


# --- the eight adversarial cases ------------------------------------------


def test_1_correct_handle_and_correct_unique_quote_is_accepted() -> None:
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    audit = validate_bound_answer('[E1 "six weeks on the reading part"]', packet)

    assert audit.cited == (ID_A,)


def test_2_correct_handle_with_a_quote_from_another_item_is_rejected() -> None:
    """Containment. The handle resolves; the fragment is not in that item."""
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    with pytest.raises(InvalidCitation, match="does not appear in that item"):
        validate_bound_answer('[E1 "people who copy him quit"]', packet)


def test_2b_the_shared_fragment_collision_is_still_rejected() -> None:
    """**The case the rejected design would have let through.**

    Prose about item B, item A's handle, and a fragment present in both quotes.
    Containment alone passes it. Uniqueness is what refuses it, and uniqueness
    is applied to the resolved item exactly as it was to a cited uuid.
    """
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    with pytest.raises(InvalidCitation, match="also appears in"):
        validate_bound_answer(f'[E1 "{SHARED}"]', packet)


def test_3_a_handle_that_names_no_item_is_rejected() -> None:
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    with pytest.raises(InvalidCitation, match="do not name an item"):
        validate_bound_answer('[E900 "six weeks on the reading part"]', packet)


def test_3b_the_rejection_says_which_handles_the_packet_actually_has() -> None:
    """A rejection a reader cannot act on gets switched off."""
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    with pytest.raises(InvalidCitation, match=r"E1 to E2"):
        validate_bound_answer('[E900 "six weeks on the reading part"]', packet)


def test_4_a_malformed_handle_does_not_silently_become_a_citation() -> None:
    """`E1x` is not a handle. It must not be read as `E1` and accepted."""
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    with pytest.raises(InvalidCitation):
        validate_bound_answer('[E1x "six weeks on the reading part"]', packet)


def test_5_two_items_with_identical_quotes_are_still_both_uncitable() -> None:
    """Handles can *name* them apart. Nothing can *show* them apart.

    This is the duplicate-quote case, and it is deliberately still a rejection.
    Accepting it would let an answer treat one row as another whenever their
    quotes match — and in the real corpus every such group mixes kinds, so the
    two rows are a `reported_claim` and an `opportunity`, or an observation and
    a project. That swap is invisible to a reader and would be waved through by
    a green validator.
    """
    same = "one sentence that produced two items"
    packet = _packet(_item(ID_A, same), _item(ID_B, same))

    with pytest.raises(InvalidCitation, match="cannot be uniquely cited"):
        validate_bound_answer(f'[E1 "{same}"]', packet)

    assert set(unbindable_items(packet)) == {ID_A, ID_B}


def test_6_a_fragment_nested_in_another_items_quote_is_rejected() -> None:
    """A shorter quote wholly inside a longer one identifies neither."""
    packet = _packet(
        _item(ID_A, "the reading part"),
        _item(ID_B, "he spent six weeks on the reading part before writing"),
    )

    with pytest.raises(InvalidCitation, match="cannot be uniquely cited"):
        validate_bound_answer('[E1 "the reading part"]', packet)


def test_6b_the_uniquely_identifying_fragment_of_a_nested_pair_still_binds() -> None:
    """The other half of case 6, and the half that stops it being over-strict.

    Uniqueness rejects the *fragment*, never the item. `E2`'s quote contains
    `E1`'s whole quote, so the shared stretch identifies neither — but the part
    of `E2` that reaches outside `E1` identifies `E2` and nothing else, and it
    binds. Without this the previous test would be equally satisfied by a rule
    that simply refused both items, which is not the rule.
    """
    packet = _packet(
        _item(ID_A, "the reading part"),
        _item(ID_B, "he spent six weeks on the reading part before writing"),
    )

    audit = validate_bound_answer('[E2 "six weeks on the reading part before"]', packet)

    assert audit.cited == (ID_B,)


def test_7_a_uuid_invented_instead_of_a_handle_is_rejected() -> None:
    """No uuid is printed in the packet, so any uuid in an answer is unsourced."""
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))
    invented = "5995bdaf-9b4b-4e9f-9a9b-4605a0e2caac"

    with pytest.raises(InvalidCitation):
        validate_bound_answer(f'[{invented} "six weeks on the reading part"]', packet)


def test_7b_a_real_item_uuid_is_rejected_even_with_a_correct_unique_quote() -> None:
    """The case that used to pass, and the reason it must not.

    Everything about this citation is otherwise perfect: the uuid is a real item
    in this packet, correctly transcribed, and the fragment is verbatim in that
    item's own quote and in no other. Membership, containment and uniqueness all
    hold on their own merits.

    It is still rejected, because the packet prints no uuid anywhere. An
    identifier the model could not have read off the evidence is not evidence
    the evidence was read, and evidence of readership is the entire thing
    binding buys — see the module docstring in `citations.py`. Accepting it also
    leaves two spellings of one identifier, only one of which goes through
    handle resolution.
    """
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    with pytest.raises(InvalidCitation, match="uuid instead of a packet handle"):
        validate_bound_answer(f'[{ID_A} "six weeks on the reading part"]', packet)


def test_7c_the_uuid_rejection_says_what_to_write_instead() -> None:
    """A rejection a reader cannot act on gets switched off."""
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    with pytest.raises(InvalidCitation, match=r"E1 to E2"):
        validate_bound_answer(f'[{ID_A} "six weeks on the reading part"]', packet)


def test_8_an_evidence_id_used_as_a_handle_is_rejected() -> None:
    """Observed four times in real answers. Now structurally impossible.

    A source id is a uuid, and uuids are not handles, so the confusion cannot
    even be spelled as a citation of an item.
    """
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    with pytest.raises(InvalidCitation):
        validate_bound_answer(f'[{EVIDENCE_ID} "six weeks on the reading part"]', packet)


# --- the binding must stay non-optional -----------------------------------


def test_a_bare_handle_is_rejected_like_a_bare_id() -> None:
    """An optional check is not a check."""
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    with pytest.raises(InvalidCitation, match="without a bound quote"):
        validate_bound_answer("as E1 shows, the reading came first", packet)


def test_a_handle_shaped_word_in_prose_is_rejected_rather_than_ignored() -> None:
    """The cost of the rule above, asserted rather than left to be discovered.

    `E9` names no item in a two-item packet, and prose that wanted to write it
    for some other reason cannot. That is a real limitation, and it is preferred
    to the alternative: quietly allowing bare handles would make the binding
    optional again.
    """
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    with pytest.raises(InvalidCitation, match="without a bound quote"):
        validate_bound_answer("the E9 grade steel was irrelevant here", packet)


def test_an_altered_quote_is_still_rejected_under_a_handle() -> None:
    """Tolerances are about our syntax, never about the quoted content."""
    packet = _packet(_item(ID_A, QUOTE_A), _item(ID_B, QUOTE_B))

    with pytest.raises(InvalidCitation, match="does not appear in that item"):
        validate_bound_answer('[E1 "six weeks on the Reading part"]', packet)


def test_echoing_the_contracts_example_fails_loudly_rather_than_silently() -> None:
    """The packet's example is `[E17 "verbatim quote"]`, and a model may copy it.

    It parses, so it is judged and rejected — rather than failing to parse and
    vanishing, which is what an abstract placeholder would have done.
    """
    packet = _packet(*(_item(uuid.uuid4(), f"quote number {n} of the note") for n in range(20)))

    with pytest.raises(InvalidCitation):
        validate_bound_answer('[E17 "verbatim quote"]', packet)
