"""Rendering a packet: deterministic, complete, and honest about absence.

The renderer is the provider-independent deliverable, so its guarantees are
asserted here rather than assumed: the same packet renders identically, a field
the evidence did not supply prints nothing at all, and `claimed_by` and `subject`
are never collapsed into one "attributed to" line.

Truncation and emptiness get their own tests because both are cases where a
renderer can lie by omission — a packet that quietly stopped at a limit, or an
empty one that reads like a failure rather than an answer.
"""

from __future__ import annotations

import datetime as dt
import uuid

from intelligence_engine.entities.schemas import ResolutionRole
from intelligence_engine.extraction.schemas import ItemKind
from intelligence_engine.packet.render import render_packet
from intelligence_engine.packet.schemas import (
    EntityContext,
    EvidencePacket,
    PacketItem,
    PacketSource,
)

EVIDENCE_ID = uuid.UUID("0df5980f-cbe6-4e99-881e-8e135f77c970")
RUN_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
ENTITY_ID = uuid.UUID("c56eabb2-4635-4026-8aa5-3f7fda84e5e9")


def _item(
    *,
    item_id: uuid.UUID | None = None,
    kind: ItemKind = ItemKind.IDEA,
    text: str = "start from a real project",
    stance: str | None = None,
    claimed_by: str | None = None,
    subject: str | None = None,
    attributes: tuple[tuple[str, str], ...] = (),
    entities: tuple[EntityContext, ...] = (),
    source_label: str | None = "a note",
    captured_at: dt.datetime | None = None,
) -> PacketItem:
    return PacketItem(
        item_id=item_id or uuid.uuid4(),
        kind=kind,
        text=text,
        stance=stance,
        claimed_by=claimed_by,
        subject=subject,
        attributes=attributes,
        grounding_quote="a verbatim quote from the note",
        extraction_item_position=0,
        extraction_run_id=RUN_ID,
        evidence_id=EVIDENCE_ID,
        source_label=source_label,
        captured_at=captured_at,
        entities=entities,
    )


def _packet(
    *items: PacketItem, total: int | None = None, truncated: bool = False
) -> EvidencePacket:
    return EvidencePacket(
        items=items,
        sources=(
            PacketSource(
                evidence_id=EVIDENCE_ID,
                source_type="conversation_note",
                source_label="a note",
                source_url=None,
                captured_at=None,
                created_at=dt.datetime(2026, 8, 22, tzinfo=dt.UTC),
                item_count=len(items),
            ),
        )
        if items
        else (),
        total_items_available=len(items) if total is None else total,
        truncated=truncated,
    )


def test_the_same_packet_renders_identically_every_time() -> None:
    packet = _packet(_item(), _item(kind=ItemKind.EVENT, text="DevGuild Lab 1"))

    assert render_packet(packet) == render_packet(packet)


def test_every_item_carries_a_handle_and_no_uuid_to_mis_transcribe() -> None:
    """Handles are the citation token, and the uuid is deliberately not printed.

    Answers corrupted uuids in real runs - a real id with one extra character
    appended, and evidence ids written where item ids belonged. An identifier the
    answer never sees cannot be copied wrong.
    """
    first, second = _item(), _item()

    rendered = render_packet(_packet(first, second))

    assert "[E1]" in rendered
    assert "[E2]" in rendered
    assert str(first.item_id) not in rendered
    assert str(second.item_id) not in rendered


def test_the_citation_rule_travels_with_the_packet() -> None:
    """A packet pasted anywhere carries its own contract, not the caller's."""
    rendered = render_packet(_packet(_item()))

    assert 'like this:  [E17 "verbatim quote"]' in rendered
    assert "rejects the whole answer" in rendered


def test_the_contract_states_the_uniqueness_rule_it_will_be_judged_by() -> None:
    """Uniqueness rejects answers, so the packet says so before one is written.

    Handles changed how an identifier is spelled. They did not change what is
    guaranteed, and a contract that advertised only the easy half would be
    setting an answer up to fail on the hard one.
    """
    # Collapsed, because the contract is wrapped to the packet width and a
    # phrase may straddle a line break. The rule is what matters, not where the
    # renderer chose to fold it.
    rendered = " ".join(render_packet(_packet(_item())).split())

    assert "identifies neither" in rendered
    assert "quote a longer stretch that appears only in the item you are citing" in rendered


def test_the_contract_states_no_example_id_a_model_could_echo() -> None:
    """An example UUID in the instructions would be cited back as a real item.

    No uuid is printed anywhere now - not in the contract and not on an item -
    so there is nothing uuid-shaped for an answer to echo back.
    """
    from intelligence_engine.packet.citations import extract_cited_ids

    rendered = render_packet(_packet())

    assert extract_cited_ids(rendered) == ()


def test_a_speaker_and_a_subject_are_rendered_as_different_statements() -> None:
    said = _item(kind=ItemKind.REPORTED_CLAIM, claimed_by="Aisyah", stance="asserted")
    about = _item(kind=ItemKind.OBSERVATION, subject="Aisyah")

    rendered = render_packet(_packet(said, about))

    assert "asserted by (as worded): Aisyah" in rendered
    assert "observed about (as worded): Aisyah" in rendered


def test_a_field_the_evidence_did_not_supply_prints_nothing() -> None:
    """Never "unknown" — that would be a claim the evidence did not make."""
    rendered = render_packet(_packet(_item(claimed_by=None, subject=None, stance=None)))

    assert "asserted by" not in rendered
    assert "observed about" not in rendered
    assert "stance" not in rendered


def test_entity_context_names_the_role_it_was_asserted_under() -> None:
    """Never collapsed to "linked to": `claimed_by` and `about` differ."""
    context = EntityContext(
        entity_id=ENTITY_ID,
        canonical_name="Zi Lan",
        role=ResolutionRole.ABOUT,
        resolution_id=uuid.uuid4(),
    )

    rendered = render_packet(_packet(_item(entities=(context,))))

    assert "entity: Zi Lan (about," in rendered


def test_a_complete_packet_says_so() -> None:
    assert "Complete: every stored grounded item" in render_packet(_packet(_item()))


def test_a_truncated_packet_says_what_it_left_out() -> None:
    """Silence here would be an invisible compression — the failure to prevent."""
    rendered = render_packet(_packet(_item(), total=46, truncated=True))

    assert "TRUNCATED" in rendered
    assert "45 stored item(s) are not shown" in rendered
    assert "1 of 46" in rendered


def test_an_empty_packet_reads_as_an_answer_not_a_failure() -> None:
    rendered = render_packet(_packet())

    assert "empty, not filtered" in rendered


def test_the_grounding_quote_travels_with_every_item() -> None:
    rendered = render_packet(_packet(_item()))

    assert "a verbatim quote from the note" in rendered
