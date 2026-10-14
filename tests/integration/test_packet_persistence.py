"""The grounded evidence packet against real PostgreSQL.

Everything runs through the real migrations, the real M1-M2-M3 path and a
deterministic fake provider - no network, no credentials, which is itself part of
what is being proved: this capability works at an Anthropic balance of zero.

The file is organised around the packet's invariants:

- the six grounded kinds enter the packet, and the two naming kinds do not;
- **M4 relation eligibility has not widened** - the same session, the same rows,
  and `list_relatable_items` still returns exactly two kinds;
- provenance and grounding survive assembly intact;
- ordering is deterministic and repeatable;
- one evidence item contributes one extraction run, the newest that succeeded;
- truncation is reported rather than silent;
- unreadable stored material fails loudly rather than vanishing.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelligence_engine.entities.schemas import (
    EntityCreate,
    EntityType,
    ResolutionCreate,
    ResolutionRole,
)
from intelligence_engine.entities.service import EntityService
from intelligence_engine.evidence.schemas import EvidenceCreate
from intelligence_engine.evidence.service import EvidenceService
from intelligence_engine.extraction.fake_provider import FakeExtractionProvider
from intelligence_engine.extraction.models import ExtractionItem, ExtractionRun
from intelligence_engine.extraction.payloads import UnreadablePayload
from intelligence_engine.extraction.schemas import ItemKind, RunStatus
from intelligence_engine.extraction.service import ExtractionService
from intelligence_engine.intelligence.repository import IntelligenceRepository
from intelligence_engine.packet.citations import (
    handle_for_index,
    packet_handles,
    validate_bound_answer,
)
from intelligence_engine.packet.errors import InvalidCitation
from intelligence_engine.packet.render import render_packet
from intelligence_engine.packet.schemas import PACKET_ITEM_KINDS, EvidencePacket
from intelligence_engine.packet.service import PacketService
from tests.packet_fixtures import (
    EVIDENCE_CONTENT,
    QUOTE_IDEA,
    SECOND_CONTENT,
    all_kinds_payload,
    reduced_payload,
    second_payload,
)

pytestmark = pytest.mark.integration

SessionFactory = async_sessionmaker[AsyncSession]

#: The four kinds the packet exists to reach and M4 must keep refusing.
PACKET_ONLY = {ItemKind.IDEA, ItemKind.OPPORTUNITY, ItemKind.PROJECT, ItemKind.EVENT}


# --- helpers --------------------------------------------------------------


async def _ingest(
    factory: SessionFactory, *, content: str, label: str, payload: dict[str, Any]
) -> uuid.UUID:
    """The real M1 -> M2 path, so the packet reads genuinely grounded material."""
    async with factory() as session:
        evidence = await EvidenceService(session).create(
            EvidenceCreate.model_validate(
                {"content": content, "source_type": "conversation_note", "source_label": label}
            )
        )
    async with factory() as session:
        result = await ExtractionService(session, FakeExtractionProvider(payload=payload)).extract(
            evidence.id
        )
        assert result.runs[0].status is RunStatus.SUCCEEDED
    return evidence.id


async def _all_kinds(factory: SessionFactory) -> uuid.UUID:
    return await _ingest(
        factory, content=EVIDENCE_CONTENT, label="DevGuild lab", payload=all_kinds_payload()
    )


async def _corpus(factory: SessionFactory) -> tuple[uuid.UUID, uuid.UUID]:
    """Two notes: one carrying all eight kinds, one carrying three."""
    first = await _all_kinds(factory)
    second = await _ingest(
        factory, content=SECOND_CONTENT, label="A week later", payload=second_payload()
    )
    return first, second


async def _resolve_everything_resolvable(factory: SessionFactory) -> uuid.UUID:
    """Run M3 over the corpus so entity context exists wherever it can exist."""
    async with factory() as session:
        entity = await EntityService(session).create_entity(
            EntityCreate(entity_type=EntityType.PERSON, canonical_name="Aisyah")
        )
    roles = {
        ItemKind.ENTITY_MENTION: ResolutionRole.MENTION,
        ItemKind.REPORTED_CLAIM: ResolutionRole.CLAIMED_BY,
        ItemKind.OBSERVATION: ResolutionRole.ABOUT,
    }
    async with factory() as session:
        items = list((await session.execute(select(ExtractionItem))).scalars().all())
    for item in items:
        role = roles.get(item.kind)
        if role is None:
            continue
        async with factory() as session:
            await EntityService(session).resolve(item.id, ResolutionCreate(role=role))
    return entity.id


async def _assemble(factory: SessionFactory, **kwargs: Any) -> EvidencePacket:
    async with factory() as session:
        return await PacketService(session).assemble(**kwargs)


async def _stored_items(factory: SessionFactory) -> list[tuple[uuid.UUID, str, Any]]:
    """Every extraction item as (id, kind, payload), sorted. A change detector."""
    async with factory() as session:
        rows = (await session.execute(select(ExtractionItem))).scalars().all()
        return sorted((item.id, item.kind.value, item.payload) for item in rows)


# --- the kinds that reach the packet --------------------------------------


async def test_the_six_grounded_kinds_enter_the_packet(session_factory: SessionFactory) -> None:
    await _all_kinds(session_factory)

    packet = await _assemble(session_factory)

    assert {item.kind for item in packet.items} == set(PACKET_ITEM_KINDS)
    assert len(packet.items) == 6


async def test_kinds_that_only_name_things_are_left_out(session_factory: SessionFactory) -> None:
    """`entity_mention` and `resource` are stored, and are not packet material."""
    await _all_kinds(session_factory)

    packet = await _assemble(session_factory)
    kinds = {item.kind for item in packet.items}

    assert ItemKind.ENTITY_MENTION not in kinds
    assert ItemKind.RESOURCE not in kinds


async def test_the_idea_kind_reaches_a_consumer_for_the_first_time(
    session_factory: SessionFactory,
) -> None:
    """The gap this package was built to close.

    Before the packet, `idea` items could not reach any downstream consumer at
    all - and the synthesis experiment found them among the most load-bearing
    evidence for a plan amendment.
    """
    await _all_kinds(session_factory)

    packet = await _assemble(session_factory)
    ideas = [item for item in packet.items if item.kind is ItemKind.IDEA]

    assert len(ideas) == 1
    assert ideas[0].grounding_quote == QUOTE_IDEA


# --- M4 relation eligibility has NOT widened ------------------------------


async def test_m4_still_sees_only_two_kinds_from_the_same_corpus(
    session_factory: SessionFactory,
) -> None:
    """The regression this whole design turns on.

    Same database, same rows, same moment: the packet carries six kinds and M4's
    candidate loader carries two. Packet inclusion does not imply M4 eligibility.
    """
    await _all_kinds(session_factory)
    entity_id = await _resolve_everything_resolvable(session_factory)

    packet = await _assemble(session_factory)
    async with session_factory() as session:
        relatable = await IntelligenceRepository(session).list_relatable_items(entity_id)

    packet_kinds = {item.kind for item in packet.items}
    m4_kinds = {item.kind for _resolution, item, _run, _evidence in relatable}

    assert packet_kinds == set(PACKET_ITEM_KINDS)
    assert m4_kinds == {ItemKind.REPORTED_CLAIM, ItemKind.OBSERVATION}
    assert packet_kinds - m4_kinds == PACKET_ONLY


async def test_assembling_a_packet_writes_nothing(session_factory: SessionFactory) -> None:
    """Read-only, so a packet can never perturb what M4 later reads."""
    await _all_kinds(session_factory)
    before = await _stored_items(session_factory)

    await _assemble(session_factory)

    assert await _stored_items(session_factory) == before


# --- provenance and grounding survive assembly ----------------------------


async def test_every_item_carries_the_provenance_a_citation_needs(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _all_kinds(session_factory)

    packet = await _assemble(session_factory)

    async with session_factory() as session:
        stored = {
            item.id: item
            for item in (await session.execute(select(ExtractionItem))).scalars().all()
        }

    for item in packet.items:
        assert item.item_id in stored
        assert item.evidence_id == evidence_id
        assert item.extraction_run_id == stored[item.item_id].extraction_run_id
        assert item.grounding_quote == stored[item.item_id].grounding_quote
        assert item.extraction_item_position == stored[item.item_id].position
        assert item.source_label == "DevGuild lab"


async def test_grounding_quotes_are_still_verbatim_in_the_source(
    session_factory: SessionFactory,
) -> None:
    """The M2 guarantee, re-checked after assembly rather than assumed."""
    await _all_kinds(session_factory)

    packet = await _assemble(session_factory)

    assert packet.items
    for item in packet.items:
        assert item.grounding_quote in EVIDENCE_CONTENT


async def test_entity_context_is_attached_where_m3_asserted_one(
    session_factory: SessionFactory,
) -> None:
    await _all_kinds(session_factory)
    entity_id = await _resolve_everything_resolvable(session_factory)

    packet = await _assemble(session_factory)
    claim = next(item for item in packet.items if item.kind is ItemKind.REPORTED_CLAIM)
    observation = next(item for item in packet.items if item.kind is ItemKind.OBSERVATION)

    assert [context.entity_id for context in claim.entities] == [entity_id]
    assert claim.entities[0].role is ResolutionRole.CLAIMED_BY
    assert observation.entities[0].role is ResolutionRole.ABOUT
    assert observation.entities[0].canonical_name == "Aisyah"


async def test_kinds_that_cannot_hold_a_resolution_still_reach_the_packet(
    session_factory: SessionFactory,
) -> None:
    """Why the packet is not entity-scoped.

    `entity_resolutions` binds each role to exactly one item kind, so `idea`,
    `opportunity`, `project` and `event` can never hold a resolution. Selecting
    on entity would drop all four - the kinds a decision most depends on.
    """
    await _all_kinds(session_factory)
    await _resolve_everything_resolvable(session_factory)

    packet = await _assemble(session_factory)
    unresolvable = [item for item in packet.items if item.kind in PACKET_ONLY]

    assert len(unresolvable) == 4
    assert all(item.entities == () for item in unresolvable)


async def test_a_speaker_never_becomes_a_subject_in_transit(
    session_factory: SessionFactory,
) -> None:
    await _all_kinds(session_factory)

    packet = await _assemble(session_factory)

    assert packet.items
    for item in packet.items:
        assert item.claimed_by is None or item.subject is None


# --- determinism ----------------------------------------------------------


async def test_the_packet_is_byte_identical_across_repeated_assembly(
    session_factory: SessionFactory,
) -> None:
    await _corpus(session_factory)

    first = render_packet(await _assemble(session_factory))
    second = render_packet(await _assemble(session_factory))

    assert first == second


async def test_items_are_ordered_by_evidence_arrival_then_position(
    session_factory: SessionFactory,
) -> None:
    first_evidence, second_evidence = await _corpus(session_factory)

    packet = await _assemble(session_factory)

    assert [item.evidence_id for item in packet.items] == [first_evidence] * 6 + [
        second_evidence
    ] * 2
    positions = [item.extraction_item_position for item in packet.items[:6]]
    assert positions == sorted(positions)


async def test_sources_are_listed_in_the_order_their_items_appear(
    session_factory: SessionFactory,
) -> None:
    first_evidence, second_evidence = await _corpus(session_factory)

    packet = await _assemble(session_factory)

    assert [source.evidence_id for source in packet.sources] == [first_evidence, second_evidence]
    assert [source.item_count for source in packet.sources] == [6, 2]


# --- one run per evidence -------------------------------------------------


async def test_a_failed_extraction_run_contributes_nothing(
    session_factory: SessionFactory,
) -> None:
    await _all_kinds(session_factory)
    async with session_factory() as session:
        # A failed run must carry an error code: the schema's own CHECK says so.
        await session.execute(
            update(ExtractionRun).values(
                status=RunStatus.FAILED,
                error_code="provider_error",
                error_message="forced failure, for this test",
            )
        )
        await session.commit()

    packet = await _assemble(session_factory)

    assert packet.items == ()
    assert packet.total_items_available == 0


async def test_re_extraction_does_not_double_count_one_note(
    session_factory: SessionFactory,
) -> None:
    """Two succeeded runs over one note must not both contribute.

    A bumped extractor version leaves the older run in place with a full item
    set. Showing both would make every count in the packet meaningless.
    """
    evidence_id = await _all_kinds(session_factory)
    async with session_factory() as session:
        # Age the first run out of the reuse key, exactly as a version bump does.
        await session.execute(
            update(ExtractionRun)
            .where(ExtractionRun.evidence_id == evidence_id)
            .values(extractor_version="0.0.1-older")
        )
        await session.commit()
    async with session_factory() as session:
        newer = await ExtractionService(
            session, FakeExtractionProvider(payload=reduced_payload())
        ).extract(evidence_id)
        assert newer.reused is False

    packet = await _assemble(session_factory)

    assert {item.extraction_run_id for item in packet.items} == {newer.runs[0].id}
    assert len(packet.items) == 2


# --- truncation is reported, never silent ---------------------------------


async def test_a_limit_that_binds_is_reported_with_what_it_cut(
    session_factory: SessionFactory,
) -> None:
    await _corpus(session_factory)

    packet = await _assemble(session_factory, limit=3)

    assert len(packet.items) == 3
    assert packet.total_items_available == 8
    assert packet.truncated is True
    assert "TRUNCATED" in render_packet(packet)


async def test_a_limit_that_does_not_bind_reports_completeness(
    session_factory: SessionFactory,
) -> None:
    await _corpus(session_factory)

    packet = await _assemble(session_factory, limit=100)

    assert packet.truncated is False
    assert packet.total_items_available == 8
    assert "Complete" in render_packet(packet)


async def test_an_empty_corpus_is_an_empty_packet_not_a_failure(
    session_factory: SessionFactory,
) -> None:
    packet = await _assemble(session_factory)

    assert packet.items == ()
    assert packet.sources == ()
    assert packet.truncated is False


# --- unreadable material fails loudly -------------------------------------


async def test_a_payload_that_cannot_be_read_back_stops_the_packet(
    session_factory: SessionFactory,
) -> None:
    """Loud, not skipped.

    Dropping the item would remove grounded material from a packet a reader
    believes is complete - and completeness is the only thing the packet claims.
    """
    await _all_kinds(session_factory)
    async with session_factory() as session:
        await session.execute(
            update(ExtractionItem)
            .where(ExtractionItem.kind == ItemKind.IDEA)
            .values(payload={"idea_text": "written under an older schema"})
        )
        await session.commit()

    with pytest.raises(UnreadablePayload):
        await _assemble(session_factory)


# --- the citation gate, against a real assembled packet -------------------


async def test_a_citation_quoting_its_own_item_binds(
    session_factory: SessionFactory,
) -> None:
    """The gate accepts, end to end: real rows, real quotes, no model."""
    await _all_kinds(session_factory)
    packet = await _assemble(session_factory)
    item = packet.items[0]

    handle = handle_for_index(0)
    audit = validate_bound_answer(f'A finding [{handle} "{item.grounding_quote}"].', packet)

    assert audit.cited == (item.item_id,)


async def test_a_citation_quoting_a_different_item_is_rejected(
    session_factory: SessionFactory,
) -> None:
    """The gate rejects, end to end. This is the misattribution the run produced.

    Both ids are real and both items are in the packet, so the membership check
    passes and only the quote binding catches it.
    """
    await _all_kinds(session_factory)
    packet = await _assemble(session_factory)
    cited, borrowed = packet.items[0], packet.items[1]
    assert cited.grounding_quote != borrowed.grounding_quote
    handle = handle_for_index(0)

    with pytest.raises(InvalidCitation, match=str(borrowed.item_id)):
        validate_bound_answer(f'A finding [{handle} "{borrowed.grounding_quote}"].', packet)


async def test_a_quote_copied_out_of_the_rendered_packet_binds(
    session_factory: SessionFactory,
) -> None:
    """The realistic path: the model reads the rendered text, not the objects.

    Long quotes are line-wrapped by the renderer, so this is what actually
    reaches a model and what it copies back.
    """
    await _all_kinds(session_factory)
    packet = await _assemble(session_factory)
    rendered = render_packet(packet)

    handles = packet_handles(packet)
    by_id = {item_id: handle for handle, item_id in handles.items()}

    for item in packet.items:
        # The handle is what the rendered packet offers; the uuid is not printed
        # at all, which is the point of the handle. So the round trip has to
        # start from the handle, exactly as a reader or a model would.
        marker = f"[{by_id[item.item_id]}]"
        block = rendered.split(marker, 1)[1]
        wrapped = block.split('quote: "', 1)[1].split('"', 1)[0]
        audit = validate_bound_answer(f'[{by_id[item.item_id]} "{wrapped}"]', packet)
        assert audit.cited == (item.item_id,)


async def test_the_packet_contract_and_the_validator_agree(
    session_factory: SessionFactory,
) -> None:
    """The instructions the packet gives are the rules the gate enforces.

    A packet that asked for one citation format while the validator required
    another would reject correct answers, and a rejected-but-correct answer is
    how a gate gets switched off.
    """
    await _all_kinds(session_factory)
    packet = await _assemble(session_factory)
    item = packet.items[0]

    assert '[E17 "verbatim quote"]' in render_packet(packet)

    handle = next(h for h, item_id in packet_handles(packet).items() if item_id == item.item_id)
    audit = validate_bound_answer(f'[{handle} "{item.grounding_quote}"]', packet)
    assert audit.cited_count == 1
