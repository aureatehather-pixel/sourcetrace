"""Entity connection against real PostgreSQL.

Everything here runs through the real migrations, the real service layer and the
deterministic fake extraction provider — no network, no credentials. The
assertions are about identity semantics, not about model behaviour.

The file is organised around the M3 invariants:

- identity is an id, not a name;
- unresolved is valid, and ambiguity stays unresolved;
- one entity accumulates many evidence items;
- a wrong resolution is correctable without touching evidence or extraction;
- every resolution is auditable;
- profiles are grounded and traceable.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelligence_engine.entities.errors import (
    EntityNotFound,
    ExtractionItemNotFound,
    IncompatibleResolutionRole,
    ResolutionAlreadyExists,
    ResolutionAlreadySuperseded,
    ResolutionNotFound,
)
from intelligence_engine.entities.models import Entity, EntityResolution
from intelligence_engine.entities.schemas import (
    EntityCreate,
    EntityType,
    ResolutionCreate,
    ResolutionMethod,
    ResolutionRole,
    ResolutionSupersede,
    UnresolvedReason,
)
from intelligence_engine.entities.service import EntityService
from intelligence_engine.evidence.fingerprint import content_fingerprint
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.evidence.schemas import EvidenceCreate
from intelligence_engine.evidence.service import EvidenceService
from intelligence_engine.extraction.fake_provider import FakeExtractionProvider
from intelligence_engine.extraction.models import ExtractionItem
from intelligence_engine.extraction.schemas import ItemKind
from intelligence_engine.extraction.service import ExtractionService
from tests.entity_fixtures import (
    ALEX_ONE_QUOTE,
    ALEX_TWO_QUOTE,
    B_QUOTE_GLYDE,
    C_QUOTE_SKILLS,
    EVIDENCE_A_CONTENT,
    EVIDENCE_ALEX_ONE_CONTENT,
    EVIDENCE_ALEX_TWO_CONTENT,
    EVIDENCE_B_CONTENT,
    EVIDENCE_C_CONTENT,
    EVIDENCE_UNNAMED_CONTENT,
    alex_payload,
    evidence_a_payload,
    evidence_b_payload,
    evidence_c_payload,
    unnamed_payload,
)

pytestmark = pytest.mark.integration

SessionFactory = async_sessionmaker[AsyncSession]

# Full-width "DevGuild", written as escapes so the code points are visible.
FULLWIDTH_DEVGUILD = "\uff24\uff45\uff56\uff27\uff55\uff49\uff4c\uff44"


# --- helpers --------------------------------------------------------------


async def _ingest(
    factory: SessionFactory, *, content: str, label: str, payload: dict[str, Any]
) -> list[ExtractionItem]:
    """Store evidence, extract it, and return the resulting items in order.

    This is the real M1 -> M2 path, so everything downstream is operating on
    genuinely persisted, genuinely grounded material.
    """
    async with factory() as session:
        evidence = await EvidenceService(session).create(
            EvidenceCreate.model_validate(
                {"content": content, "source_type": "conversation_note", "source_label": label}
            )
        )
    async with factory() as session:
        service = ExtractionService(session, FakeExtractionProvider(payload=payload))
        result = await service.extract(evidence.id)
        assert result.runs[0].status.value == "succeeded"

    async with factory() as session:
        rows = await session.execute(
            select(ExtractionItem)
            .where(ExtractionItem.extraction_run_id == result.runs[0].id)
            .order_by(ExtractionItem.position)
        )
        return list(rows.scalars().all())


async def _create_entity(
    factory: SessionFactory,
    *,
    name: str,
    entity_type: EntityType = EntityType.PERSON,
    disambiguator: str | None = None,
) -> uuid.UUID:
    async with factory() as session:
        entity = await EntityService(session).create_entity(
            EntityCreate(entity_type=entity_type, canonical_name=name, disambiguator=disambiguator)
        )
        return entity.id


async def _resolve(
    factory: SessionFactory,
    item_id: uuid.UUID,
    *,
    role: ResolutionRole = ResolutionRole.MENTION,
    entity_id: uuid.UUID | None = None,
    note: str | None = None,
) -> Any:
    async with factory() as session:
        return await EntityService(session).resolve(
            item_id, ResolutionCreate(role=role, entity_id=entity_id, note=note)
        )


def _mentions(items: list[ExtractionItem]) -> list[ExtractionItem]:
    return [item for item in items if item.kind is ItemKind.ENTITY_MENTION]


def _claims(items: list[ExtractionItem]) -> list[ExtractionItem]:
    return [item for item in items if item.kind is ItemKind.REPORTED_CLAIM]


# --- identity is not a name -----------------------------------------------


async def test_two_people_with_the_same_name_coexist(session_factory: SessionFactory) -> None:
    """Invariant 4. A unique name would make one of these two impossible."""
    first = await _create_entity(session_factory, name="Alex Lee", disambiguator="design team")
    second = await _create_entity(
        session_factory, name="Alex Lee", disambiguator="payments, Singapore bank"
    )

    assert first != second

    async with session_factory() as session:
        count = await session.scalar(
            select(func.count()).select_from(Entity).where(Entity.canonical_name == "Alex Lee")
        )
    assert count == 2


async def test_the_normalized_key_is_computed_not_supplied(
    session_factory: SessionFactory,
) -> None:
    padded = f"  {FULLWIDTH_DEVGUILD} "
    entity_id = await _create_entity(session_factory, name=padded)

    async with session_factory() as session:
        entity = await session.get(Entity, entity_id)
        assert entity is not None
        # The display name is preserved exactly; only the key is normalised.
        assert entity.canonical_name == padded
        assert entity.normalized_name == "devguild"


async def test_a_name_that_normalises_to_nothing_is_refused(
    session_factory: SessionFactory,
) -> None:
    """Proving the gate rejects: an empty key would match every other empty key."""
    async with session_factory() as session:
        session.add(
            Entity(
                id=uuid.uuid4(),
                entity_type=EntityType.PERSON,
                canonical_name="x",
                normalized_name="",
            )
        )
        with pytest.raises(IntegrityError, match="normalized_name_not_empty"):
            await session.commit()


# --- resolution: the conservative baseline --------------------------------


async def test_exact_match_resolves_a_single_candidate(
    session_factory: SessionFactory,
) -> None:
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="Kestrel Python meetup",
        payload=evidence_a_payload(),
    )
    entity_id = await _create_entity(session_factory, name="Zilan")

    outcome = await _resolve(session_factory, _mentions(items)[0].id)

    assert outcome.status == "resolved"
    assert outcome.resolution is not None
    assert outcome.resolution.entity_id == entity_id
    assert outcome.resolution.method is ResolutionMethod.EXACT_MATCH


async def test_exact_match_is_case_and_whitespace_insensitive(
    session_factory: SessionFactory,
) -> None:
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    entity_id = await _create_entity(session_factory, name="  ZILAN  ")

    outcome = await _resolve(session_factory, _mentions(items)[0].id)

    assert outcome.status == "resolved"
    assert outcome.resolution is not None
    assert outcome.resolution.entity_id == entity_id


async def test_no_candidate_leaves_the_mention_unresolved(
    session_factory: SessionFactory,
) -> None:
    """Invariant 5. Nothing is created to make the match succeed."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )

    outcome = await _resolve(session_factory, _mentions(items)[0].id)

    assert outcome.status == "unresolved"
    assert outcome.reason is UnresolvedReason.NO_CANDIDATE
    assert outcome.resolution is None

    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Entity)) == 0
        assert await session.scalar(select(func.count()).select_from(EntityResolution)) == 0


async def test_a_similar_but_not_identical_name_does_not_match(
    session_factory: SessionFactory,
) -> None:
    """`Zi Lan` is not `Zilan`. A false negative here is the cheap failure."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    await _create_entity(session_factory, name="Zi Lan")

    outcome = await _resolve(session_factory, _mentions(items)[0].id)

    assert outcome.status == "unresolved"
    assert outcome.reason is UnresolvedReason.NO_CANDIDATE


async def test_matching_is_type_aware(session_factory: SessionFactory) -> None:
    """A person mention must not resolve to an identically named organisation."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    await _create_entity(session_factory, name="Zilan", entity_type=EntityType.ORGANIZATION)

    outcome = await _resolve(session_factory, _mentions(items)[0].id)

    assert outcome.status == "unresolved"
    assert outcome.reason is UnresolvedReason.NO_CANDIDATE


# --- THE FALSE-MERGE TESTS ------------------------------------------------


async def test_two_same_name_entities_are_never_automatically_merged(
    session_factory: SessionFactory,
) -> None:
    """Mandatory regression. Two distinct people are both called Alex Lee.

    The matcher must refuse rather than pick one, and must say why so a human
    can decide. Silently choosing either would fuse two identities permanently.
    """
    items = await _ingest(
        session_factory,
        content=EVIDENCE_ALEX_ONE_CONTENT,
        label="design review",
        payload=alex_payload(ALEX_ONE_QUOTE, "On the design team"),
    )
    design_alex = await _create_entity(
        session_factory, name="Alex Lee", disambiguator="design team"
    )
    payments_alex = await _create_entity(
        session_factory, name="Alex Lee", disambiguator="payments, Singapore bank"
    )

    outcome = await _resolve(session_factory, _mentions(items)[0].id)

    assert outcome.status == "unresolved"
    assert outcome.reason is UnresolvedReason.AMBIGUOUS_CANDIDATES
    assert outcome.resolution is None
    # Both candidates are reported, so the operator can immediately pick one.
    assert set(outcome.candidate_entity_ids) == {design_alex, payments_alex}

    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(EntityResolution)) == 0


async def test_two_independent_alex_lee_mentions_stay_separate(
    session_factory: SessionFactory,
) -> None:
    """Same name, two evidence items, two people. They must not converge."""
    first_items = await _ingest(
        session_factory,
        content=EVIDENCE_ALEX_ONE_CONTENT,
        label="design review",
        payload=alex_payload(ALEX_ONE_QUOTE, "On the design team"),
    )
    second_items = await _ingest(
        session_factory,
        content=EVIDENCE_ALEX_TWO_CONTENT,
        label="banking intro",
        payload=alex_payload(ALEX_TWO_QUOTE, "Payments infrastructure at a bank"),
    )
    design_alex = await _create_entity(
        session_factory, name="Alex Lee", disambiguator="design team"
    )
    payments_alex = await _create_entity(
        session_factory, name="Alex Lee", disambiguator="payments, Singapore bank"
    )

    # Only a human can tell these apart, so only a human does.
    await _resolve(session_factory, _mentions(first_items)[0].id, entity_id=design_alex)
    await _resolve(session_factory, _mentions(second_items)[0].id, entity_id=payments_alex)

    async with session_factory() as session:
        design_profile = await EntityService(session).get_profile(design_alex)
        payments_profile = await EntityService(session).get_profile(payments_alex)

    assert design_profile.evidence_count == 1
    assert payments_profile.evidence_count == 1
    assert design_profile.sources[0].evidence_id != payments_profile.sources[0].evidence_id


async def test_an_unnamed_attribution_never_manufactures_a_person(
    session_factory: SessionFactory,
) -> None:
    """`Engineer at DevGuild` identifies nobody, even with DevGuild on file."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_UNNAMED_CONTENT,
        label="DevGuild meetup",
        payload=unnamed_payload(),
    )
    await _create_entity(session_factory, name="DevGuild", entity_type=EntityType.ORGANIZATION)

    outcome = await _resolve(session_factory, _claims(items)[0].id, role=ResolutionRole.CLAIMED_BY)

    assert outcome.status == "unresolved"
    assert outcome.reason is UnresolvedReason.NO_CANDIDATE

    async with session_factory() as session:
        people = await session.scalar(
            select(func.count()).select_from(Entity).where(Entity.entity_type == "person")
        )
    assert people == 0


async def test_an_unattributed_claim_has_no_name_to_offer(
    session_factory: SessionFactory,
) -> None:
    payload = {
        "items": [
            {
                "kind": "reported_claim",
                "data": {"claim": "Reading comes first.", "claimed_by": None, "stance": None},
                "grounding": {"quote": "the first six weeks were mostly reading, not writing"},
            }
        ]
    }
    items = await _ingest(
        session_factory, content=EVIDENCE_B_CONTENT, label="coffee", payload=payload
    )

    outcome = await _resolve(session_factory, _claims(items)[0].id, role=ResolutionRole.CLAIMED_BY)

    assert outcome.status == "unresolved"
    assert outcome.reason is UnresolvedReason.NO_MATCHABLE_NAME


# --- role/kind pairing ----------------------------------------------------


async def test_a_claimed_by_role_cannot_be_applied_to_a_mention(
    session_factory: SessionFactory,
) -> None:
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    entity_id = await _create_entity(session_factory, name="Zilan")

    with pytest.raises(IncompatibleResolutionRole):
        await _resolve(
            session_factory,
            _mentions(items)[0].id,
            role=ResolutionRole.CLAIMED_BY,
            entity_id=entity_id,
        )


async def test_the_database_itself_refuses_a_mismatched_role(
    session_factory: SessionFactory,
) -> None:
    """The service gate is not the only gate. Proving the CHECK actually fires."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    entity_id = await _create_entity(session_factory, name="Zilan")
    mention = _mentions(items)[0]

    async with session_factory() as session:
        session.add(
            EntityResolution(
                id=uuid.uuid4(),
                extraction_item_id=mention.id,
                source_item_kind=ItemKind.ENTITY_MENTION,
                entity_id=entity_id,
                role=ResolutionRole.CLAIMED_BY,  # wrong role for an entity_mention
                method=ResolutionMethod.MANUAL,
                is_active=True,
                note=None,
            )
        )
        with pytest.raises(IntegrityError, match="role_matches_source_item_kind"):
            await session.commit()


async def test_the_composite_fk_refuses_a_lie_about_the_item_kind(
    session_factory: SessionFactory,
) -> None:
    """`source_item_kind` cannot disagree with the item it points at."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    entity_id = await _create_entity(session_factory, name="Zilan")
    mention = _mentions(items)[0]

    async with session_factory() as session:
        session.add(
            EntityResolution(
                id=uuid.uuid4(),
                extraction_item_id=mention.id,  # actually an entity_mention
                source_item_kind=ItemKind.REPORTED_CLAIM,  # claimed otherwise
                entity_id=entity_id,
                role=ResolutionRole.CLAIMED_BY,
                method=ResolutionMethod.MANUAL,
                is_active=True,
                note=None,
            )
        )
        with pytest.raises(IntegrityError, match="fk_entity_resolutions_extraction_item"):
            await session.commit()


# --- one active resolution per item/role ----------------------------------


async def test_resolving_an_already_resolved_item_is_refused(
    session_factory: SessionFactory,
) -> None:
    """Overwriting would erase what the system previously believed."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    first = await _create_entity(session_factory, name="Zilan")
    second = await _create_entity(session_factory, name="Zilan Orr")
    mention = _mentions(items)[0]

    await _resolve(session_factory, mention.id, entity_id=first)

    with pytest.raises(ResolutionAlreadyExists):
        await _resolve(session_factory, mention.id, entity_id=second)


async def test_the_database_refuses_a_second_active_resolution(
    session_factory: SessionFactory,
) -> None:
    """The partial unique index, proven to reject rather than merely declared."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    first = await _create_entity(session_factory, name="Zilan")
    second = await _create_entity(session_factory, name="Zilan Orr")
    mention = _mentions(items)[0]
    await _resolve(session_factory, mention.id, entity_id=first)

    async with session_factory() as session:
        session.add(
            EntityResolution(
                id=uuid.uuid4(),
                extraction_item_id=mention.id,
                source_item_kind=ItemKind.ENTITY_MENTION,
                entity_id=second,
                role=ResolutionRole.MENTION,
                method=ResolutionMethod.MANUAL,
                is_active=True,
                note=None,
            )
        )
        with pytest.raises(IntegrityError, match="uq_entity_resolutions_active_item_role"):
            await session.commit()


# --- correction -----------------------------------------------------------


async def test_a_wrong_resolution_is_corrected_by_supersession(
    session_factory: SessionFactory,
) -> None:
    """Invariant 8. The old assertion survives; the new one takes over."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_ALEX_ONE_CONTENT,
        label="design review",
        payload=alex_payload(ALEX_ONE_QUOTE, "On the design team"),
    )
    wrong = await _create_entity(
        session_factory, name="Alex Lee", disambiguator="payments, Singapore bank"
    )
    right = await _create_entity(session_factory, name="Alex Lee", disambiguator="design team")
    mention = _mentions(items)[0]

    first = await _resolve(session_factory, mention.id, entity_id=wrong)
    assert first.resolution is not None

    async with session_factory() as session:
        corrected = await EntityService(session).supersede(
            first.resolution.id,
            ResolutionSupersede(entity_id=right, note="wrong Alex Lee - this is the designer"),
        )

    assert corrected.status == "resolved"
    assert corrected.resolution is not None
    assert corrected.resolution.entity_id == right
    assert corrected.superseded is not None
    assert corrected.superseded.id == first.resolution.id
    assert corrected.superseded.is_active is False
    assert corrected.superseded.superseded_by_id == corrected.resolution.id
    assert corrected.superseded.superseded_at is not None
    # The retired row still says what it originally asserted.
    assert corrected.superseded.entity_id == wrong


async def test_history_survives_correction(session_factory: SessionFactory) -> None:
    """Invariant 7. Both decisions remain readable, newest first."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_ALEX_ONE_CONTENT,
        label="design review",
        payload=alex_payload(ALEX_ONE_QUOTE, "On the design team"),
    )
    wrong = await _create_entity(session_factory, name="Alex Lee", disambiguator="payments")
    right = await _create_entity(session_factory, name="Alex Lee", disambiguator="design")
    mention = _mentions(items)[0]

    first = await _resolve(session_factory, mention.id, entity_id=wrong)
    assert first.resolution is not None
    async with session_factory() as session:
        await EntityService(session).supersede(
            first.resolution.id, ResolutionSupersede(entity_id=right, note="corrected")
        )

    async with session_factory() as session:
        history = await EntityService(session).list_resolutions(mention.id)

    assert len(history.items) == 2
    assert [row.is_active for row in history.items] == [True, False]
    assert [row.entity_id for row in history.items] == [right, wrong]


async def test_a_correction_can_detach_without_naming_a_replacement(
    session_factory: SessionFactory,
) -> None:
    """ "That is not him, and I do not know who it is" is a valid correction."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    wrong = await _create_entity(session_factory, name="Zilan")
    mention = _mentions(items)[0]
    first = await _resolve(session_factory, mention.id, entity_id=wrong)
    assert first.resolution is not None

    async with session_factory() as session:
        outcome = await EntityService(session).supersede(
            first.resolution.id, ResolutionSupersede(note="different Zilan")
        )

    assert outcome.status == "unresolved"
    assert outcome.reason is UnresolvedReason.DETACHED_BY_CORRECTION
    assert outcome.resolution is None
    assert outcome.superseded is not None
    assert outcome.superseded.superseded_by_id is None

    # The mention is genuinely free again, and re-resolvable.
    async with session_factory() as session:
        profile = await EntityService(session).get_profile(wrong)
    assert profile.mentions == []

    second = await _resolve(session_factory, mention.id, entity_id=wrong)
    assert second.status == "resolved"


async def test_a_superseded_resolution_cannot_be_superseded_again(
    session_factory: SessionFactory,
) -> None:
    """Otherwise the history forks into two chains both claiming to be current."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    first_entity = await _create_entity(session_factory, name="Zilan")
    second_entity = await _create_entity(session_factory, name="Zilan Orr")
    mention = _mentions(items)[0]
    first = await _resolve(session_factory, mention.id, entity_id=first_entity)
    assert first.resolution is not None

    async with session_factory() as session:
        await EntityService(session).supersede(
            first.resolution.id, ResolutionSupersede(entity_id=second_entity)
        )

    async with session_factory() as session:
        with pytest.raises(ResolutionAlreadySuperseded):
            await EntityService(session).supersede(
                first.resolution.id, ResolutionSupersede(entity_id=second_entity)
            )


async def test_correction_does_not_touch_evidence_or_extraction(
    session_factory: SessionFactory,
) -> None:
    """Invariants 1 and 2, checked against the database rather than intent."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    mention = _mentions(items)[0]
    before_payload = dict(mention.payload)
    before_quote = mention.grounding_quote

    wrong = await _create_entity(session_factory, name="Zilan")
    right = await _create_entity(session_factory, name="Zilan Orr")
    first = await _resolve(session_factory, mention.id, entity_id=wrong)
    assert first.resolution is not None
    async with session_factory() as session:
        await EntityService(session).supersede(
            first.resolution.id, ResolutionSupersede(entity_id=right)
        )

    async with session_factory() as session:
        stored = (await session.execute(select(EvidenceItem))).scalars().all()
        assert len(stored) == 1
        evidence = stored[0]
        # Byte-for-byte, and the fingerprint still describes the stored bytes.
        assert evidence.content == EVIDENCE_A_CONTENT
        assert content_fingerprint(evidence.content) == evidence.content_hash

        reloaded = await session.get(ExtractionItem, mention.id)
        assert reloaded is not None
        assert reloaded.payload == before_payload
        assert reloaded.grounding_quote == before_quote
        assert reloaded.kind is ItemKind.ENTITY_MENTION


# --- accumulation and the profile ----------------------------------------


async def test_one_entity_accumulates_two_independent_evidence_items(
    session_factory: SessionFactory,
) -> None:
    """THE central M3 product test.

    Two separate evidence items, extracted separately, resolved to one entity.
    The profile must carry grounded material from both, with provenance intact.
    """
    items_a = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="Kestrel Python meetup",
        payload=evidence_a_payload(),
    )
    items_b = await _ingest(
        session_factory,
        content=EVIDENCE_B_CONTENT,
        label="Follow-up coffee",
        payload=evidence_b_payload(),
    )
    zilan = await _create_entity(
        session_factory, name="Zilan", disambiguator="Kestrel Python meetup, query planner rebuild"
    )

    # Evidence A: mention resolved by hand (nothing existed to match at first),
    # then both claims resolve deterministically off the same name.
    await _resolve(session_factory, _mentions(items_a)[0].id, entity_id=zilan)
    for claim in _claims(items_a):
        outcome = await _resolve(session_factory, claim.id, role=ResolutionRole.CLAIMED_BY)
        assert outcome.status == "resolved"
        assert outcome.resolution is not None
        assert outcome.resolution.method is ResolutionMethod.EXACT_MATCH

    # Evidence B: the mention now resolves deterministically too.
    mention_b = await _resolve(session_factory, _mentions(items_b)[0].id)
    assert mention_b.status == "resolved"
    assert mention_b.resolution is not None
    assert mention_b.resolution.method is ResolutionMethod.EXACT_MATCH
    for claim in _claims(items_b):
        await _resolve(session_factory, claim.id, role=ResolutionRole.CLAIMED_BY)

    async with session_factory() as session:
        profile = await EntityService(session).get_profile(zilan)

    assert profile.entity.id == zilan
    assert profile.evidence_count == 2

    # Both sources are named, and both contributed.
    labels = {source.source_label for source in profile.sources}
    assert labels == {"Kestrel Python meetup", "Follow-up coffee"}

    evidence_ids = {entry.evidence_id for entry in profile.mentions} | {
        entry.evidence_id for entry in profile.attributed_claims
    }
    assert len(evidence_ids) == 2

    # Three claims, one of which qualifies an earlier one - both survive, each
    # with its own quote. Nothing was smoothed into a single statement.
    assert len(profile.attributed_claims) == 3
    stances = {entry.stance for entry in profile.attributed_claims}
    assert stances == {"recommended", "asserted"}

    # Every line traces to text that is verbatim in its own evidence item.
    async with session_factory() as session:
        contents = {
            row.id: row.content
            for row in (await session.execute(select(EvidenceItem))).scalars().all()
        }
    for entry in [*profile.mentions, *profile.attributed_claims]:
        assert entry.grounding_quote in contents[entry.evidence_id]


async def test_a_profile_excludes_material_it_merely_co_occurs_with(
    session_factory: SessionFactory,
) -> None:
    """Co-occurrence is not attribution.

    Evidence B mentions Glyde in the same note. Resolving Zilan must not sweep
    the Glyde mention into his profile just because they share an evidence item.
    """
    items_b = await _ingest(
        session_factory,
        content=EVIDENCE_B_CONTENT,
        label="Follow-up coffee",
        payload=evidence_b_payload(),
    )
    zilan = await _create_entity(session_factory, name="Zilan")
    glyde = await _create_entity(session_factory, name="Glyde", entity_type=EntityType.ORGANIZATION)

    for mention in _mentions(items_b):
        await _resolve(session_factory, mention.id)

    async with session_factory() as session:
        service = EntityService(session)
        zilan_profile = await service.get_profile(zilan)
        glyde_profile = await service.get_profile(glyde)

    assert [entry.name_as_mentioned for entry in zilan_profile.mentions] == ["Zilan"]
    assert [entry.name_as_mentioned for entry in glyde_profile.mentions] == ["Glyde"]
    assert glyde_profile.attributed_claims == []
    assert B_QUOTE_GLYDE not in {entry.grounding_quote for entry in zilan_profile.mentions}


async def test_a_superseded_resolution_leaves_the_profile(
    session_factory: SessionFactory,
) -> None:
    """A profile states what the system believes now, not what it once believed."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    wrong = await _create_entity(session_factory, name="Zilan", disambiguator="wrong one")
    right = await _create_entity(session_factory, name="Zilan", disambiguator="right one")
    mention = _mentions(items)[0]
    first = await _resolve(session_factory, mention.id, entity_id=wrong)
    assert first.resolution is not None

    async with session_factory() as session:
        before = await EntityService(session).get_profile(wrong)
    assert len(before.mentions) == 1

    async with session_factory() as session:
        await EntityService(session).supersede(
            first.resolution.id, ResolutionSupersede(entity_id=right)
        )

    async with session_factory() as session:
        service = EntityService(session)
        after_wrong = await service.get_profile(wrong)
        after_right = await service.get_profile(right)

    assert after_wrong.mentions == []
    assert after_wrong.evidence_count == 0
    assert len(after_right.mentions) == 1
    # But the retired assertion is still on the record.
    async with session_factory() as session:
        history = await EntityService(session).list_resolutions(mention.id)
    assert {row.entity_id for row in history.items} == {wrong, right}


async def test_an_empty_profile_is_a_valid_profile(session_factory: SessionFactory) -> None:
    entity_id = await _create_entity(session_factory, name="Nobody Yet")

    async with session_factory() as session:
        profile = await EntityService(session).get_profile(entity_id)

    assert profile.evidence_count == 0
    assert profile.sources == []
    assert profile.mentions == []
    assert profile.attributed_claims == []


# --- referential protection -----------------------------------------------


async def test_a_resolved_entity_cannot_be_deleted(session_factory: SessionFactory) -> None:
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    entity_id = await _create_entity(session_factory, name="Zilan")
    await _resolve(session_factory, _mentions(items)[0].id, entity_id=entity_id)

    async with session_factory() as session:
        with pytest.raises(IntegrityError):
            await session.execute(
                text("DELETE FROM entities WHERE id = :id"), {"id": str(entity_id)}
            )
            await session.commit()


async def test_a_resolved_extraction_run_cannot_be_deleted(
    session_factory: SessionFactory,
) -> None:
    """RESTRICT beats the M2 CASCADE: resolutions protect the items beneath them."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    entity_id = await _create_entity(session_factory, name="Zilan")
    mention = _mentions(items)[0]
    await _resolve(session_factory, mention.id, entity_id=entity_id)

    async with session_factory() as session:
        with pytest.raises(IntegrityError):
            await session.execute(
                text("DELETE FROM extraction_runs WHERE id = :id"),
                {"id": str(mention.extraction_run_id)},
            )
            await session.commit()


# --- not-found paths ------------------------------------------------------


async def test_unknown_ids_raise_the_right_errors(session_factory: SessionFactory) -> None:
    missing = uuid.uuid4()

    async with session_factory() as session:
        service = EntityService(session)
        with pytest.raises(EntityNotFound):
            await service.get_entity(missing)
        with pytest.raises(EntityNotFound):
            await service.get_profile(missing)
        with pytest.raises(ExtractionItemNotFound):
            await service.resolve(missing, ResolutionCreate(role=ResolutionRole.MENTION))
        with pytest.raises(ResolutionNotFound):
            await service.supersede(missing, ResolutionSupersede())


async def test_resolving_to_an_unknown_entity_is_refused(
    session_factory: SessionFactory,
) -> None:
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )

    with pytest.raises(EntityNotFound):
        await _resolve(session_factory, _mentions(items)[0].id, entity_id=uuid.uuid4())


# --- the `about` role: observations reach an entity without becoming claims ---
#
# Every test below is a rejection case or an adversarial case. The positive path
# is cheap to get right; these are the ones that decide whether the role is safe.


def _observations(items: list[ExtractionItem]) -> list[ExtractionItem]:
    return [item for item in items if item.kind is ItemKind.OBSERVATION]


async def _ingest_profile(factory: SessionFactory) -> list[ExtractionItem]:
    return await _ingest(
        factory,
        content=EVIDENCE_C_CONTENT,
        label="public profile",
        payload=evidence_c_payload(),
    )


async def test_the_database_refuses_about_on_a_reported_claim(
    session_factory: SessionFactory,
) -> None:
    """`about` is for observations only. Proving the widened CHECK still bites.

    Without this, a claim could be re-labelled as a fact about its own speaker,
    which is exactly the collapse the reported-claim design prevents.
    """
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    entity_id = await _create_entity(session_factory, name="Zilan")

    async with session_factory() as session:
        session.add(
            EntityResolution(
                id=uuid.uuid4(),
                extraction_item_id=_claims(items)[0].id,
                source_item_kind=ItemKind.REPORTED_CLAIM,
                entity_id=entity_id,
                role=ResolutionRole.ABOUT,  # wrong role for a reported_claim
                method=ResolutionMethod.MANUAL,
                is_active=True,
                note=None,
            )
        )
        with pytest.raises(IntegrityError, match="role_matches_source_item_kind"):
            await session.commit()


async def test_the_database_refuses_claimed_by_on_an_observation(
    session_factory: SessionFactory,
) -> None:
    """The other direction: an observation must never acquire a speaker."""
    items = await _ingest_profile(session_factory)
    entity_id = await _create_entity(session_factory, name="Zilan")

    async with session_factory() as session:
        session.add(
            EntityResolution(
                id=uuid.uuid4(),
                extraction_item_id=_observations(items)[0].id,
                source_item_kind=ItemKind.OBSERVATION,
                entity_id=entity_id,
                role=ResolutionRole.CLAIMED_BY,  # an observation has no claimant
                method=ResolutionMethod.MANUAL,
                is_active=True,
                note=None,
            )
        )
        with pytest.raises(IntegrityError, match="role_matches_source_item_kind"):
            await session.commit()


async def test_one_observation_cannot_hold_two_active_about_resolutions(
    session_factory: SessionFactory,
) -> None:
    """Single-subject is enforced, not merely intended.

    The approved design splits a comparative observation at extraction time. If
    this index did not hold, a second subject could be bolted on afterwards and
    supersession would silently stop meaning correction.
    """
    items = await _ingest_profile(session_factory)
    first = await _create_entity(session_factory, name="Zilan")
    second = await _create_entity(session_factory, name="Kai")
    observation = _observations(items)[0]

    await _resolve(session_factory, observation.id, role=ResolutionRole.ABOUT, entity_id=first)

    async with session_factory() as session:
        session.add(
            EntityResolution(
                id=uuid.uuid4(),
                extraction_item_id=observation.id,
                source_item_kind=ItemKind.OBSERVATION,
                entity_id=second,
                role=ResolutionRole.ABOUT,
                method=ResolutionMethod.MANUAL,
                is_active=True,
                note=None,
            )
        )
        with pytest.raises(IntegrityError, match="uq_entity_resolutions_active_item_role"):
            await session.commit()


async def test_co_occurring_names_do_not_become_subjects(
    session_factory: SessionFactory,
) -> None:
    """The co-occurrence trap, run as an experiment rather than argued about.

    The AWS observation names Zilan, AWS, TVU and Kestrel Bay. All four
    exist as entities on file, so a co-occurrence-based design would have every
    reason to attach the observation to each. Only the sentence's actual subject
    may resolve; the other three must end up with nothing.
    """
    items = await _ingest_profile(session_factory)
    zilan = await _create_entity(session_factory, name="Zilan")
    aws = await _create_entity(session_factory, name="AWS", entity_type=EntityType.ORGANIZATION)
    tvu = await _create_entity(session_factory, name="TVU", entity_type=EntityType.ORGANIZATION)
    kestrel_bay = await _create_entity(
        session_factory, name="Kestrel Bay", entity_type=EntityType.PLACE
    )

    aws_observation = next(
        item for item in _observations(items) if "AWS" in item.payload["observation"]
    )
    outcome = await _resolve(session_factory, aws_observation.id, role=ResolutionRole.ABOUT)

    assert outcome.status == "resolved"
    assert outcome.resolution is not None
    assert outcome.resolution.entity_id == zilan

    # The other three gained nothing at all from being named in the sentence.
    async with session_factory() as session:
        for other in (aws, tvu, kestrel_bay):
            count = await session.scalar(
                select(func.count())
                .select_from(EntityResolution)
                .where(EntityResolution.entity_id == other)
            )
            assert count == 0


async def test_a_first_person_observation_creates_no_entity(
    session_factory: SessionFactory,
) -> None:
    """`subject = "I"` must stay unresolved. Author modelling is not built.

    The failure this guards against is a self-entity appearing as a side effect
    of ingestion, which would then start collecting the author's whole corpus.
    """
    items = await _ingest_profile(session_factory)
    await _create_entity(session_factory, name="Zilan")

    self_observation = next(item for item in _observations(items) if item.payload["subject"] == "I")
    outcome = await _resolve(session_factory, self_observation.id, role=ResolutionRole.ABOUT)

    assert outcome.status == "unresolved"
    assert outcome.reason is UnresolvedReason.NO_CANDIDATE

    async with session_factory() as session:
        names = list(
            await session.scalars(select(Entity.canonical_name).order_by(Entity.canonical_name))
        )
        assert names == ["Zilan"], "ingesting a first-person observation invented an identity"


async def test_an_ambiguous_subject_declines_and_writes_nothing(
    session_factory: SessionFactory,
) -> None:
    """Two people share the name. The matcher must refuse to pick one."""
    items = await _ingest_profile(session_factory)
    await _create_entity(session_factory, name="Zilan", disambiguator="the one from DevGuild")
    await _create_entity(session_factory, name="Zilan", disambiguator="a different Zilan")

    observation = _observations(items)[0]
    outcome = await _resolve(session_factory, observation.id, role=ResolutionRole.ABOUT)

    assert outcome.status == "unresolved"
    assert outcome.reason is UnresolvedReason.AMBIGUOUS_CANDIDATES

    async with session_factory() as session:
        count = await session.scalar(select(func.count()).select_from(EntityResolution))
        assert count == 0


async def test_an_observation_with_no_subject_offers_no_name(
    session_factory: SessionFactory,
) -> None:
    """An absent subject is "the evidence did not say", not "matched nothing"."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_C_CONTENT,
        label="public profile",
        payload={
            "items": [
                {
                    "kind": "observation",
                    "data": {"observation": "The profile listed two certificates."},
                    "grounding": {"quote": C_QUOTE_SKILLS},
                }
            ]
        },
    )
    await _create_entity(session_factory, name="Zilan")

    outcome = await _resolve(session_factory, items[0].id, role=ResolutionRole.ABOUT)

    assert outcome.status == "unresolved"
    assert outcome.reason is UnresolvedReason.NO_MATCHABLE_NAME


async def test_the_profile_keeps_what_was_said_apart_from_what_was_observed(
    session_factory: SessionFactory,
) -> None:
    """The whole point, checked end to end.

    Zilan both said things (evidence A) and had things observed about him
    (evidence C). The profile must show both and must never merge them: a reader
    has to be able to tell "he told me X" from "his profile showed Y".
    """
    spoken = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="meetup",
        payload=evidence_a_payload(),
    )
    observed = await _ingest_profile(session_factory)
    entity_id = await _create_entity(session_factory, name="Zilan")

    await _resolve(session_factory, _claims(spoken)[0].id, role=ResolutionRole.CLAIMED_BY)
    for item in _observations(observed):
        await _resolve(session_factory, item.id, role=ResolutionRole.ABOUT)

    async with session_factory() as session:
        profile = await EntityService(session).get_profile(entity_id)

    assert len(profile.attributed_claims) == 1
    # Three subjects say "Zilan"; the first-person one resolves to nobody.
    assert len(profile.subject_observations) == 3

    claim_texts = {claim.claim for claim in profile.attributed_claims}
    observation_texts = {row.observation for row in profile.subject_observations}
    assert not (claim_texts & observation_texts), "said and observed were merged"

    for row in profile.subject_observations:
        assert row.subject == "Zilan"
        assert row.grounding_quote in EVIDENCE_C_CONTENT
