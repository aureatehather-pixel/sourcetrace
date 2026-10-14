"""The `pie process` orchestration, against real PostgreSQL.

The CLI adds no persistence and no domain logic, so nothing here re-tests M1-M4.
What it tests is the handful of *orchestration judgements* that live only in
`cli/pipeline.py`, each of which would fail silently rather than loudly if it
were wrong:

- mentions are resolved before attributions, so one decision settles both;
- an attribution string is never offered as something to create an identity from;
- non-interactive mode never creates or resolves anything on its own;
- re-processing the same note writes nothing new and asks nothing again;
- analysis is spent only on entities that gained an attributed claim.

The decider is a recording fake, so "was the human asked, and about what" is an
assertion rather than an assumption. Providers are the deterministic fakes: this
file is about orchestration, never about model quality.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelligence_engine.cli.commands import EvidenceMetadata
from intelligence_engine.cli.context import Services
from intelligence_engine.cli.decisions import (
    AlwaysSkip,
    Answer,
    Candidate,
    CreateEntity,
    Question,
    Skip,
    UseEntity,
)
from intelligence_engine.cli.pipeline import process
from intelligence_engine.entities.models import Entity, EntityResolution
from intelligence_engine.entities.schemas import ResolutionRole
from intelligence_engine.entities.service import EntityService
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.evidence.service import EvidenceService
from intelligence_engine.extraction.fake_provider import FakeExtractionProvider
from intelligence_engine.extraction.service import ExtractionService
from intelligence_engine.intelligence.models import IntelligenceRelation
from intelligence_engine.intelligence.service import IntelligenceService
from intelligence_engine.packet.service import PacketService
from tests.entity_fixtures import (
    EVIDENCE_A_CONTENT,
    EVIDENCE_B_CONTENT,
    EVIDENCE_C_CONTENT,
    EVIDENCE_UNNAMED_CONTENT,
    evidence_a_payload,
    evidence_b_payload,
    evidence_c_payload,
    unnamed_payload,
)
from tests.intelligence_fixtures import classifier_response, qualifies

pytestmark = pytest.mark.integration

SessionFactory = async_sessionmaker[AsyncSession]


class _RecordingDecider:
    """A scripted human. Records every question it was actually asked."""

    def __init__(self, *answers: Answer) -> None:
        self._answers = list(answers)
        self.questions: list[Question] = []

    def decide(self, question: Question) -> Answer:
        self.questions.append(question)
        return self._answers.pop(0) if self._answers else Skip()


def _services(
    session: AsyncSession,
    *,
    payload: dict[str, Any],
    classifier_output: str | None = None,
    batch_size: int = 8,
) -> Services:
    return Services(
        evidence=EvidenceService(session),
        extraction=ExtractionService(session, FakeExtractionProvider(payload=payload)),
        entities=EntityService(session),
        intelligence=IntelligenceService(
            session,
            FakeExtractionProvider(raw_output=classifier_output or classifier_response()),
            batch_size=batch_size,
        ),
        packet=PacketService(session),
    )


async def _count(session_factory: SessionFactory, model: Any) -> int:
    async with session_factory() as session:
        return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


async def test_one_decision_settles_both_the_mention_and_the_attribution(
    session_factory: SessionFactory,
) -> None:
    """Mentions are resolved first, so the claims attributed to that name follow.

    If the ordering were reversed, `claimed_by="Zilan"` would be reached while no
    Zilan existed, and the operator would be asked once per claim about a name
    they had already identified.
    """
    decider = _RecordingDecider(
        CreateEntity(canonical_name="Zilan", disambiguator="Kestrel meetup")
    )

    async with session_factory() as session:
        result = await process(
            _services(session, payload=evidence_a_payload()),
            content=EVIDENCE_A_CONTENT,
            meta=EvidenceMetadata(source_type="conversation_note"),
            force_new=False,
            analyze=False,
            decider=decider,
        )

    assert len(decider.questions) == 1
    assert decider.questions[0].name == "Zilan"
    assert decider.questions[0].role == ResolutionRole.MENTION.value

    created = [record for record in result.resolutions if record.outcome == "created"]
    resolved = [record for record in result.resolutions if record.outcome == "resolved"]
    assert len(created) == 1
    assert created[0].role is ResolutionRole.MENTION

    # Every claim attributed to that name landed on the same entity, unasked.
    assert resolved
    assert all(record.role is ResolutionRole.CLAIMED_BY for record in resolved)
    assert {record.entity_id for record in resolved} == {created[0].entity_id}
    assert await _count(session_factory, Entity) == 1


async def test_an_attribution_string_is_never_offered_for_creation(
    session_factory: SessionFactory,
) -> None:
    """`claimed_by = "Engineer at DevGuild"` identifies nobody, and must stay that way.

    The operator is not asked at all here: the only answers that could apply are
    "use an existing entity" (there are none) and "skip". Offering "create" would
    manufacture a person out of a job title.
    """
    decider = _RecordingDecider(CreateEntity(canonical_name="Engineer", disambiguator=None))

    async with session_factory() as session:
        result = await process(
            _services(session, payload=unnamed_payload()),
            content=EVIDENCE_UNNAMED_CONTENT,
            meta=EvidenceMetadata(),
            force_new=False,
            analyze=False,
            decider=decider,
        )

    assert decider.questions == []
    assert await _count(session_factory, Entity) == 0
    assert all(record.outcome == "declined" for record in result.resolutions)
    assert result.resolutions
    assert all(
        "Create the identity deliberately" in "".join(r.follow_up) for r in result.resolutions
    )


async def test_non_interactive_creates_nothing_and_reports_the_commands(
    session_factory: SessionFactory,
) -> None:
    """An unattended run must never make an identity decision on its own."""
    async with session_factory() as session:
        result = await process(
            _services(session, payload=evidence_a_payload()),
            content=EVIDENCE_A_CONTENT,
            meta=EvidenceMetadata(),
            force_new=False,
            analyze=False,
            decider=AlwaysSkip(),
        )

    assert await _count(session_factory, Entity) == 0
    assert await _count(session_factory, EntityResolution) == 0
    assert all(record.outcome == "declined" for record in result.resolutions)

    mention = next(r for r in result.resolutions if r.role is ResolutionRole.MENTION)
    assert any("pie entity new --type person" in command for command in mention.follow_up)
    assert any("pie resolve" in command for command in mention.follow_up)


async def test_reprocessing_the_same_note_writes_nothing_new(
    session_factory: SessionFactory,
) -> None:
    """Re-running is safe, cheap, and does not re-ask a settled question."""
    decider = _RecordingDecider(CreateEntity(canonical_name="Zilan", disambiguator=None))

    async with session_factory() as session:
        await process(
            _services(session, payload=evidence_a_payload()),
            content=EVIDENCE_A_CONTENT,
            meta=EvidenceMetadata(source_type="conversation_note"),
            force_new=False,
            analyze=False,
            decider=decider,
        )

    before = (
        await _count(session_factory, EvidenceItem),
        await _count(session_factory, Entity),
        await _count(session_factory, EntityResolution),
    )

    repeat = _RecordingDecider()
    async with session_factory() as session:
        second = await process(
            _services(session, payload=evidence_a_payload()),
            content=EVIDENCE_A_CONTENT,
            meta=EvidenceMetadata(source_type="conversation_note"),
            force_new=False,
            analyze=False,
            decider=repeat,
        )

    assert second.evidence.reused is True
    assert second.extraction is not None
    assert second.extraction.reused is True
    assert repeat.questions == []
    assert all(record.outcome == "already_resolved" for record in second.resolutions)
    assert (
        await _count(session_factory, EvidenceItem),
        await _count(session_factory, Entity),
        await _count(session_factory, EntityResolution),
    ) == before


async def test_force_new_records_a_second_observation(session_factory: SessionFactory) -> None:
    """The duplicate check is a convenience, not a uniqueness rule."""
    async with session_factory() as session:
        await process(
            _services(session, payload=evidence_a_payload()),
            content=EVIDENCE_A_CONTENT,
            meta=EvidenceMetadata(),
            force_new=False,
            analyze=False,
            decider=AlwaysSkip(),
        )
    async with session_factory() as session:
        second = await process(
            _services(session, payload=evidence_a_payload()),
            content=EVIDENCE_A_CONTENT,
            meta=EvidenceMetadata(),
            force_new=True,
            analyze=False,
            decider=AlwaysSkip(),
        )

    assert second.evidence.reused is False
    assert await _count(session_factory, EvidenceItem) == 2


async def test_a_mention_only_note_spends_nothing_on_analysis(
    session_factory: SessionFactory,
) -> None:
    """Only attributed claims can stand in a relation, so nothing else is analysed."""
    decider = _RecordingDecider(CreateEntity(canonical_name="Zilan", disambiguator=None))
    mention_only = {
        "items": [
            item for item in evidence_a_payload()["items"] if item["kind"] == "entity_mention"
        ]
    }

    async with session_factory() as session:
        result = await process(
            _services(session, payload=mention_only),
            content=EVIDENCE_A_CONTENT,
            meta=EvidenceMetadata(),
            force_new=False,
            analyze=True,
            decider=decider,
        )

    assert result.newly_attributed_entities == []
    assert result.analyses == []
    assert await _count(session_factory, IntelligenceRelation) == 0


async def test_the_second_note_about_one_person_produces_intelligence(
    session_factory: SessionFactory,
) -> None:
    """The whole point: two notes, one identity, relations across them."""
    decider = _RecordingDecider(CreateEntity(canonical_name="Zilan", disambiguator=None))
    async with session_factory() as session:
        await process(
            _services(session, payload=evidence_a_payload()),
            content=EVIDENCE_A_CONTENT,
            meta=EvidenceMetadata(),
            force_new=False,
            analyze=True,
            decider=decider,
        )

    # The name is on file now, so the follow-up note never asks about it again.
    # A genuinely new name in that note (the organisation "Glyde") is still asked
    # about, which is the point: only settled questions stay settled.
    quiet = _RecordingDecider()
    async with session_factory() as session:
        second = await process(
            _services(
                session,
                payload=evidence_b_payload(),
                # Two candidate pairs: evidence A carries two attributed claims
                # and evidence B one, and same-evidence pairs are filtered out.
                classifier_output=classifier_response(qualifies("pair_01"), qualifies("pair_02")),
            ),
            content=EVIDENCE_B_CONTENT,
            meta=EvidenceMetadata(),
            force_new=False,
            analyze=True,
            decider=quiet,
        )

    assert "Zilan" not in {question.name for question in quiet.questions}
    assert len(second.analyses) == 1
    assert second.analyses[0].response.performed_calls is True
    assert second.analyses[0].intelligence.derived_relations
    assert await _count(session_factory, IntelligenceRelation) > 0


async def test_ambiguity_offers_the_candidates_and_asserts_nothing_alone(
    session_factory: SessionFactory,
) -> None:
    """Two people share a name: the CLI must present both, never pick one."""
    async with session_factory() as session:
        entities = EntityService(session)
        first = await entities.create_entity(
            _entity("Zilan", "the one from the meetup"),
        )
        second = await entities.create_entity(
            _entity("Zilan", "different Zilan, from the job fair"),
        )

    decider = _RecordingDecider(UseEntity(second.id))
    async with session_factory() as session:
        result = await process(
            _services(session, payload=evidence_a_payload()),
            content=EVIDENCE_A_CONTENT,
            meta=EvidenceMetadata(),
            force_new=False,
            analyze=False,
            decider=decider,
        )

    question = decider.questions[0]
    assert {candidate.entity_id for candidate in question.candidates} == {first.id, second.id}
    assert isinstance(question.candidates[0], Candidate)

    mention = next(r for r in result.resolutions if r.role is ResolutionRole.MENTION)
    assert mention.outcome == "resolved"
    assert mention.entity_id == second.id
    assert await _count(session_factory, Entity) == 2


def _entity(name: str, disambiguator: str) -> Any:
    from intelligence_engine.entities.schemas import EntityCreate, EntityType

    return EntityCreate(
        entity_type=EntityType.PERSON, canonical_name=name, disambiguator=disambiguator
    )


async def test_an_unknown_entity_id_is_a_clean_failure(session_factory: SessionFactory) -> None:
    """A stale id from an old terminal must not surface as a traceback."""
    from intelligence_engine.entities.errors import EntityNotFound

    async with session_factory() as session:
        with pytest.raises(EntityNotFound):
            await _services(session, payload=evidence_a_payload()).intelligence.analyze(
                uuid.uuid4()
            )


async def test_an_entity_reached_only_through_an_observation_is_still_analysed(
    session_factory: SessionFactory,
) -> None:
    """The third role filter, which fails silently when it is missed.

    `pie process` decides which entities are worth an analysis from the roles it
    just resolved. If that gate stays claim-only, an entity connected purely
    through observations is resolved correctly, shown in the profile, and then
    never analysed - no error, no empty result, just intelligence that is never
    produced. Nothing else in the suite notices, so this test is the guard.
    """
    decider = _RecordingDecider(CreateEntity(canonical_name="Zilan", disambiguator=None))
    observations_only = {
        "items": [
            item
            for item in evidence_c_payload()["items"]
            if item["kind"] in {"entity_mention", "observation"}
        ]
    }

    async with session_factory() as session:
        result = await process(
            _services(session, payload=observations_only),
            content=EVIDENCE_C_CONTENT,
            meta=EvidenceMetadata(),
            force_new=False,
            analyze=True,
            decider=decider,
        )

    about = [record for record in result.resolutions if record.role is ResolutionRole.ABOUT]
    resolved_about = [record for record in about if record.entity_id is not None]

    # Three observations name Zilan; the first-person one resolves to nobody.
    assert len(resolved_about) == 3
    assert result.newly_attributed_entities != [], (
        "an entity connected only through observations was never queued for analysis"
    )
    assert len(result.newly_attributed_entities) == 1
