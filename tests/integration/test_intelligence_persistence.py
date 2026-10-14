"""Cross-evidence intelligence against real PostgreSQL.

Everything here runs through the real migrations, the real service layer and a
deterministic fake provider — no network, no credentials. The assertions are
about derivation semantics and cost, not about model quality; model quality is
what `evaluation/` is for.

The file is organised around the M4 invariants:

- derived relations are structurally separate from grounded information;
- every relation names the exact grounded items it depends on;
- M1, M2 and M3 are never modified;
- unchanged inputs under the current classifier cost nothing;
- a relation known to be wrong stops being served and is not regenerated;
- `no_meaningful_relation` is a stored, reusable, unsurfaced answer;
- malformed output fails safely, and sound work already paid for survives.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelligence_engine.entities.errors import EntityNotFound
from intelligence_engine.entities.models import Entity
from intelligence_engine.entities.schemas import (
    EntityCreate,
    EntityType,
    ResolutionCreate,
    ResolutionRole,
    ResolutionSupersede,
)
from intelligence_engine.entities.service import EntityService
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.evidence.schemas import EvidenceCreate
from intelligence_engine.evidence.service import EvidenceService
from intelligence_engine.extraction.fake_provider import FakeExtractionProvider
from intelligence_engine.extraction.models import ExtractionItem
from intelligence_engine.extraction.schemas import ItemKind
from intelligence_engine.extraction.service import ExtractionService
from intelligence_engine.intelligence import versioning
from intelligence_engine.intelligence.errors import (
    InvalidRelationOutput,
    RelationAlreadyRejected,
    RelationNotFound,
)
from intelligence_engine.intelligence.models import IntelligenceRelation, RelationAnalysisRun
from intelligence_engine.intelligence.repository import IntelligenceRepository
from intelligence_engine.intelligence.schemas import (
    AnalysisStatus,
    RelationDirection,
    RelationReject,
    RelationStatus,
    RelationType,
)
from intelligence_engine.intelligence.service import IntelligenceService
from tests.entity_fixtures import (
    EVIDENCE_A_CONTENT,
    EVIDENCE_B_CONTENT,
    EVIDENCE_C_CONTENT,
    evidence_a_payload,
    evidence_b_payload,
    evidence_c_payload,
)
from tests.intelligence_fixtures import classifier_response, no_relation, qualifies, reinforces

pytestmark = pytest.mark.integration

SessionFactory = async_sessionmaker[AsyncSession]


# --- helpers --------------------------------------------------------------


#: A classifier version deliberately unequal to the running one, for the tests
#: that prove a bump re-classifies rather than reusing. Derived instead of
#: hard-coded: a literal "1.1.0" here silently stopped testing anything the day
#: CLASSIFIER_VERSION became 1.1.0, and the reuse assertion then passed for the
#: wrong reason.
#: The suffix is what makes it safe: derived this way it can never coincide with
#: the running version, so no guard assertion is needed to keep it honest.
BUMPED_VERSION = f"{versioning.CLASSIFIER_VERSION}-bumped-for-test"


async def _ingest(
    factory: SessionFactory, *, content: str, label: str, payload: dict[str, Any]
) -> list[ExtractionItem]:
    """The real M1 -> M2 path, so M4 operates on genuinely grounded material."""
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
        assert result.runs[0].status.value == "succeeded"

    async with factory() as session:
        rows = await session.execute(
            select(ExtractionItem)
            .where(ExtractionItem.extraction_run_id == result.runs[0].id)
            .order_by(ExtractionItem.position)
        )
        return list(rows.scalars().all())


async def _connected_entity(factory: SessionFactory) -> tuple[uuid.UUID, list[ExtractionItem]]:
    """Two notes about one person, extracted and fully resolved through M3.

    Evidence A carries the advice and the query-planner claim; evidence B the
    six-weeks-of-reading claim. Three attributed claims, three theoretical pairs,
    two of them cross-evidence.
    """
    items_a = await _ingest(
        factory,
        content=EVIDENCE_A_CONTENT,
        label="Kestrel Python meetup",
        payload=evidence_a_payload(),
    )
    items_b = await _ingest(
        factory, content=EVIDENCE_B_CONTENT, label="Coffee, Brookside", payload=evidence_b_payload()
    )

    async with factory() as session:
        entity = await EntityService(session).create_entity(
            EntityCreate(entity_type=EntityType.PERSON, canonical_name="Zilan")
        )
    entity_id = entity.id

    claims = [item for item in (*items_a, *items_b) if item.kind is ItemKind.REPORTED_CLAIM]
    for claim in claims:
        async with factory() as session:
            outcome = await EntityService(session).resolve(
                claim.id, ResolutionCreate(role=ResolutionRole.CLAIMED_BY)
            )
            assert outcome.status == "resolved"

    return entity_id, claims


def _service(
    session: AsyncSession,
    provider: FakeExtractionProvider,
    **overrides: Any,
) -> IntelligenceService:
    return IntelligenceService(session, provider, **overrides)


def _all_qualify(count: int) -> str:
    return classifier_response(*[qualifies(f"pair_{n:02d}") for n in range(1, count + 1)])


def _all_decline(count: int) -> str:
    return classifier_response(*[no_relation(f"pair_{n:02d}") for n in range(1, count + 1)])


# --- the derivation happens, and it is provenanced ------------------------


async def test_analysis_derives_relations_from_connected_claims(
    session_factory: SessionFactory,
) -> None:
    entity_id, claims = await _connected_entity(session_factory)
    provider = FakeExtractionProvider(raw_output=_all_qualify(2))

    async with session_factory() as session:
        response = await _service(session, provider).analyze(entity_id)

    assert response.performed_calls is True
    assert response.telemetry.theoretical_pairs == 3
    assert response.telemetry.candidate_pairs == 2
    assert response.telemetry.classified_pairs == 2
    assert response.telemetry.new_relations == 2
    assert len(response.relations) == 2
    assert {relation.relation_type for relation in response.relations} == {RelationType.QUALIFIES}
    assert len(claims) == 3


async def test_every_relation_names_its_exact_source_items(
    session_factory: SessionFactory,
) -> None:
    """Invariant 4. A derived assertion that cannot be traced is not evidence."""
    entity_id, claims = await _connected_entity(session_factory)
    claim_ids = {claim.id for claim in claims}

    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        rows = (await session.execute(select(IntelligenceRelation))).scalars().all()

    assert rows
    for relation in rows:
        assert relation.left_extraction_item_id in claim_ids
        assert relation.right_extraction_item_id in claim_ids
        assert relation.left_extraction_item_id < relation.right_extraction_item_id
        assert relation.entity_id == entity_id
        assert relation.classifier_name == "relation_classifier"
        assert relation.provider == "fake"
        assert len(relation.input_fingerprint) == 64


async def test_pairs_never_come_from_the_same_evidence_item(
    session_factory: SessionFactory,
) -> None:
    """Cross-evidence is the definition, not an optimisation."""
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT left_run.evidence_id, right_run.evidence_id "
                    "FROM intelligence_relations r "
                    "JOIN extraction_items li ON li.id = r.left_extraction_item_id "
                    "JOIN extraction_runs left_run ON left_run.id = li.extraction_run_id "
                    "JOIN extraction_items ri ON ri.id = r.right_extraction_item_id "
                    "JOIN extraction_runs right_run ON right_run.id = ri.extraction_run_id"
                )
            )
        ).all()

    assert rows
    assert all(row[0] != row[1] for row in rows)


async def test_the_run_records_what_the_analysis_cost(session_factory: SessionFactory) -> None:
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        response = await _service(
            session, FakeExtractionProvider(raw_output=_all_qualify(2))
        ).analyze(entity_id)

    async with session_factory() as session:
        run = await session.get(RelationAnalysisRun, response.analysis_run_id)

    assert run is not None
    assert run.status is AnalysisStatus.SUCCEEDED
    assert run.completed_at is not None
    assert run.theoretical_pairs == 3
    assert run.candidate_pairs == 2
    assert run.classified_pairs == 2
    assert run.batch_count == 1
    assert run.external_call_count == 1
    assert run.input_tokens == 100
    assert run.output_tokens == 50


async def test_an_entity_with_nothing_to_relate_costs_nothing(
    session_factory: SessionFactory,
) -> None:
    """One evidence item cannot produce a cross-evidence pair."""
    items = await _ingest(
        session_factory,
        content=EVIDENCE_A_CONTENT,
        label="Kestrel Python meetup",
        payload=evidence_a_payload(),
    )
    async with session_factory() as session:
        entity = await EntityService(session).create_entity(
            EntityCreate(entity_type=EntityType.PERSON, canonical_name="Zilan")
        )
    for claim in [item for item in items if item.kind is ItemKind.REPORTED_CLAIM]:
        async with session_factory() as session:
            await EntityService(session).resolve(
                claim.id, ResolutionCreate(role=ResolutionRole.CLAIMED_BY)
            )

    provider = FakeExtractionProvider(raw_output=_all_qualify(1))
    async with session_factory() as session:
        response = await _service(session, provider).analyze(entity.id)

    assert provider.call_count == 0
    assert response.performed_calls is False
    assert response.analysis_run_id is None
    assert response.telemetry.candidate_pairs == 0

    async with session_factory() as session:
        # No model call means no run row: there is nothing to record but a
        # decision not to spend.
        assert await session.scalar(select(func.count()).select_from(RelationAnalysisRun)) == 0


async def test_an_unknown_entity_is_a_404_not_an_empty_result(
    session_factory: SessionFactory,
) -> None:
    async with session_factory() as session:
        with pytest.raises(EntityNotFound):
            await _service(session, FakeExtractionProvider()).analyze(uuid.uuid4())


# --- what the analysis did not look at ------------------------------------


async def test_telemetry_reports_the_entity_population_not_only_what_was_loaded(
    session_factory: SessionFactory,
) -> None:
    """The item limit discards evidence, and the discarding must be visible.

    `MAX_RELATABLE_ITEMS` is 200 and `list_relatable_items` orders oldest-first,
    so an entity past 200 relatable items is analysed on its 200 oldest and never
    told. Worse, `theoretical_pairs` is computed over the loaded rows, so the one
    number that looked like a population figure was itself a product of the
    truncation — a telemetry field derived from the truncated set cannot describe
    the truncation.

    The limit is injected here rather than built up to: 201 resolutions would
    make this a slow test, and slow tests for reporting behaviour do not get
    written. The mechanism under test is the same one at either bound.
    """
    entity_id, claims = await _connected_entity(session_factory)
    assert len(claims) == 3

    provider = FakeExtractionProvider(raw_output=_all_decline(1))
    async with session_factory() as session:
        response = await _service(session, provider, max_relatable_items=2).analyze(entity_id)

    telemetry = response.telemetry
    assert telemetry.relatable_items_total == 3
    assert telemetry.relatable_items_loaded == 2
    assert telemetry.relatable_items_truncated is True

    # And the pair figure is over the loaded subset, which is now sayable rather
    # than something a reader has to infer.
    assert telemetry.theoretical_pairs == 1


async def test_an_untruncated_analysis_says_so(
    session_factory: SessionFactory,
) -> None:
    """The control: `truncated` must be a fact about this run, not a constant."""
    entity_id, claims = await _connected_entity(session_factory)

    provider = FakeExtractionProvider(raw_output=_all_decline(2))
    async with session_factory() as session:
        response = await _service(session, provider).analyze(entity_id)

    telemetry = response.telemetry
    assert telemetry.relatable_items_total == len(claims)
    assert telemetry.relatable_items_loaded == len(claims)
    assert telemetry.relatable_items_truncated is False


async def test_truncation_is_reported_even_when_nothing_is_classified(
    session_factory: SessionFactory,
) -> None:
    """A fully reused analysis writes no run row, and must still report the loss.

    This is the path where the omission would have been most durable: no
    `relation_analysis_runs` row is written at all, so the response is the only
    place the fact could appear.
    """
    entity_id, _claims = await _connected_entity(session_factory)

    first = FakeExtractionProvider(raw_output=_all_decline(1))
    async with session_factory() as session:
        await _service(session, first, max_relatable_items=2).analyze(entity_id)

    second = FakeExtractionProvider(raw_output=_all_decline(1))
    async with session_factory() as session:
        response = await _service(session, second, max_relatable_items=2).analyze(entity_id)

    assert second.call_count == 0
    assert response.analysis_run_id is None
    assert response.telemetry.relatable_items_total == 3
    assert response.telemetry.relatable_items_loaded == 2
    assert response.telemetry.relatable_items_truncated is True


async def test_the_truncated_items_are_the_newest_ones(
    session_factory: SessionFactory,
) -> None:
    """Names the cost of the current ordering rather than leaving it implied.

    Oldest-first plus a limit keeps the *oldest* evidence, so what an entity past
    the bound loses is everything it learned most recently. Whether that is the
    right policy is not decided here; that it is what happens is pinned, so a
    change of ordering cannot pass unnoticed.
    """
    entity_id, claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        rows = await IntelligenceRepository(session).list_relatable_items(entity_id, limit=2)
        loaded = [item.id for _resolution, item, _run, _evidence in rows]

    async with session_factory() as session:
        every = await IntelligenceRepository(session).list_relatable_items(entity_id)
        ordered = [item.id for _resolution, item, _run, _evidence in every]

    assert len(ordered) == len(claims)
    assert loaded == ordered[:2]
    assert ordered[-1] not in loaded


# --- cost: unchanged inputs are free --------------------------------------


async def test_a_second_identical_analysis_makes_no_provider_call(
    session_factory: SessionFactory,
) -> None:
    """The cost invariant, proven rather than asserted."""
    entity_id, _claims = await _connected_entity(session_factory)

    first_provider = FakeExtractionProvider(raw_output=_all_qualify(2))
    async with session_factory() as session:
        first = await _service(session, first_provider).analyze(entity_id)

    second_provider = FakeExtractionProvider(raw_output=_all_qualify(2))
    async with session_factory() as session:
        second = await _service(session, second_provider).analyze(entity_id)

    assert first_provider.call_count == 1
    assert second_provider.call_count == 0
    assert second.performed_calls is False
    assert second.analysis_run_id is None
    assert second.telemetry.reused_pairs == 2
    assert second.telemetry.classified_pairs == 0
    assert {relation.id for relation in second.relations} == {
        relation.id for relation in first.relations
    }

    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(IntelligenceRelation)) == 2
        assert await session.scalar(select(func.count()).select_from(RelationAnalysisRun)) == 1


async def test_declining_is_persisted_so_it_is_never_paid_for_twice(
    session_factory: SessionFactory,
) -> None:
    """`no_meaningful_relation` will be the majority answer. Caching it is the point."""
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_decline(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        stored = (await session.execute(select(IntelligenceRelation))).scalars().all()
    assert len(stored) == 2
    assert all(row.relation_type is RelationType.NO_MEANINGFUL_RELATION for row in stored)
    assert all(row.explanation is None for row in stored)

    repeat = FakeExtractionProvider(raw_output=_all_decline(2))
    async with session_factory() as session:
        response = await _service(session, repeat).analyze(entity_id)

    assert repeat.call_count == 0
    assert response.telemetry.reused_pairs == 2


async def test_only_new_material_is_classified_when_evidence_arrives(
    session_factory: SessionFactory,
) -> None:
    """Adding a note must not re-buy the answers already on file."""
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_decline(2))).analyze(
            entity_id
        )

    third = await _ingest(
        session_factory,
        content=(
            "Third note about Zilan. He said the reading phase is where most people quit, "
            "and that he now refuses to recommend the approach without saying so."
        ),
        label="Third note",
        payload={
            "items": [
                {
                    "kind": "reported_claim",
                    "data": {
                        "claim": "He refuses to recommend the approach without warning about "
                        "the reading phase.",
                        "claimed_by": "Zilan",
                        "stance": "asserted",
                    },
                    "grounding": {
                        "quote": "he now refuses to recommend the approach without saying so"
                    },
                }
            ]
        },
    )
    async with session_factory() as session:
        await EntityService(session).resolve(
            third[0].id, ResolutionCreate(role=ResolutionRole.CLAIMED_BY)
        )

    provider = FakeExtractionProvider(raw_output=_all_decline(3))
    async with session_factory() as session:
        response = await _service(session, provider).analyze(entity_id)

    # Four claims: 6 theoretical pairs, 5 cross-evidence, 2 already settled.
    assert response.telemetry.theoretical_pairs == 6
    assert response.telemetry.candidate_pairs == 5
    assert response.telemetry.reused_pairs == 2
    assert response.telemetry.classified_pairs == 3
    assert provider.call_count == 1


async def test_a_classifier_version_bump_reclassifies(session_factory: SessionFactory) -> None:
    """A behaviour change must not serve output the old prompt produced."""
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_decline(2))).analyze(
            entity_id
        )

    bumped = FakeExtractionProvider(raw_output=_all_qualify(2))
    async with session_factory() as session:
        response = await _service(session, bumped, classifier_version=BUMPED_VERSION).analyze(
            entity_id
        )

    assert bumped.call_count == 1
    assert response.telemetry.reused_pairs == 0
    assert response.telemetry.classified_pairs == 2

    async with session_factory() as session:
        # Both generations survive: the old one as history, the new one current.
        assert await session.scalar(select(func.count()).select_from(IntelligenceRelation)) == 4


async def test_a_schema_version_bump_also_reclassifies(session_factory: SessionFactory) -> None:
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_decline(2))).analyze(
            entity_id
        )

    bumped = FakeExtractionProvider(raw_output=_all_decline(2))
    async with session_factory() as session:
        await _service(session, bumped, schema_version=2).analyze(entity_id)

    assert bumped.call_count == 1


# --- batching and bounds --------------------------------------------------


async def test_candidate_generation_is_capped(session_factory: SessionFactory) -> None:
    entity_id, _claims = await _connected_entity(session_factory)

    provider = FakeExtractionProvider(raw_output=_all_qualify(1))
    async with session_factory() as session:
        response = await _service(session, provider, max_candidate_pairs=1).analyze(entity_id)

    assert response.telemetry.theoretical_pairs == 3
    assert response.telemetry.candidate_pairs == 1
    assert response.telemetry.classified_pairs == 1
    assert provider.call_count == 1


async def test_pairs_are_split_into_bounded_batches(session_factory: SessionFactory) -> None:
    entity_id, _claims = await _connected_entity(session_factory)

    provider = FakeExtractionProvider(raw_output=_all_decline(1))
    async with session_factory() as session:
        response = await _service(session, provider, batch_size=1).analyze(entity_id)

    assert response.telemetry.batch_count == 2
    assert response.telemetry.external_call_count == 2
    assert provider.call_count == 2
    # Tokens accumulate across calls rather than reporting only the last one.
    assert response.telemetry.input_tokens == 200
    assert response.telemetry.output_tokens == 100


# --- malformed output fails safely ----------------------------------------


async def test_a_malformed_response_persists_no_relations(
    session_factory: SessionFactory,
) -> None:
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        with pytest.raises(InvalidRelationOutput):
            await _service(
                session, FakeExtractionProvider(raw_output="I found some interesting links!")
            ).analyze(entity_id)

    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(IntelligenceRelation)) == 0
        run = (await session.execute(select(RelationAnalysisRun))).scalars().one()

    assert run.status is AnalysisStatus.FAILED
    assert run.error_code == "invalid_relation_output"
    assert run.completed_at is not None


async def test_a_hallucinated_pair_id_rejects_the_batch(session_factory: SessionFactory) -> None:
    """The M4 equivalent of M2's verbatim-quote check.

    An explanation bound to the wrong two claims would look perfectly well-formed
    in storage and no reader could ever detect it.
    """
    entity_id, _claims = await _connected_entity(session_factory)
    lying = classifier_response(qualifies("pair_01"), qualifies("pair_77"))

    async with session_factory() as session:
        with pytest.raises(InvalidRelationOutput, match="not supplied"):
            await _service(session, FakeExtractionProvider(raw_output=lying)).analyze(entity_id)

    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(IntelligenceRelation)) == 0


async def test_an_incomplete_batch_rejects_the_batch(session_factory: SessionFactory) -> None:
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        with pytest.raises(InvalidRelationOutput, match="not answered"):
            await _service(
                session,
                FakeExtractionProvider(raw_output=classifier_response(qualifies("pair_01"))),
            ).analyze(entity_id)


async def test_a_failed_batch_keeps_the_batches_already_paid_for(
    session_factory: SessionFactory,
) -> None:
    """The deliberate difference from M2's all-or-nothing run.

    Batches are independent: pair 2 being answered badly says nothing about pair
    1, which is individually valid and individually reusable. Discarding it would
    mean paying for it again.
    """
    entity_id, _claims = await _connected_entity(session_factory)

    class _SecondBatchFails(FakeExtractionProvider):
        async def extract(self, **kwargs: Any) -> Any:
            result = await super().extract(**kwargs)
            if self.call_count == 1:
                return result
            return type(result)(
                raw_output="not json at all",
                provider=result.provider,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

    provider = _SecondBatchFails(raw_output=_all_decline(1))
    async with session_factory() as session:
        with pytest.raises(InvalidRelationOutput):
            await _service(session, provider, batch_size=1).analyze(entity_id)

    assert provider.call_count == 2
    async with session_factory() as session:
        stored = (await session.execute(select(IntelligenceRelation))).scalars().all()
        run = (await session.execute(select(RelationAnalysisRun))).scalars().one()

    assert len(stored) == 1
    assert run.status is AnalysisStatus.FAILED
    assert run.external_call_count == 1

    # And the survivor is genuinely reusable rather than orphaned.
    retry = FakeExtractionProvider(raw_output=_all_decline(1))
    async with session_factory() as session:
        response = await _service(session, retry, batch_size=1).analyze(entity_id)
    assert response.telemetry.reused_pairs == 1
    assert retry.call_count == 1


# --- review: a wrong relation stops being served --------------------------


async def test_a_rejected_relation_is_withdrawn_but_not_deleted(
    session_factory: SessionFactory,
) -> None:
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        response = await _service(
            session, FakeExtractionProvider(raw_output=_all_qualify(2))
        ).analyze(entity_id)
    target = response.relations[0]

    async with session_factory() as session:
        rejected = await _service(session, FakeExtractionProvider()).reject_relation(
            target.id,
            RelationReject(note="Different projects entirely; this is not a qualification."),
        )

    assert rejected.status is RelationStatus.REJECTED
    assert rejected.rejected_at is not None
    assert rejected.rejection_note is not None
    # The assertion itself is untouched - the row stays a faithful record.
    assert rejected.relation_type == target.relation_type
    assert rejected.explanation == target.explanation
    assert rejected.created_at == target.created_at

    async with session_factory() as session:
        view = await _service(session, FakeExtractionProvider()).get_intelligence(entity_id)
    assert target.id not in {item.relation.id for item in view.derived_relations}

    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(IntelligenceRelation)) == 2


async def test_a_rejected_relation_is_not_regenerated(session_factory: SessionFactory) -> None:
    """Otherwise the next analysis would pay for the same wrong answer."""
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        response = await _service(
            session, FakeExtractionProvider(raw_output=_all_qualify(2))
        ).analyze(entity_id)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider()).reject_relation(
            response.relations[0].id, RelationReject(note="Wrong.")
        )

    provider = FakeExtractionProvider(raw_output=_all_qualify(2))
    async with session_factory() as session:
        repeat = await _service(session, provider).analyze(entity_id)

    assert provider.call_count == 0
    assert repeat.telemetry.reused_pairs == 2


async def test_a_version_bump_re_opens_a_rejected_pair(session_factory: SessionFactory) -> None:
    """The escape hatch: a better classifier gets a fresh attempt at the question."""
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        response = await _service(
            session, FakeExtractionProvider(raw_output=_all_qualify(2))
        ).analyze(entity_id)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider()).reject_relation(
            response.relations[0].id, RelationReject(note="Wrong.")
        )

    provider = FakeExtractionProvider(raw_output=_all_decline(2))
    async with session_factory() as session:
        await _service(session, provider, classifier_version=BUMPED_VERSION).analyze(entity_id)

    assert provider.call_count == 1


async def test_rejecting_twice_is_refused(session_factory: SessionFactory) -> None:
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        response = await _service(
            session, FakeExtractionProvider(raw_output=_all_qualify(2))
        ).analyze(entity_id)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider()).reject_relation(
            response.relations[0].id, RelationReject(note="Wrong.")
        )

    async with session_factory() as session:
        with pytest.raises(RelationAlreadyRejected):
            await _service(session, FakeExtractionProvider()).reject_relation(
                response.relations[0].id, RelationReject(note="Still wrong.")
            )


async def test_rejecting_an_unknown_relation_is_a_404(session_factory: SessionFactory) -> None:
    async with session_factory() as session:
        with pytest.raises(RelationNotFound):
            await _service(session, FakeExtractionProvider()).reject_relation(
                uuid.uuid4(), RelationReject()
            )


# --- the read model -------------------------------------------------------


async def test_the_view_separates_derived_relations_from_grounded_claims(
    session_factory: SessionFactory,
) -> None:
    """Invariant 5. A reader must always be able to tell which is which."""
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )
    async with session_factory() as session:
        view = await _service(session, FakeExtractionProvider()).get_intelligence(entity_id)

    assert len(view.grounded_claims) == 3
    assert len(view.derived_relations) == 2

    for derived in view.derived_relations:
        assert derived.relation.explanation
        assert derived.left.grounding_quote
        assert derived.right.grounding_quote
        assert derived.left.evidence_id != derived.right.evidence_id
        assert derived.relation.classifier_version == versioning.CLASSIFIER_VERSION
        assert derived.reading in {"left qualifies right", "right qualifies left"}


async def test_no_meaningful_relation_is_stored_but_never_surfaced(
    session_factory: SessionFactory,
) -> None:
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_decline(2))).analyze(
            entity_id
        )
    async with session_factory() as session:
        view = await _service(session, FakeExtractionProvider()).get_intelligence(entity_id)
        stored = await session.scalar(select(func.count()).select_from(IntelligenceRelation))

    assert stored == 2
    assert view.derived_relations == []


async def test_output_from_a_retired_classifier_version_is_not_served(
    session_factory: SessionFactory,
) -> None:
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        await _service(
            session, FakeExtractionProvider(raw_output=_all_qualify(2)), classifier_version="0.9.0"
        ).analyze(entity_id)

    async with session_factory() as session:
        view = await _service(session, FakeExtractionProvider()).get_intelligence(entity_id)

    assert view.derived_relations == []


async def test_detaching_a_claim_in_m3_withdraws_the_relations_built_on_it(
    session_factory: SessionFactory,
) -> None:
    """An M3 correction propagates into M4 with M4 writing nothing.

    A derived assertion must not outlive the connection it was derived from.
    """
    entity_id, claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )
    async with session_factory() as session:
        before = await _service(session, FakeExtractionProvider()).get_intelligence(entity_id)
    assert len(before.derived_relations) == 2

    async with session_factory() as session:
        history = await EntityService(session).list_resolutions(claims[0].id)
        active = next(row for row in history.items if row.is_active)
    async with session_factory() as session:
        await EntityService(session).supersede(
            active.id, ResolutionSupersede(note="Not actually his claim.")
        )

    async with session_factory() as session:
        after = await _service(session, FakeExtractionProvider()).get_intelligence(entity_id)

    assert len(after.derived_relations) < len(before.derived_relations)
    async with session_factory() as session:
        # Withdrawn from the view, not deleted from the table.
        assert await session.scalar(select(func.count()).select_from(IntelligenceRelation)) == 2


async def test_the_view_carries_no_generated_narrative(session_factory: SessionFactory) -> None:
    """No biography, no summary, no AI prose dressed as a finding."""
    entity_id, _claims = await _connected_entity(session_factory)
    async with session_factory() as session:
        view = await _service(session, FakeExtractionProvider()).get_intelligence(entity_id)

    fields = set(type(view).model_fields)
    assert fields == {"entity", "grounded_claims", "derived_relations"}
    for forbidden in ("summary", "biography", "narrative", "profile_text", "overview"):
        assert forbidden not in fields


# --- M1/M2/M3 are never modified ------------------------------------------


async def test_analysis_modifies_nothing_beneath_it(session_factory: SessionFactory) -> None:
    """Invariants 1-3, checked against the bytes rather than against intent."""
    entity_id, claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        before_items = {
            item.id: (item.kind, item.payload, item.grounding_quote, item.position)
            for item in (await session.execute(select(ExtractionItem))).scalars().all()
        }
        before_resolutions = await session.scalar(
            text("SELECT count(*) FROM entity_resolutions WHERE is_active")
        )

    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        # PostgreSQL recomputes the fingerprint; the evidence must still match it.
        mismatches = await session.scalar(
            text(
                "SELECT count(*) FROM evidence_items "
                "WHERE content_hash <> encode(sha256(content::bytea), 'hex')"
            )
        )
        after_items = {
            item.id: (item.kind, item.payload, item.grounding_quote, item.position)
            for item in (await session.execute(select(ExtractionItem))).scalars().all()
        }
        after_resolutions = await session.scalar(
            text("SELECT count(*) FROM entity_resolutions WHERE is_active")
        )
        evidence_count = await session.scalar(select(func.count()).select_from(EvidenceItem))

    assert mismatches == 0
    assert after_items == before_items
    assert after_resolutions == before_resolutions
    assert evidence_count == 2
    assert len(claims) == 3


# --- database-level truths ------------------------------------------------


async def _stage(session: AsyncSession, **overrides: Any) -> IntelligenceRelation:
    """A minimal valid relation row, with fields overridden to test one CHECK."""
    entity_id = (await session.execute(select(Entity.id))).scalars().first()
    run_id = (await session.execute(select(RelationAnalysisRun.id))).scalars().first()
    items = (
        (
            await session.execute(
                select(ExtractionItem).where(ExtractionItem.kind == "reported_claim")
            )
        )
        .scalars()
        .all()
    )
    ordered = sorted(items, key=lambda item: item.id)
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "analysis_run_id": run_id,
        "entity_id": entity_id,
        "left_extraction_item_id": ordered[0].id,
        "right_extraction_item_id": ordered[-1].id,
        "left_item_kind": ordered[0].kind,
        "right_item_kind": ordered[-1].kind,
        "relation_type": RelationType.REINFORCES,
        "direction": RelationDirection.SYMMETRIC,
        "explanation": "The second supplies concrete support for the first.",
        "classifier_name": "manual_test",
        "classifier_version": "9.9.9",
        "schema_version": 1,
        # Required since the generating identity joined the reuse key and the
        # unique constraint. Constants rather than per-call values: every caller
        # is testing some *other* constraint, and two rows differing here would
        # stop colliding on the one they mean to test.
        "provider": "manual_test_provider",
        "model": "manual-test-model",
        "input_fingerprint": "a" * 64,
        "status": RelationStatus.CURRENT,
    }
    relation = IntelligenceRelation(**{**defaults, **overrides})
    session.add(relation)
    return relation


async def test_a_reversed_pair_is_refused_by_the_database(
    session_factory: SessionFactory,
) -> None:
    """Proving the gate rejects. Without this, one pair could hold two rows."""
    entity_id, _claims = await _connected_entity(session_factory)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        items = sorted(
            (
                await session.execute(
                    select(ExtractionItem).where(ExtractionItem.kind == "reported_claim")
                )
            )
            .scalars()
            .all(),
            key=lambda item: item.id,
        )
        await _stage(
            session,
            left_extraction_item_id=items[-1].id,
            right_extraction_item_id=items[0].id,
        )
        with pytest.raises(IntegrityError, match="pair_is_canonically_ordered"):
            await session.commit()


async def test_a_relation_to_itself_is_refused(session_factory: SessionFactory) -> None:
    entity_id, _claims = await _connected_entity(session_factory)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        items = (
            (
                await session.execute(
                    select(ExtractionItem).where(ExtractionItem.kind == "reported_claim")
                )
            )
            .scalars()
            .all()
        )
        await _stage(
            session,
            left_extraction_item_id=items[0].id,
            right_extraction_item_id=items[0].id,
            left_item_kind=items[0].kind,
            right_item_kind=items[0].kind,
        )
        with pytest.raises(IntegrityError, match="pair_is_canonically_ordered"):
            await session.commit()


async def test_a_symmetric_relation_may_not_state_a_direction(
    session_factory: SessionFactory,
) -> None:
    entity_id, _claims = await _connected_entity(session_factory)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        await _stage(
            session,
            relation_type=RelationType.CONTRADICTS,
            direction=RelationDirection.LEFT_TO_RIGHT,
        )
        with pytest.raises(IntegrityError, match="direction_matches_relation_type"):
            await session.commit()


async def test_a_qualification_must_state_a_direction(session_factory: SessionFactory) -> None:
    entity_id, _claims = await _connected_entity(session_factory)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        await _stage(
            session,
            relation_type=RelationType.QUALIFIES,
            direction=RelationDirection.SYMMETRIC,
        )
        with pytest.raises(IntegrityError, match="direction_matches_relation_type"):
            await session.commit()


async def test_a_meaningful_relation_cannot_be_stored_unexplained(
    session_factory: SessionFactory,
) -> None:
    entity_id, _claims = await _connected_entity(session_factory)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        await _stage(session, explanation=None)
        with pytest.raises(IntegrityError, match="meaningful_relation_is_explained"):
            await session.commit()


async def test_a_mention_cannot_stand_in_a_relation(session_factory: SessionFactory) -> None:
    """Naming a thing cannot reinforce, qualify or contradict anything."""
    entity_id, _claims = await _connected_entity(session_factory)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        mention = (
            (
                await session.execute(
                    select(ExtractionItem).where(ExtractionItem.kind == "entity_mention")
                )
            )
            .scalars()
            .first()
        )
        assert mention is not None
        claim = (
            (
                await session.execute(
                    select(ExtractionItem).where(ExtractionItem.kind == "reported_claim")
                )
            )
            .scalars()
            .first()
        )
        assert claim is not None
        left, right = sorted((mention, claim), key=lambda item: item.id)
        await _stage(
            session,
            left_extraction_item_id=left.id,
            right_extraction_item_id=right.id,
            left_item_kind=left.kind,
            right_item_kind=right.kind,
        )
        with pytest.raises(IntegrityError, match="item_kind_is_relatable"):
            await session.commit()


async def test_the_denormalised_kind_cannot_disagree_with_the_item(
    session_factory: SessionFactory,
) -> None:
    """The M3 learning applied again: the composite FK is what makes the CHECK true."""
    entity_id, _claims = await _connected_entity(session_factory)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        await _stage(session, left_item_kind=ItemKind.OBSERVATION)
        with pytest.raises(IntegrityError, match="fk_intelligence_relations_left_item"):
            await session.commit()


async def test_one_settled_answer_per_pair_per_classifier_version(
    session_factory: SessionFactory,
) -> None:
    entity_id, _claims = await _connected_entity(session_factory)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        # Duplicate a pair that was genuinely stored, rather than whichever pair
        # `_stage` would pick by uuid order - only two of the three possible
        # pairs are cross-evidence, so the third would not collide and the test
        # would pass or fail depending on how the uuids happened to sort.
        existing = (await session.execute(select(IntelligenceRelation))).scalars().first()
        assert existing is not None
        await _stage(
            session,
            left_extraction_item_id=existing.left_extraction_item_id,
            right_extraction_item_id=existing.right_extraction_item_id,
            left_item_kind=existing.left_item_kind,
            right_item_kind=existing.right_item_kind,
            classifier_name=existing.classifier_name,
            classifier_version=existing.classifier_version,
            schema_version=existing.schema_version,
            # The generator is part of the key now, so it has to be copied too.
            # Leaving `_stage`'s own defaults here would make the second row a
            # different model's answer, which is legitimately not a duplicate -
            # and the test would pass or fail for a reason it is not about.
            provider=existing.provider,
            model=existing.model,
        )
        with pytest.raises(IntegrityError, match="unique_settled_pair_per_classifier"):
            await session.commit()


async def test_two_models_may_each_hold_one_settled_answer_for_a_pair(
    session_factory: SessionFactory,
) -> None:
    """The other half of the constraint, and the reason it had to widen.

    One answer per pair per classifier version *per generator*. Two models
    answering the same pair is not a duplicate — it is what a model comparison
    is — and under the narrower constraint it was an integrity error, which made
    the comparison impossible rather than merely unreliable.
    """
    entity_id, _claims = await _connected_entity(session_factory)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        existing = (await session.execute(select(IntelligenceRelation))).scalars().first()
        assert existing is not None
        await _stage(
            session,
            left_extraction_item_id=existing.left_extraction_item_id,
            right_extraction_item_id=existing.right_extraction_item_id,
            left_item_kind=existing.left_item_kind,
            right_item_kind=existing.right_item_kind,
            classifier_name=existing.classifier_name,
            classifier_version=existing.classifier_version,
            schema_version=existing.schema_version,
            provider=existing.provider,
            model=f"{existing.model}-a-different-model",
        )
        await session.commit()

    async with session_factory() as session:
        stored = (
            await session.execute(
                select(IntelligenceRelation).where(
                    IntelligenceRelation.left_extraction_item_id
                    == existing.left_extraction_item_id,
                    IntelligenceRelation.right_extraction_item_id
                    == existing.right_extraction_item_id,
                )
            )
        ).scalars()
        assert len({relation.model for relation in stored}) == 2


async def test_an_entity_with_derived_history_cannot_be_deleted(
    session_factory: SessionFactory,
) -> None:
    entity_id, _claims = await _connected_entity(session_factory)
    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(2))).analyze(
            entity_id
        )

    async with session_factory() as session:
        with pytest.raises(IntegrityError):
            await session.execute(
                text("DELETE FROM entities WHERE id = :id"), {"id": str(entity_id)}
            )
            await session.commit()


# --- false relations: what must never be manufactured ---------------------


async def test_a_declined_pair_produces_no_intelligence(
    session_factory: SessionFactory,
) -> None:
    """Mandatory regression: unrelated information -> no meaningful relation."""
    entity_id, _claims = await _connected_entity(session_factory)

    async with session_factory() as session:
        response = await _service(
            session, FakeExtractionProvider(raw_output=_all_decline(2))
        ).analyze(entity_id)

    assert response.telemetry.new_meaningful_relations == 0
    async with session_factory() as session:
        view = await _service(session, FakeExtractionProvider()).get_intelligence(entity_id)
    assert view.derived_relations == []


async def test_a_relation_is_never_manufactured_without_the_classifier_asserting_it(
    session_factory: SessionFactory,
) -> None:
    """Nothing in the pipeline invents a relation. Candidates are not findings."""
    entity_id, _claims = await _connected_entity(session_factory)

    mixed = classifier_response(reinforces("pair_01"), no_relation("pair_02"))
    async with session_factory() as session:
        response = await _service(session, FakeExtractionProvider(raw_output=mixed)).analyze(
            entity_id
        )

    assert response.telemetry.candidate_pairs == 2
    assert response.telemetry.new_relations == 2
    assert response.telemetry.new_meaningful_relations == 1


# --- observations reaching M4 through the `about` role ----------------------
#
# The three sites that filter on resolution role fail in different ways. Only
# one of them fails loudly, so each is exercised here against a real database
# rather than trusted because the constant looks right.


async def _entity_with_claims_and_observations(
    factory: SessionFactory,
) -> tuple[uuid.UUID, list[ExtractionItem], list[ExtractionItem]]:
    """One person who both said things and had things observed about them.

    Evidence A and B carry attributed claims; evidence C is a public profile
    that carries observations. Nobody asserted the evidence C material, which is
    the whole point of pairing it against what the person actually said.
    """
    items_a = await _ingest(
        factory,
        content=EVIDENCE_A_CONTENT,
        label="Kestrel Python meetup",
        payload=evidence_a_payload(),
    )
    items_c = await _ingest(
        factory, content=EVIDENCE_C_CONTENT, label="Public profile", payload=evidence_c_payload()
    )

    async with factory() as session:
        entity = await EntityService(session).create_entity(
            EntityCreate(entity_type=EntityType.PERSON, canonical_name="Zilan")
        )
    entity_id = entity.id

    claims = [item for item in items_a if item.kind is ItemKind.REPORTED_CLAIM]
    for claim in claims:
        async with factory() as session:
            outcome = await EntityService(session).resolve(
                claim.id, ResolutionCreate(role=ResolutionRole.CLAIMED_BY)
            )
            assert outcome.status == "resolved"

    resolved_observations: list[ExtractionItem] = []
    for item in items_c:
        if item.kind is not ItemKind.OBSERVATION:
            continue
        async with factory() as session:
            outcome = await EntityService(session).resolve(
                item.id, ResolutionCreate(role=ResolutionRole.ABOUT)
            )
        if outcome.status == "resolved":
            resolved_observations.append(item)

    # Three of the four observations name Zilan; the first-person one does not.
    assert len(resolved_observations) == 3
    return entity_id, claims, resolved_observations


async def test_an_observation_becomes_relatable_material(
    session_factory: SessionFactory,
) -> None:
    """`list_relatable_items` must return observations, not only claims.

    This is the loud failure of the three: if the role filter here stays
    claim-only, the entity simply has less material and no relation is ever
    proposed.
    """
    entity_id, claims, observations = await _entity_with_claims_and_observations(session_factory)

    async with session_factory() as session:
        rows = await IntelligenceRepository(session).list_relatable_items(entity_id)

    returned = {row[1].id for row in rows}
    assert returned == {claim.id for claim in claims} | {row.id for row in observations}

    kinds = {row[1].kind for row in rows}
    assert ItemKind.OBSERVATION in kinds, "observations never reached M4"
    assert ItemKind.REPORTED_CLAIM in kinds


async def test_an_observation_carries_its_subject_not_a_claimant(
    session_factory: SessionFactory,
) -> None:
    """The epistemic invariant, checked on the object M4 actually reasons over.

    An observation arriving with `claimed_by` set would tell the classifier its
    subject asserted it. That is the collapse this whole change exists to avoid.
    """
    entity_id, _claims, _observations = await _entity_with_claims_and_observations(session_factory)

    async with session_factory() as session:
        service = _service(session, FakeExtractionProvider())
        items, _total = await service._load_relatable_items(entity_id)

    for item in items:
        if item.kind is ItemKind.OBSERVATION:
            assert item.subject == "Zilan"
            assert item.claimed_by is None
            assert item.stance is None
        else:
            assert item.subject is None
            assert item.claimed_by is not None


async def test_a_relation_built_on_an_observation_is_served_and_then_reused(
    session_factory: SessionFactory,
) -> None:
    """The silent failure, made loud.

    If the reuse/serve join is narrower than the loading query, relations get
    classified and persisted and then never appear and never reuse: the operator
    sees nothing while every run pays the model again. Neither symptom shows up
    as a test failure anywhere else, so both halves are asserted here.
    """
    entity_id, _claims, _observations = await _entity_with_claims_and_observations(session_factory)

    first = FakeExtractionProvider(raw_output=_all_qualify(6))
    async with session_factory() as session:
        response = await _service(session, first).analyze(entity_id)

    # 2 claims from evidence A x 3 observations from evidence C. Same-evidence
    # pairs are filtered, so every pair here crosses the said/observed line.
    assert first.call_count == 1
    assert response.telemetry.classified_pairs == 6

    # Served: at least one surfaced relation must actually stand on an observation.
    async with session_factory() as session:
        view = await _service(session, FakeExtractionProvider()).get_intelligence(entity_id)

    sides = [side for derived in view.derived_relations for side in (derived.left, derived.right)]
    assert any(side.item_kind is ItemKind.OBSERVATION for side in sides), (
        "an observation-derived relation was persisted but never served"
    )
    for side in sides:
        if side.item_kind is ItemKind.OBSERVATION:
            assert side.subject == "Zilan"
            assert side.claimed_by is None

    # Reused: a second identical analysis must not pay the model again.
    second = FakeExtractionProvider(raw_output=_all_qualify(6))
    async with session_factory() as session:
        again = await _service(session, second).analyze(entity_id)

    assert second.call_count == 0, "observation pairs were re-classified instead of reused"
    assert again.telemetry.reused_pairs == 6
    assert again.telemetry.classified_pairs == 0


async def test_detaching_an_about_resolution_withdraws_its_relations(
    session_factory: SessionFactory,
) -> None:
    """An M3 correction must propagate into M4 without M4 writing anything."""
    entity_id, _claims, observations = await _entity_with_claims_and_observations(session_factory)

    async with session_factory() as session:
        await _service(session, FakeExtractionProvider(raw_output=_all_qualify(6))).analyze(
            entity_id
        )

    async with session_factory() as session:
        view = await _service(session, FakeExtractionProvider()).get_intelligence(entity_id)
    before = len(view.derived_relations)
    assert before > 0

    detached = observations[0]
    async with session_factory() as session:
        history = await EntityService(session).list_resolutions(detached.id)
    active = next(row for row in history.items if row.is_active)

    async with session_factory() as session:
        await EntityService(session).supersede(
            active.id, ResolutionSupersede(note="Not actually about him.")
        )

    async with session_factory() as session:
        after_view = await _service(session, FakeExtractionProvider()).get_intelligence(entity_id)

    surviving = {
        side.extraction_item_id
        for derived in after_view.derived_relations
        for side in (derived.left, derived.right)
    }
    assert detached.id not in surviving, "a relation outlived the resolution it was derived from"
    assert len(after_view.derived_relations) < before
