"""Reuse must not serve one model's output as another model's answer.

The defect these tests exist for was found by inspection and is easy to state:
both derived layers decided whether to pay for a model call by looking up stored
output, and neither lookup consulted the provider or model that produced it. So
the sequence

    extract with model A  ->  switch to model B  ->  extract again

made exactly one call, to A, and reported the second run as reused — while every
row said the work was current. A comparison between two providers built on that
measures the first provider twice.

Every test below is an equality-of-mechanism test rather than a claim about any
vendor. The doubles differ only in their `identity`, so nothing here depends on
a network, a key, or how any real model behaves.

The three cases the corrections owed, for each of M2 and M4:

- identical provider and model  -> reuse, no call;
- changed model, same provider  -> no reuse, a call;
- changed provider, same model  -> no reuse, a call.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelligence_engine.evidence.schemas import EvidenceCreate
from intelligence_engine.evidence.service import EvidenceService
from intelligence_engine.extraction.fake_provider import FakeExtractionProvider
from intelligence_engine.extraction.models import ExtractionRun
from intelligence_engine.extraction.provider import ProviderIdentity
from intelligence_engine.extraction.schemas import ExtractionResponse
from intelligence_engine.extraction.service import ExtractionService
from intelligence_engine.intelligence.models import IntelligenceRelation
from intelligence_engine.intelligence.service import IntelligenceService
from tests.entity_fixtures import (
    EVIDENCE_A_CONTENT,
    EVIDENCE_B_CONTENT,
    evidence_a_payload,
    evidence_b_payload,
)
from tests.extraction_fixtures import EVIDENCE_CONTENT, valid_payload
from tests.intelligence_fixtures import classifier_response, qualifies

SessionFactory = async_sessionmaker[AsyncSession]

pytestmark = pytest.mark.integration


class RelabelledProvider(FakeExtractionProvider):
    """A fake whose vendor name differs, and nothing else does.

    Changing the *provider* half of the identity needs a double that is
    byte-identical in behaviour, or a test could pass because the two doubles
    returned different output rather than because the identity was consulted.
    """

    def __init__(self, *, provider: str, model: str, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._relabelled = ProviderIdentity(provider=provider, model=model)

    @property
    def identity(self) -> ProviderIdentity:
        return self._relabelled


# --- M2 -------------------------------------------------------------------


async def _evidence(factory: SessionFactory) -> uuid.UUID:
    async with factory() as session:
        created = await EvidenceService(session).create(
            EvidenceCreate.model_validate(
                {
                    "content": EVIDENCE_CONTENT,
                    "source_type": "conversation_note",
                    "source_label": "DevGuild",
                }
            )
        )
    return created.id


async def _extract(
    factory: SessionFactory, evidence_id: uuid.UUID, provider: Any
) -> ExtractionResponse:
    async with factory() as session:
        return await ExtractionService(session, provider).extract(evidence_id)


def _distinct(values: Any) -> list[str]:
    """The distinct non-null values of a nullable column, sorted.

    `extraction_runs.provider` and `.model` stay nullable — a run that failed
    before a generator was recorded is real history — so the null has to be
    excluded explicitly rather than sorted alongside strings.
    """
    return sorted({value for value in values if value is not None})


async def test_the_same_model_reuses_a_stored_extraction(
    session_factory: SessionFactory,
) -> None:
    """The control. Without it, the two tests below could pass by never reusing."""
    evidence_id = await _evidence(session_factory)

    first = FakeExtractionProvider(payload=valid_payload(), model="MODEL-A")
    await _extract(session_factory, evidence_id, first)

    second = FakeExtractionProvider(payload=valid_payload(), model="MODEL-A")
    response = await _extract(session_factory, evidence_id, second)

    assert first.call_count == 1
    assert second.call_count == 0
    assert response.reused is True


async def test_a_different_model_does_not_reuse_a_stored_extraction(
    session_factory: SessionFactory,
) -> None:
    """The defect, in the shape that was about to invalidate a provider bake-off."""
    evidence_id = await _evidence(session_factory)

    first = FakeExtractionProvider(payload=valid_payload(), model="MODEL-A")
    await _extract(session_factory, evidence_id, first)

    second = FakeExtractionProvider(payload=valid_payload(), model="MODEL-B")
    response = await _extract(session_factory, evidence_id, second)

    assert second.call_count == 1
    assert response.reused is False

    async with session_factory() as session:
        models = (
            await session.execute(
                select(ExtractionRun.model).where(ExtractionRun.evidence_id == evidence_id)
            )
        ).scalars()
        assert _distinct(models) == ["MODEL-A", "MODEL-B"]


async def test_a_different_provider_does_not_reuse_a_stored_extraction(
    session_factory: SessionFactory,
) -> None:
    """The model string alone is not an identity. Two vendors may use the same name."""
    evidence_id = await _evidence(session_factory)

    first = RelabelledProvider(provider="vendor-one", model="SHARED-NAME", payload=valid_payload())
    await _extract(session_factory, evidence_id, first)

    second = RelabelledProvider(provider="vendor-two", model="SHARED-NAME", payload=valid_payload())
    response = await _extract(session_factory, evidence_id, second)

    assert second.call_count == 1
    assert response.reused is False

    async with session_factory() as session:
        providers = (
            await session.execute(
                select(ExtractionRun.provider).where(ExtractionRun.evidence_id == evidence_id)
            )
        ).scalars()
        assert _distinct(providers) == ["vendor-one", "vendor-two"]


async def test_the_requested_identity_is_recorded_before_the_call(
    session_factory: SessionFactory,
) -> None:
    """A failed attempt says which generator failed.

    It used to say nothing: `provider` and `model` were written on success from
    the response, so the two columns were null in exactly the case where the
    question "which model was this?" is most worth answering. The lookup needs
    them written at the attempt anyway; this is the second reason.
    """
    from intelligence_engine.extraction.errors import ProviderTimeout
    from intelligence_engine.extraction.fake_provider import ScriptedFailureProvider

    evidence_id = await _evidence(session_factory)
    provider = ScriptedFailureProvider(ProviderTimeout("timed out"), model="MODEL-A")

    with pytest.raises(ProviderTimeout):
        await _extract(session_factory, evidence_id, provider)

    async with session_factory() as session:
        run = (
            await session.execute(
                select(ExtractionRun).where(ExtractionRun.evidence_id == evidence_id)
            )
        ).scalar_one()
        assert run.status.value == "failed"
        assert run.provider == "fake"
        assert run.model == "MODEL-A"


# --- M4 -------------------------------------------------------------------


async def _connected_entity(factory: SessionFactory) -> uuid.UUID:
    """Two notes about one person, resolved through M3. Two cross-evidence pairs."""
    from intelligence_engine.entities.schemas import (
        EntityCreate,
        EntityType,
        ResolutionCreate,
        ResolutionRole,
    )
    from intelligence_engine.entities.service import EntityService
    from intelligence_engine.extraction.models import ExtractionItem
    from intelligence_engine.extraction.schemas import ItemKind

    items: list[ExtractionItem] = []
    for content, label, payload in (
        (EVIDENCE_A_CONTENT, "Kestrel Python meetup", evidence_a_payload()),
        (EVIDENCE_B_CONTENT, "Coffee, Brookside", evidence_b_payload()),
    ):
        async with factory() as session:
            evidence = await EvidenceService(session).create(
                EvidenceCreate.model_validate(
                    {"content": content, "source_type": "conversation_note", "source_label": label}
                )
            )
        async with factory() as session:
            result = await ExtractionService(
                session, FakeExtractionProvider(payload=payload)
            ).extract(evidence.id)
        async with factory() as session:
            rows = await session.execute(
                select(ExtractionItem)
                .where(ExtractionItem.extraction_run_id == result.runs[0].id)
                .order_by(ExtractionItem.position)
            )
            items.extend(rows.scalars().all())

    async with factory() as session:
        entity = await EntityService(session).create_entity(
            EntityCreate(entity_type=EntityType.PERSON, canonical_name="Zilan")
        )

    for item in items:
        if item.kind is not ItemKind.REPORTED_CLAIM:
            continue
        async with factory() as session:
            await EntityService(session).resolve(
                item.id, ResolutionCreate(role=ResolutionRole.CLAIMED_BY)
            )
    return entity.id


def _two_qualifications() -> str:
    return classifier_response(qualifies("pair_01"), qualifies("pair_02"))


async def test_the_same_model_reuses_a_settled_relation(
    session_factory: SessionFactory,
) -> None:
    entity_id = await _connected_entity(session_factory)

    first = FakeExtractionProvider(raw_output=_two_qualifications(), model="MODEL-A")
    async with session_factory() as session:
        await IntelligenceService(session, first).analyze(entity_id)

    second = FakeExtractionProvider(raw_output=_two_qualifications(), model="MODEL-A")
    async with session_factory() as session:
        response = await IntelligenceService(session, second).analyze(entity_id)

    assert second.call_count == 0
    assert response.performed_calls is False
    assert response.telemetry.reused_pairs == 2


async def test_a_different_model_reclassifies_rather_than_reusing(
    session_factory: SessionFactory,
) -> None:
    """Two settled answers for one pair, one per model. That is what a bake-off is.

    This is also where the widened unique constraint earns its migration. With
    only the lookup corrected, this call would miss the lookup, insert, and hit a
    constraint still saying one answer per pair per classifier version — so
    "ask the other model" would have raised an integrity error.
    """
    entity_id = await _connected_entity(session_factory)

    first = FakeExtractionProvider(raw_output=_two_qualifications(), model="MODEL-A")
    async with session_factory() as session:
        await IntelligenceService(session, first).analyze(entity_id)

    second = FakeExtractionProvider(raw_output=_two_qualifications(), model="MODEL-B")
    async with session_factory() as session:
        response = await IntelligenceService(session, second).analyze(entity_id)

    assert second.call_count == 1
    assert response.performed_calls is True
    assert response.telemetry.reused_pairs == 0
    assert response.telemetry.classified_pairs == 2

    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(IntelligenceRelation)) == 4
        models = (await session.execute(select(IntelligenceRelation.model))).scalars()
        assert sorted(set(models)) == ["MODEL-A", "MODEL-B"]


async def test_a_different_provider_reclassifies_rather_than_reusing(
    session_factory: SessionFactory,
) -> None:
    entity_id = await _connected_entity(session_factory)

    first = RelabelledProvider(
        provider="vendor-one", model="SHARED-NAME", raw_output=_two_qualifications()
    )
    async with session_factory() as session:
        await IntelligenceService(session, first).analyze(entity_id)

    second = RelabelledProvider(
        provider="vendor-two", model="SHARED-NAME", raw_output=_two_qualifications()
    )
    async with session_factory() as session:
        response = await IntelligenceService(session, second).analyze(entity_id)

    assert second.call_count == 1
    assert response.telemetry.reused_pairs == 0

    async with session_factory() as session:
        providers = (await session.execute(select(IntelligenceRelation.provider))).scalars()
        assert sorted(set(providers)) == ["vendor-one", "vendor-two"]


async def test_a_rejected_relation_stays_rejected_only_for_the_model_that_produced_it(
    session_factory: SessionFactory,
) -> None:
    """The reuse widening and the review mechanism, checked against each other.

    A human rejecting model A's answer settles the question *for model A*: A is
    never asked again. It says nothing about model B, which has not answered yet,
    so B is asked. Had the identity been ignored, B would have been served A's
    rejected answer and the rejection would have silently become a permanent
    verdict on a model that never spoke.
    """
    from intelligence_engine.intelligence.schemas import RelationReject

    entity_id = await _connected_entity(session_factory)

    first = FakeExtractionProvider(raw_output=_two_qualifications(), model="MODEL-A")
    async with session_factory() as session:
        created = await IntelligenceService(session, first).analyze(entity_id)

    async with session_factory() as session:
        await IntelligenceService(session, first).reject_relation(
            created.relations[0].id, RelationReject(note="wrong on the evidence")
        )

    repeat = FakeExtractionProvider(raw_output=_two_qualifications(), model="MODEL-A")
    async with session_factory() as session:
        await IntelligenceService(session, repeat).analyze(entity_id)
    assert repeat.call_count == 0

    other = FakeExtractionProvider(raw_output=_two_qualifications(), model="MODEL-B")
    async with session_factory() as session:
        await IntelligenceService(session, other).analyze(entity_id)
    assert other.call_count == 1
