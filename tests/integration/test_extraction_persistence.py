"""Extraction persistence against real PostgreSQL.

Uses the deterministic fake provider throughout — no network, no credentials,
no nondeterminism — so these assertions are about persistence semantics, not
about model behaviour.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelligence_engine.core.errors import EvidenceNotFound
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.evidence.schemas import EvidenceCreate
from intelligence_engine.evidence.service import EvidenceService
from intelligence_engine.extraction.errors import (
    InvalidProviderOutput,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    UngroundedItem,
)
from intelligence_engine.extraction.fake_provider import (
    FakeExtractionProvider,
    ScriptedFailureProvider,
)
from intelligence_engine.extraction.models import ExtractionItem, ExtractionRun
from intelligence_engine.extraction.provider import ExtractionProvider
from intelligence_engine.extraction.schemas import (
    ExtractionResponse,
    ItemKind,
    RunStatus,
)
from intelligence_engine.extraction.service import ExtractionService
from tests.extraction_fixtures import (
    EVIDENCE_CONTENT,
    QUOTE_CLAIM,
    claim_item,
    payload_with_items,
    valid_payload,
)

pytestmark = pytest.mark.integration

SessionFactory = async_sessionmaker[AsyncSession]


async def _create_evidence(
    factory: SessionFactory, *, content: str = EVIDENCE_CONTENT, label: str = "DevGuild"
) -> uuid.UUID:
    async with factory() as session:
        created = await EvidenceService(session).create(
            EvidenceCreate.model_validate(
                {
                    "content": content,
                    "source_type": "conversation_note",
                    "source_label": label,
                    "captured_at": "2026-08-22T15:30:00+08:00",
                    "metadata": {"event": "DevGuild"},
                }
            )
        )
    return created.id


async def _run_extraction(
    factory: SessionFactory,
    evidence_id: uuid.UUID,
    provider: ExtractionProvider,
    **version_overrides: Any,
) -> ExtractionResponse:
    async with factory() as session:
        service = ExtractionService(session, provider, **version_overrides)
        return await service.extract(evidence_id)


# --- the critical invariant ----------------------------------------------


async def test_evidence_is_byte_for_byte_unchanged_by_extraction(
    session_factory: SessionFactory,
) -> None:
    """M2's single most important guarantee.

    Derived information must never be written back over raw evidence. Content
    and fingerprint are compared before and after a full successful run.
    """
    evidence_id = await _create_evidence(session_factory)

    async with session_factory() as session:
        before = await session.get(EvidenceItem, evidence_id)
        assert before is not None
        content_before, hash_before = before.content, before.content_hash
        updated_before = before.created_at

    await _run_extraction(
        session_factory, evidence_id, FakeExtractionProvider(payload=valid_payload())
    )

    async with session_factory() as session:
        after = await session.get(EvidenceItem, evidence_id)
        assert after is not None
        assert after.content == content_before == EVIDENCE_CONTENT
        assert after.content_hash == hash_before
        assert after.created_at == updated_before
        assert after.source_label == "DevGuild"
        assert after.meta == {"event": "DevGuild"}


async def test_evidence_row_count_and_content_survive_a_failed_run(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory)
    provider = ScriptedFailureProvider(ProviderTimeout("timed out"))

    with pytest.raises(ProviderTimeout):
        await _run_extraction(session_factory, evidence_id, provider)

    async with session_factory() as session:
        evidence = await session.get(EvidenceItem, evidence_id)
        assert evidence is not None
        assert evidence.content == EVIDENCE_CONTENT


# --- successful persistence ----------------------------------------------


async def test_successful_run_persists_run_and_all_items_atomically(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory)
    provider = FakeExtractionProvider(payload=valid_payload())

    response = await _run_extraction(session_factory, evidence_id, provider)

    assert response.reused is False
    run_id = response.runs[0].id

    async with session_factory() as session:
        run = await session.get(ExtractionRun, run_id)
        assert run is not None
        assert run.status is RunStatus.SUCCEEDED
        assert run.completed_at is not None
        assert run.error_code is None
        assert run.provider == "fake"
        assert run.model == "fake-deterministic-1"
        assert run.input_tokens == 100
        assert run.output_tokens == 50
        assert len(run.items) == 4


async def test_items_preserve_kind_payload_grounding_and_order(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory)
    provider = FakeExtractionProvider(payload=valid_payload())

    response = await _run_extraction(session_factory, evidence_id, provider)

    async with session_factory() as session:
        run = await session.get(ExtractionRun, response.runs[0].id)
        assert run is not None
        items = run.items

    assert [item.position for item in items] == [0, 1, 2, 3]
    assert [item.kind for item in items] == [
        ItemKind.REPORTED_CLAIM,
        ItemKind.REPORTED_CLAIM,
        ItemKind.ENTITY_MENTION,
        ItemKind.OBSERVATION,
    ]
    assert items[0].grounding_quote == QUOTE_CLAIM
    # Epistemics survive to storage: attribution is in the payload.
    assert items[0].payload["claimed_by"] == "An engineer at DevGuild"
    assert items[2].payload == {
        "name": "DevGuild",
        "entity_type": "organization",
        "described_as": "Where the engineer was encountered",
    }


async def test_every_grounding_quote_is_verbatim_in_the_stored_evidence(
    session_factory: SessionFactory,
) -> None:
    """Checked against the database, not against the fixture."""
    evidence_id = await _create_evidence(session_factory)
    await _run_extraction(
        session_factory, evidence_id, FakeExtractionProvider(payload=valid_payload())
    )

    async with session_factory() as session:
        evidence = await session.get(EvidenceItem, evidence_id)
        items = (await session.execute(select(ExtractionItem))).scalars().all()

    assert evidence is not None
    assert items
    for item in items:
        assert item.grounding_quote in evidence.content


async def test_jsonb_payload_is_queryable(session_factory: SessionFactory) -> None:
    evidence_id = await _create_evidence(session_factory)
    await _run_extraction(
        session_factory, evidence_id, FakeExtractionProvider(payload=valid_payload())
    )

    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT count(*) FROM extraction_items "
                "WHERE kind = 'reported_claim' AND payload ->> 'stance' = 'asserted'"
            )
        )
    assert result.scalar_one() == 2


async def test_empty_extraction_is_a_valid_success(session_factory: SessionFactory) -> None:
    """Evidence that supports nothing structured is not a failure."""
    evidence_id = await _create_evidence(session_factory)

    response = await _run_extraction(
        session_factory, evidence_id, FakeExtractionProvider(payload={"items": []})
    )

    assert response.runs[0].status is RunStatus.SUCCEEDED
    assert response.runs[0].items == []


# --- failure semantics ----------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ProviderTimeout("timed out"), "provider_timeout"),
        (ProviderUnavailable("upstream down"), "provider_unavailable"),
    ],
)
async def test_provider_failure_records_a_failed_run_with_no_items(
    session_factory: SessionFactory, error: ProviderError, expected_code: str
) -> None:
    evidence_id = await _create_evidence(session_factory)

    with pytest.raises(type(error)):
        await _run_extraction(session_factory, evidence_id, ScriptedFailureProvider(error))

    async with session_factory() as session:
        run = (await session.execute(select(ExtractionRun))).scalars().one()
        item_count = (
            await session.execute(select(func.count()).select_from(ExtractionItem))
        ).scalar_one()

    assert run.status is RunStatus.FAILED
    assert run.error_code == expected_code
    assert run.completed_at is not None
    # The requested identity, written at the attempt rather than on success. It
    # used to be null here, which meant a failed run could not say which
    # generator failed - null in exactly the case where the question matters
    # most. Reuse needs it recorded before the call anyway.
    assert run.provider == "fake"
    assert run.model == "fake-deterministic-1"
    assert item_count == 0


async def test_invalid_provider_output_fails_the_run_and_stores_nothing(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory)
    provider = FakeExtractionProvider(raw_output="I extracted some things for you!")

    with pytest.raises(InvalidProviderOutput):
        await _run_extraction(session_factory, evidence_id, provider)

    async with session_factory() as session:
        run = (await session.execute(select(ExtractionRun))).scalars().one()
        items = (await session.execute(select(ExtractionItem))).scalars().all()

    assert run.status is RunStatus.FAILED
    assert run.error_code == "invalid_provider_output"
    assert items == []


async def test_unknown_item_kind_fails_the_run(session_factory: SessionFactory) -> None:
    evidence_id = await _create_evidence(session_factory)
    provider = FakeExtractionProvider(
        payload=payload_with_items(
            {
                "kind": "psychological_profile",
                "data": {"summary": "driven"},
                "grounding": {"quote": QUOTE_CLAIM},
            }
        )
    )

    with pytest.raises(InvalidProviderOutput):
        await _run_extraction(session_factory, evidence_id, provider)

    async with session_factory() as session:
        run = (await session.execute(select(ExtractionRun))).scalars().one()
    assert run.status is RunStatus.FAILED


async def test_ungrounded_output_fails_the_run_and_stores_nothing(
    session_factory: SessionFactory,
) -> None:
    """A fabricated quote rejects the whole run, including its valid items."""
    evidence_id = await _create_evidence(session_factory)
    provider = FakeExtractionProvider(
        payload=payload_with_items(
            claim_item(),
            claim_item(quote="He said he was the best engineer in the country"),
        )
    )

    with pytest.raises(UngroundedItem):
        await _run_extraction(session_factory, evidence_id, provider)

    async with session_factory() as session:
        run = (await session.execute(select(ExtractionRun))).scalars().one()
        items = (await session.execute(select(ExtractionItem))).scalars().all()

    assert run.status is RunStatus.FAILED
    assert run.error_code == "ungrounded_item"
    assert items == []


async def test_a_failed_run_is_never_reused(session_factory: SessionFactory) -> None:
    """Reuse serves successes only; a failure must not suppress a retry."""
    evidence_id = await _create_evidence(session_factory)

    with pytest.raises(ProviderTimeout):
        await _run_extraction(
            session_factory, evidence_id, ScriptedFailureProvider(ProviderTimeout("t"))
        )

    good = FakeExtractionProvider(payload=valid_payload())
    response = await _run_extraction(session_factory, evidence_id, good)

    assert response.reused is False
    assert good.call_count == 1


async def test_retry_attempts_coexist_as_history(session_factory: SessionFactory) -> None:
    """Timeout, then malformed output, then success: three rows, all kept."""
    evidence_id = await _create_evidence(session_factory)

    with pytest.raises(ProviderTimeout):
        await _run_extraction(
            session_factory, evidence_id, ScriptedFailureProvider(ProviderTimeout("t"))
        )
    with pytest.raises(InvalidProviderOutput):
        await _run_extraction(
            session_factory, evidence_id, FakeExtractionProvider(raw_output="{oops")
        )
    await _run_extraction(
        session_factory, evidence_id, FakeExtractionProvider(payload=valid_payload())
    )

    async with session_factory() as session:
        runs = (
            (await session.execute(select(ExtractionRun).order_by(ExtractionRun.started_at)))
            .scalars()
            .all()
        )

    assert len(runs) == 3
    assert [run.status for run in runs] == [
        RunStatus.FAILED,
        RunStatus.FAILED,
        RunStatus.SUCCEEDED,
    ]
    assert [run.error_code for run in runs[:2]] == [
        "provider_timeout",
        "invalid_provider_output",
    ]


# --- reuse ----------------------------------------------------------------


async def test_second_extraction_reuses_without_calling_the_model(
    session_factory: SessionFactory,
) -> None:
    """The cost-control guarantee: unchanged inputs must not pay twice."""
    evidence_id = await _create_evidence(session_factory)
    provider = FakeExtractionProvider(payload=valid_payload())

    first = await _run_extraction(session_factory, evidence_id, provider)
    second = await _run_extraction(session_factory, evidence_id, provider)

    assert provider.call_count == 1
    assert first.reused is False
    assert second.reused is True
    assert second.runs[0].id == first.runs[0].id
    assert len(second.runs[0].items) == 4

    async with session_factory() as session:
        run_count = (
            await session.execute(select(func.count()).select_from(ExtractionRun))
        ).scalar_one()
    assert run_count == 1  # reuse creates no new attempt


async def test_a_bumped_extractor_version_forces_reprocessing(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory)
    provider = FakeExtractionProvider(payload=valid_payload())

    await _run_extraction(session_factory, evidence_id, provider)
    # Deliberately not the current default: the point is that a *different*
    # version invalidates reuse, which a value equal to the default cannot show.
    second = await _run_extraction(
        session_factory, evidence_id, provider, extractor_version="99.0.0"
    )

    assert provider.call_count == 2
    assert second.reused is False
    assert second.runs[0].extractor_version == "99.0.0"


async def test_a_bumped_schema_version_forces_reprocessing(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory)
    provider = FakeExtractionProvider(payload=valid_payload())

    await _run_extraction(session_factory, evidence_id, provider)
    second = await _run_extraction(session_factory, evidence_id, provider, schema_version=99)

    assert provider.call_count == 2
    assert second.reused is False
    assert second.runs[0].schema_version == 99


async def test_a_different_extractor_name_forces_reprocessing(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory)
    provider = FakeExtractionProvider(payload=valid_payload())

    await _run_extraction(session_factory, evidence_id, provider)
    await _run_extraction(session_factory, evidence_id, provider, extractor_name="other_extractor")

    assert provider.call_count == 2


async def test_a_different_input_content_hash_is_not_reused(
    session_factory: SessionFactory,
) -> None:
    """The hash is part of the key, so stale output cannot be served.

    Evidence is immutable, so this is simulated by rewriting the stored run's
    hash - which is exactly the state a future re-ingestion would produce.
    """
    evidence_id = await _create_evidence(session_factory)
    provider = FakeExtractionProvider(payload=valid_payload())
    await _run_extraction(session_factory, evidence_id, provider)

    async with session_factory() as session:
        run = (await session.execute(select(ExtractionRun))).scalars().one()
        run.input_content_hash = "0" * 64
        await session.commit()

    second = await _run_extraction(session_factory, evidence_id, provider)

    assert provider.call_count == 2
    assert second.reused is False


async def test_reuse_is_scoped_to_one_evidence_item(
    session_factory: SessionFactory,
) -> None:
    """Identical text from a different source is different evidence.

    Carries the M1 epistemic rule forward: two independent observations must
    each get their own extraction, never share one.
    """
    first_id = await _create_evidence(session_factory, label="Engineer at DevGuild")
    second_id = await _create_evidence(session_factory, label="Hiring manager, fintech")
    provider = FakeExtractionProvider(payload=valid_payload())

    await _run_extraction(session_factory, first_id, provider)
    second = await _run_extraction(session_factory, second_id, provider)

    assert provider.call_count == 2
    assert second.reused is False
    assert second.runs[0].evidence_id == second_id


async def test_reuse_returns_the_newest_successful_run(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory)

    await _run_extraction(
        session_factory, evidence_id, FakeExtractionProvider(payload=valid_payload())
    )
    # Force a second success by bumping, then reverting, the version.
    await _run_extraction(
        session_factory,
        evidence_id,
        FakeExtractionProvider(payload=payload_with_items(claim_item())),
        extractor_version="9.9.9",
    )

    reused = await _run_extraction(
        session_factory, evidence_id, FakeExtractionProvider(payload=valid_payload())
    )

    assert reused.reused is True
    assert (
        len(reused.runs[0].items) == 4
    )  # the 1.0.0 run, not the 9.9.9 one # type: ignore[attr-defined]


# --- not found ------------------------------------------------------------


async def test_extracting_unknown_evidence_raises_not_found(
    session_factory: SessionFactory,
) -> None:
    provider = FakeExtractionProvider(payload=valid_payload())

    with pytest.raises(EvidenceNotFound):
        await _run_extraction(session_factory, uuid.uuid4(), provider)

    assert provider.call_count == 0  # no attempt row, no model call


# --- database-level truths ------------------------------------------------


async def test_foreign_key_blocks_deleting_evidence_that_has_extractions(
    session_factory: SessionFactory,
) -> None:
    """RESTRICT: provenance history cannot be silently orphaned."""
    evidence_id = await _create_evidence(session_factory)
    await _run_extraction(
        session_factory, evidence_id, FakeExtractionProvider(payload=valid_payload())
    )

    async with session_factory() as session:
        with pytest.raises(IntegrityError):
            await session.execute(
                text("DELETE FROM evidence_items WHERE id = :id"), {"id": evidence_id}
            )
            await session.commit()


async def test_deleting_a_run_cascades_to_its_items(
    session_factory: SessionFactory,
) -> None:
    """CASCADE: an item has no meaning apart from its run."""
    evidence_id = await _create_evidence(session_factory)
    response = await _run_extraction(
        session_factory, evidence_id, FakeExtractionProvider(payload=valid_payload())
    )

    async with session_factory() as session:
        await session.execute(
            text("DELETE FROM extraction_runs WHERE id = :id"),
            {"id": response.runs[0].id},
        )
        await session.commit()

    async with session_factory() as session:
        remaining = (
            await session.execute(select(func.count()).select_from(ExtractionItem))
        ).scalar_one()
    assert remaining == 0


async def test_foreign_key_rejects_a_run_for_nonexistent_evidence(
    session: AsyncSession,
) -> None:
    session.add(
        ExtractionRun(
            id=uuid.uuid4(),
            evidence_id=uuid.uuid4(),
            input_content_hash="a" * 64,
            extractor_name="x",
            extractor_version="1",
            schema_version=1,
            status=RunStatus.RUNNING,
            started_at=func.now(),
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_database_rejects_a_succeeded_run_without_a_completion_time(
    session_factory: SessionFactory,
) -> None:
    """A finished run must record when it finished, whatever the code does."""
    evidence_id = await _create_evidence(session_factory)

    async with session_factory() as session:
        with pytest.raises(IntegrityError, match="completed_at_matches_status"):
            await session.execute(
                text(
                    "INSERT INTO extraction_runs (id, evidence_id, input_content_hash, "
                    "extractor_name, extractor_version, schema_version, status, started_at, "
                    "window_start, window_end) "
                    "VALUES (:id, :ev, :h, 'x', '1', 1, 'succeeded', now(), 0, 1)"
                ),
                {"id": uuid.uuid4(), "ev": evidence_id, "h": "a" * 64},
            )
            await session.commit()


async def test_database_rejects_a_failed_run_without_an_error_code(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory)

    async with session_factory() as session:
        with pytest.raises(IntegrityError, match="error_code_matches_status"):
            await session.execute(
                text(
                    "INSERT INTO extraction_runs (id, evidence_id, input_content_hash, "
                    "extractor_name, extractor_version, schema_version, status, "
                    "started_at, completed_at, "
                    "window_start, window_end) "
                    "VALUES (:id, :ev, :h, 'x', '1', 1, 'failed', now(), now(), 0, 1)"
                ),
                {"id": uuid.uuid4(), "ev": evidence_id, "h": "a" * 64},
            )
            await session.commit()


async def test_database_rejects_an_unknown_run_status(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory)

    async with session_factory() as session:
        with pytest.raises(IntegrityError):
            await session.execute(
                text(
                    "INSERT INTO extraction_runs (id, evidence_id, input_content_hash, "
                    "extractor_name, extractor_version, schema_version, status, started_at, "
                    "window_start, window_end) "
                    "VALUES (:id, :ev, :h, 'x', '1', 1, 'mostly_done', now(), 0, 1)"
                ),
                {"id": uuid.uuid4(), "ev": evidence_id, "h": "a" * 64},
            )
            await session.commit()


async def test_database_rejects_duplicate_item_positions(
    session_factory: SessionFactory,
) -> None:
    """Ordering must be unambiguous; UUIDs and created_at do not order items."""
    evidence_id = await _create_evidence(session_factory)
    response = await _run_extraction(
        session_factory, evidence_id, FakeExtractionProvider(payload=valid_payload())
    )

    async with session_factory() as session:
        with pytest.raises(IntegrityError, match="unique_position_per_run"):
            await session.execute(
                text(
                    "INSERT INTO extraction_items (id, extraction_run_id, kind, payload, "
                    "grounding_quote, position) "
                    "VALUES (:id, :run, 'observation', :payload, 'some quote', 0)"
                ),
                {
                    "id": uuid.uuid4(),
                    "run": response.runs[0].id,
                    "payload": json.dumps({"observation": "duplicate position"}),
                },
            )
            await session.commit()
