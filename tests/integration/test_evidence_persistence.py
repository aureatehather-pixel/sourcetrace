"""Persistence semantics against real PostgreSQL.

These assertions depend on PostgreSQL behaviour - JSONB round-tripping,
TIMESTAMPTZ instant preservation, regex CHECK constraints, row comparisons -
none of which SQLite reproduces.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelligence_engine.core.errors import EvidenceNotFound
from intelligence_engine.evidence.fingerprint import content_fingerprint
from intelligence_engine.evidence.models import EvidenceItem
from intelligence_engine.evidence.schemas import EvidenceCreate
from intelligence_engine.evidence.service import EvidenceService

pytestmark = pytest.mark.integration

ADVICE = "Engineer told me serious projects mattered more than LeetCode."


def payload(**overrides: object) -> EvidenceCreate:
    values: dict[str, object] = {"content": ADVICE}
    values.update(overrides)
    return EvidenceCreate.model_validate(values)


async def test_create_commits_durably_and_is_visible_to_a_new_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Invariant 1 - evidence is committed, not merely staged in a session."""
    async with session_factory() as writer:
        created = await EvidenceService(writer).create(
            payload(source_type="conversation_note", source_label="DevGuild")
        )

    async with session_factory() as reader:
        fetched = await EvidenceService(reader).get(created.id)

    assert fetched.id == created.id
    assert fetched.content == ADVICE


async def test_content_is_stored_byte_for_byte(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Invariant 2 - faithfulness. No trimming, no newline rewriting."""
    raw = "  Line one\n\n\tLine two with a tab  \n"

    async with session_factory() as writer:
        created = await EvidenceService(writer).create(payload(content=raw))

    async with session_factory() as reader:
        stored = (await EvidenceService(reader).get(created.id)).content

    assert stored == raw
    assert content_fingerprint(stored) == created.content_hash


async def test_identical_content_different_provenance_creates_two_records(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The critical epistemic test.

    Two engineers independently giving the same advice is two observations.
    Deduplicating them would destroy evidence.
    """
    async with session_factory() as writer:
        service = EvidenceService(writer)
        first = await service.create(
            payload(
                source_label="Senior engineer at DevGuild",
                source_type="conversation_note",
            )
        )
        second = await service.create(
            payload(
                source_label="Hiring manager at a fintech",
                source_type="conversation_note",
            )
        )

    assert first.id != second.id
    assert first.content_hash == second.content_hash  # same text, same fingerprint

    async with session_factory() as reader:
        service = EvidenceService(reader)
        assert (await service.get(first.id)).source_label == "Senior engineer at DevGuild"
        assert (await service.get(second.id)).source_label == "Hiring manager at a fintech"
        assert len((await service.list(limit=10)).items) == 2


async def test_provenance_survives_the_round_trip(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Invariant 6."""
    captured = dt.datetime(2026, 8, 22, 15, 30, tzinfo=dt.timezone(dt.timedelta(hours=8)))

    async with session_factory() as writer:
        created = await EvidenceService(writer).create(
            payload(
                source_type="article",
                source_label="Notes on hiring signals",
                source_url="https://example.org/hiring-signals",
                captured_at=captured,
                metadata={"event": "DevGuild", "confidence": "high", "attendees": 40},
            )
        )

    async with session_factory() as reader:
        fetched = await EvidenceService(reader).get(created.id)

    assert fetched.source_type == "article"
    assert fetched.source_label == "Notes on hiring signals"
    assert fetched.source_url == "https://example.org/hiring-signals"
    assert fetched.metadata == {"event": "DevGuild", "confidence": "high", "attendees": 40}
    assert fetched.captured_at == captured  # same instant


async def test_absent_provenance_stays_null_and_is_not_fabricated(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as writer:
        created = await EvidenceService(writer).create(payload())

    async with session_factory() as reader:
        fetched = await EvidenceService(reader).get(created.id)

    assert fetched.source_type is None
    assert fetched.source_label is None
    assert fetched.source_url is None
    assert fetched.captured_at is None
    assert fetched.metadata == {}


async def test_timestamps_are_timezone_aware_and_stored_as_utc(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Invariant 7 - TIMESTAMPTZ preserves the instant, not the wall clock."""
    utc8 = dt.timezone(dt.timedelta(hours=8))
    captured = dt.datetime(2026, 8, 22, 15, 30, tzinfo=utc8)

    async with session_factory() as writer:
        created = await EvidenceService(writer).create(payload(captured_at=captured))

    async with session_factory() as reader:
        fetched = await EvidenceService(reader).get(created.id)
        raw = await reader.execute(
            text("SELECT captured_at AT TIME ZONE 'UTC' FROM evidence_items WHERE id = :id"),
            {"id": created.id},
        )

    assert fetched.captured_at is not None
    assert fetched.captured_at.tzinfo is not None
    assert fetched.created_at.tzinfo is not None
    assert raw.scalar_one() == dt.datetime(2026, 8, 22, 7, 30)  # 15:30+08:00 in UTC


async def test_jsonb_preserves_nested_structure_and_types(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    metadata = {
        "event": "DevGuild",
        "people": ["engineer", "hiring manager"],
        "nested": {"round": 2, "remote": True, "score": 4.5},
        "absent": None,
    }

    async with session_factory() as writer:
        created = await EvidenceService(writer).create(payload(metadata=metadata))

    async with session_factory() as reader:
        fetched = await EvidenceService(reader).get(created.id)
        typed = await reader.execute(
            text(
                "SELECT jsonb_typeof(metadata), metadata #>> :path "
                "FROM evidence_items WHERE id = :id"
            ),
            {"id": created.id, "path": "{nested,round}"},
        )

    assert fetched.metadata == metadata
    assert typed.one() == ("object", "2")


async def test_get_unknown_id_raises_evidence_not_found(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as reader:
        with pytest.raises(EvidenceNotFound):
            await EvidenceService(reader).get(uuid.uuid4())


async def test_database_rejects_empty_content_even_if_pydantic_is_bypassed(
    session: AsyncSession,
) -> None:
    """Invariant 8 - critical rules live in the database, not only in Pydantic."""
    session.add(
        EvidenceItem(id=uuid.uuid4(), content="", content_hash=content_fingerprint(""), meta={})
    )
    with pytest.raises(IntegrityError, match="content_not_empty"):
        await session.commit()


async def test_database_rejects_a_malformed_content_hash(session: AsyncSession) -> None:
    session.add(EvidenceItem(id=uuid.uuid4(), content=ADVICE, content_hash="NOT-A-SHA256", meta={}))
    with pytest.raises(IntegrityError, match="content_hash_is_sha256_hex"):
        await session.commit()


async def test_a_failed_write_leaves_no_partial_state(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Creation either fully persists or fails cleanly - never halfway."""
    async with session_factory() as writer:
        good = await EvidenceService(writer).create(payload(source_label="valid"))

    async with session_factory() as breaker:
        breaker.add(
            EvidenceItem(id=uuid.uuid4(), content="", content_hash=content_fingerprint(""), meta={})
        )
        with pytest.raises(IntegrityError):
            await breaker.commit()
        await breaker.rollback()

    async with session_factory() as reader:
        page = await EvidenceService(reader).list(limit=10)

    assert [item.id for item in page.items] == [good.id]


async def test_ids_are_stable_across_reads(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Invariant 3."""
    async with session_factory() as writer:
        created = await EvidenceService(writer).create(payload())

    for _ in range(3):
        async with session_factory() as reader:
            assert (await EvidenceService(reader).get(created.id)).id == created.id


async def test_no_update_operation_is_exposed_by_the_repository_or_service() -> None:
    """Invariant 4 - there is no application path that rewrites raw evidence."""
    from intelligence_engine.evidence.repository import EvidenceRepository

    forbidden = {"update", "patch", "edit", "set_content", "delete", "remove"}
    assert forbidden.isdisjoint(dir(EvidenceRepository))
    assert forbidden.isdisjoint(dir(EvidenceService))
