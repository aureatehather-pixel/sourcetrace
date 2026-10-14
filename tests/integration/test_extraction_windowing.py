"""Windowed extraction, end to end, and the invariants it must not break.

One evidence item, several extraction runs, still exactly one logical source.
The tests that matter most here are the last two: that two windows of one
document never become M4 cross-evidence candidates, and that the packet still
counts that document as one source. Those are the failures that splitting the
export into several `evidence_items` would have caused, and they are the reason
windowing lives on the run rather than on the evidence.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelligence_engine.entities.schemas import ResolutionRole
from intelligence_engine.evidence.schemas import EvidenceCreate
from intelligence_engine.evidence.service import EvidenceService
from intelligence_engine.extraction.errors import (
    ExtractionError,
    IrreducibleOverflow,
    ProviderUnavailable,
)
from intelligence_engine.extraction.fake_provider import PROVIDER_NAME
from intelligence_engine.extraction.provider import ProviderIdentity, ProviderResult
from intelligence_engine.extraction.schemas import (
    MAX_ITEMS_PER_RUN,
    MAX_QUOTE_LENGTH,
    ExtractionResponse,
    RunStatus,
)
from intelligence_engine.extraction.service import MAX_SUBDIVISION_DEPTH, ExtractionService
from intelligence_engine.extraction.windowing import plan_windows
from intelligence_engine.intelligence.candidates import RelatableItem, generate_candidates
from intelligence_engine.packet.repository import PacketRepository
from intelligence_engine.packet.service import PacketService

pytestmark = pytest.mark.integration

SessionFactory = async_sessionmaker[AsyncSession]

#: Six clearly-delimited sections. At a 120-character budget this divides into
#: several windows; at the shipped default it is a single window, so the same
#: fixture exercises both paths.
SECTIONS = [
    "## One\n\nAda said the parser was the bottleneck.",
    "## Two\n\nBrian said the parser was fine and the disk was slow.",
    "## Three\n\nCleo shipped a fix on Tuesday.",
    "## Four\n\nDara measured a forty percent improvement.",
    "## Five\n\nEli said measurement without a baseline proves nothing.",
    "## Six\n\nFinn rewrote the baseline harness afterwards.",
]
WINDOWED_CONTENT = "\n\n".join(SECTIONS)

#: A budget that forces WINDOWED_CONTENT into more than one window.
SMALL_BUDGET = 120


class PerWindowProvider:
    """Returns a different payload per call, grounded in the window it is given.

    The real provider sees one window at a time and can only quote what it was
    shown. This fake behaves the same way: it quotes a fragment of the prompt it
    actually received, so a grounding check against the wrong slice fails here
    exactly as it would in production.
    """

    def __init__(self, *, quotes_per_call: int = 1) -> None:
        self.calls: list[str] = []
        self._quotes_per_call = quotes_per_call

    @property
    def identity(self) -> ProviderIdentity:
        """Fixed, and it must match what `extract` reports.

        Reuse keys on this, so a double whose identity disagreed with the rows
        it produces would never hit its own cache — and every reuse assertion in
        this file would pass for the wrong reason.
        """
        return ProviderIdentity(provider=PROVIDER_NAME, model="fake-window")

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def extract(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> ProviderResult:
        self.calls.append(user_prompt)
        # The window text is the tail of the user prompt; quote a real sentence
        # out of it so grounding passes honestly rather than by construction.
        sentences = [
            line
            for line in user_prompt.splitlines()
            if line.startswith(("Ada", "Brian", "Cleo", "Dara", "Eli", "Finn"))
        ]
        chosen = sentences[: self._quotes_per_call] or ["said"]
        items = [
            {
                "kind": "reported_claim",
                "data": {"claim": f"claim {index}", "claimed_by": "someone", "stance": "asserted"},
                "grounding": {"quote": sentence},
            }
            for index, sentence in enumerate(chosen)
        ]
        import json

        return ProviderResult(
            raw_output=json.dumps({"items": items}),
            provider=PROVIDER_NAME,
            model="fake-window",
            input_tokens=10,
            output_tokens=5,
        )


class FailOnWindowProvider(PerWindowProvider):
    """Succeeds until a chosen window, then fails — a partial source."""

    def __init__(self, *, fail_at_call: int) -> None:
        super().__init__()
        self._fail_at_call = fail_at_call

    async def extract(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> ProviderResult:
        if len(self.calls) == self._fail_at_call:
            self.calls.append(user_prompt)
            raise ProviderUnavailable("provider fell over on this window")
        return await super().extract(
            system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema
        )


class OverLimitProvider(PerWindowProvider):
    """Returns more items than `MAX_ITEMS_PER_RUN` allows, for every interval.

    Deliberately unconditional, which makes it the pathological case: no amount
    of subdivision reduces the count, so the descent must terminate on its own
    guards rather than by the provider eventually behaving.

    The quote is taken from whatever text arrived, however small, because
    subdivision keeps shrinking the interval. Picking a quote that only exists
    in the full-size window would make the fixture fail with a grounding error
    partway down the recursion and test the wrong thing.
    """

    async def extract(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> ProviderResult:
        self.calls.append(user_prompt)
        import json

        from intelligence_engine.extraction.fake_provider import extract_evidence_block

        evidence_text = extract_evidence_block(user_prompt)
        sentence = next(
            (
                line
                for line in evidence_text.splitlines()
                if line.startswith(("Ada", "Brian", "Cleo", "Dara", "Eli", "Finn"))
            ),
            evidence_text.strip()[:MAX_QUOTE_LENGTH] or "placeholder",
        )
        items = [
            {
                "kind": "reported_claim",
                "data": {"claim": f"claim {index}", "claimed_by": "x", "stance": "asserted"},
                "grounding": {"quote": sentence},
            }
            for index in range(MAX_ITEMS_PER_RUN + 1)
        ]
        return ProviderResult(
            raw_output=json.dumps({"items": items}),
            provider=PROVIDER_NAME,
            model="fake-window",
            input_tokens=10,
            output_tokens=5,
        )


async def _create_evidence(factory: SessionFactory, *, content: str) -> uuid.UUID:
    async with factory() as session:
        created = await EvidenceService(session).create(
            EvidenceCreate.model_validate(
                {
                    "content": content,
                    "source_type": "conversation_export",
                    "source_label": "windowed source",
                    "captured_at": "2026-08-27T09:00:00+08:00",
                }
            )
        )
    return created.id


async def _extract(
    factory: SessionFactory,
    evidence_id: uuid.UUID,
    provider: Any,
    *,
    budget: int = SMALL_BUDGET,
) -> ExtractionResponse:
    async with factory() as session:
        service = ExtractionService(session, provider, window_max_chars=budget)
        return await service.extract(evidence_id)


# --- the small-document path is unchanged ----------------------------------


async def test_content_within_budget_still_produces_exactly_one_run(
    session_factory: SessionFactory,
) -> None:
    """Windowing must be invisible to evidence that never needed it."""
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    provider = PerWindowProvider()

    result = await _extract(session_factory, evidence_id, provider, budget=100_000)

    assert provider.call_count == 1
    assert len(result.runs) == 1
    assert result.complete is True
    run = result.runs[0]
    assert (run.window_start, run.window_end) == (0, len(WINDOWED_CONTENT))


async def test_a_single_window_prompt_says_nothing_about_windows(
    session_factory: SessionFactory,
) -> None:
    """No 'part N of M' line when there is one part.

    This is what lets an unwindowed document produce a byte-identical prompt to
    the one it produced before this feature existed, so no stored run is
    invalidated and no extractor version bump is owed.
    """
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    provider = PerWindowProvider()

    await _extract(session_factory, evidence_id, provider, budget=100_000)

    assert "excerpt: part" not in provider.calls[0]


async def test_a_windowed_prompt_says_it_is_reading_only_a_part(
    session_factory: SessionFactory,
) -> None:
    """A model reading one sixth of a conversation is told it is reading a part.

    Deliberately without an ordinal or a total. Under adaptive extraction the
    number of intervals changes whenever one of them overflows, so "part 1 of 6"
    would describe the same characters differently between runs while reuse —
    keyed on the hash of those characters — treated the two as interchangeable.
    Saying only that this is an excerpt removes the moving part.
    """
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    provider = PerWindowProvider()

    result = await _extract(session_factory, evidence_id, provider)

    assert len(result.runs) > 1
    assert "an excerpt of this source, not the whole of it" in provider.calls[0]
    # The ordinal is gone, in every call, not merely absent from the first.
    assert not any("part 1 of" in call for call in provider.calls)


# --- several runs, one source ----------------------------------------------


async def test_one_source_produces_several_runs_all_pointing_at_it(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    expected = plan_windows(WINDOWED_CONTENT, max_chars=SMALL_BUDGET)

    result = await _extract(session_factory, evidence_id, PerWindowProvider())

    assert len(result.runs) == len(expected)
    assert result.complete is True
    # The invariant this whole design exists for.
    assert {run.evidence_id for run in result.runs} == {evidence_id}
    # Identity is the interval, and the runs tile the source in source order.
    assert [(run.window_start, run.window_end) for run in result.runs] == [
        (window.start, window.end) for window in expected
    ]


async def test_each_run_records_the_exact_slice_it_read(
    session_factory: SessionFactory,
) -> None:
    """`content[window_start:window_end]` must hash to `input_content_hash`.

    This equality is the mechanical proof that per-window grounding implies
    source-level grounding: the window really is a substring of the source.
    """
    from intelligence_engine.evidence.fingerprint import content_fingerprint

    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)

    result = await _extract(session_factory, evidence_id, PerWindowProvider())

    for run in result.runs:
        slice_text = WINDOWED_CONTENT[run.window_start : run.window_end]
        assert content_fingerprint(slice_text) == run.input_content_hash


async def test_every_grounding_quote_is_verbatim_in_the_original_source(
    session_factory: SessionFactory,
) -> None:
    """The guarantee a reader actually depends on, stated against the whole document."""
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)

    result = await _extract(session_factory, evidence_id, PerWindowProvider())

    assert result.items
    for item in result.items:
        assert item.grounding_quote in WINDOWED_CONTENT


# --- reuse and idempotency -------------------------------------------------


async def test_rerunning_reuses_every_window_and_costs_nothing(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    first_provider = PerWindowProvider()
    first = await _extract(session_factory, evidence_id, first_provider)

    second_provider = PerWindowProvider()
    second = await _extract(session_factory, evidence_id, second_provider)

    assert second_provider.call_count == 0, "a reused window must not call the provider"
    assert second.reused is True
    assert [run.id for run in second.runs] == [run.id for run in first.runs]
    assert first_provider.call_count == len(first.runs)


async def test_a_failed_window_is_retried_while_successful_ones_are_reused(
    session_factory: SessionFactory,
) -> None:
    """The requirement that makes a partial failure cheap to recover from."""
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    plan = plan_windows(WINDOWED_CONTENT, max_chars=SMALL_BUDGET)
    assert len(plan) >= 3

    failing = FailOnWindowProvider(fail_at_call=1)
    with pytest.raises(ExtractionError):
        await _extract(session_factory, evidence_id, failing)

    recovering = PerWindowProvider()
    result = await _extract(session_factory, evidence_id, recovering)

    # Window 0 succeeded before and is reused; everything from 1 on is fresh.
    assert recovering.call_count == len(plan) - 1
    assert result.complete is True
    assert result.reused is False, "money was spent, so this is not a pure cache hit"


async def test_changing_the_source_invalidates_the_stored_windows(
    session_factory: SessionFactory,
) -> None:
    """Different content is a different document, and no window carries over."""
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    await _extract(session_factory, evidence_id, PerWindowProvider())

    other_id = await _create_evidence(
        session_factory, content=WINDOWED_CONTENT.replace("Ada", "Ivy")
    )
    provider = PerWindowProvider()
    await _extract(session_factory, other_id, provider)

    assert provider.call_count > 0, "a changed source must be re-extracted"


async def test_a_bumped_extractor_version_reprocesses_every_window(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    first = await _extract(session_factory, evidence_id, PerWindowProvider())

    provider = PerWindowProvider()
    async with session_factory() as session:
        service = ExtractionService(
            session,
            provider,
            extractor_version="99.0.0",
            window_max_chars=SMALL_BUDGET,
        )
        second = await service.extract(evidence_id)

    assert provider.call_count == len(first.runs)
    assert {run.id for run in second.runs}.isdisjoint({run.id for run in first.runs})


# --- the guards stay per window --------------------------------------------


async def test_an_interval_that_always_overflows_fails_loudly_rather_than_forever(
    session_factory: SessionFactory,
) -> None:
    """The cap is not relaxed, and exceeding it is not answered by giving up.

    A provider that returns over the limit *no matter how small the interval*
    is the pathological case: subdivision cannot help, because the density is
    not real. The run must therefore terminate — bounded by depth and by the
    minimum interval width — and terminate as `IrreducibleOverflow`, naming the
    exact character range it could not read.

    What must never happen: truncating the item list to fit, or looping.
    """
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)

    with pytest.raises(IrreducibleOverflow) as caught:
        await _extract(session_factory, evidence_id, OverLimitProvider())

    assert caught.value.error_code == "irreducible_overflow"
    assert caught.value.end > caught.value.start
    assert "unread" in str(caught.value)


async def test_nothing_is_stored_for_an_interval_that_broke_the_limit(
    session_factory: SessionFactory,
) -> None:
    """Whole-run rejection, not partial acceptance of a response known to be wrong."""
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)

    with pytest.raises(IrreducibleOverflow):
        await _extract(session_factory, evidence_id, OverLimitProvider())

    async with session_factory() as session:
        service = ExtractionService(session, PerWindowProvider(), window_max_chars=SMALL_BUDGET)
        runs = await service.list_runs(evidence_id)
    failed = [run for run in runs.items if run.status is RunStatus.FAILED]
    assert failed and all(run.items == [] for run in failed)
    # Every failed attempt is recorded as an overflow, so the history says *why*
    # the source is unread rather than only that something went wrong.
    assert all(run.error_code == "extraction_overflow" for run in failed)


async def test_a_pathological_interval_costs_a_bounded_number_of_calls(
    session_factory: SessionFactory,
) -> None:
    """Termination is not enough; the descent must also be affordable.

    Each level at least halves, and `MAX_SUBDIVISION_DEPTH` caps the levels, so
    one unreadable region can cost at most a bounded number of provider calls
    before the run stops. Without the depth bound this would be unbounded even
    though it would eventually terminate on interval width.
    """
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    provider = OverLimitProvider()

    with pytest.raises(IrreducibleOverflow):
        await _extract(session_factory, evidence_id, provider)

    assert provider.call_count <= 2**MAX_SUBDIVISION_DEPTH


# --- source-level completeness ---------------------------------------------


async def test_a_failed_window_leaves_the_source_incomplete(
    session_factory: SessionFactory,
) -> None:
    """PIE must never present a half-read document as a whole one."""
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)

    with pytest.raises(ExtractionError):
        await _extract(session_factory, evidence_id, FailOnWindowProvider(fail_at_call=1))

    async with session_factory() as session:
        packet = await PacketService(session).assemble(limit=200)

    source = next(s for s in packet.sources if s.evidence_id == evidence_id)
    assert source.complete is False
    assert source.covered_chars < source.source_length
    # The unread region is named exactly, not merely counted.
    assert source.uncovered
    assert sum(end - start for start, end in source.uncovered) == (
        source.source_length - source.covered_chars
    )


async def test_a_fully_extracted_windowed_source_reports_complete(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    await _extract(session_factory, evidence_id, PerWindowProvider())

    async with session_factory() as session:
        packet = await PacketService(session).assemble(limit=200)

    source = next(s for s in packet.sources if s.evidence_id == evidence_id)
    assert source.complete is True
    assert source.uncovered == ()
    assert source.covered_chars == source.source_length == len(WINDOWED_CONTENT)


# --- the two downstream invariants that justify the design -----------------


async def test_the_packet_counts_a_windowed_document_as_one_source(
    session_factory: SessionFactory,
) -> None:
    """Six windows are one source. Rendering them as six would be a lie.

    A reader — or a reasoning model — counting sources must not be told that one
    conversation is several independent documents that happen to agree.
    """
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    result = await _extract(session_factory, evidence_id, PerWindowProvider())
    assert len(result.runs) > 1

    async with session_factory() as session:
        packet = await PacketService(session).assemble(limit=200)

    matching = [s for s in packet.sources if s.evidence_id == evidence_id]
    assert len(matching) == 1, "one evidence item, one packet source, whatever the window count"
    assert matching[0].item_count == len(result.items)


async def test_the_packet_shows_items_from_every_window(
    session_factory: SessionFactory,
) -> None:
    """The run-selection fix: partitioning by evidence alone would keep one window."""
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    result = await _extract(session_factory, evidence_id, PerWindowProvider())

    async with session_factory() as session:
        packet = await PacketService(session).assemble(limit=200)

    packet_runs = {item.extraction_run_id for item in packet.items}
    assert packet_runs == {run.id for run in result.runs}


async def test_two_windows_of_one_source_are_not_cross_evidence_candidates(
    session_factory: SessionFactory,
) -> None:
    """**The regression guard this design exists to make possible.**

    Splitting the export into several evidence items would have made every
    intra-conversation pair eligible for M4, letting two paraphrases of one
    remark be labelled `reinforces` — manufactured corroboration from a single
    utterance. Windows share one `evidence_id`, so M4's one hard filter still
    suppresses them, and this test fails the moment that stops being true.
    """
    evidence_id = await _create_evidence(session_factory, content=WINDOWED_CONTENT)
    result = await _extract(session_factory, evidence_id, PerWindowProvider())
    assert len({run.id for run in result.runs}) > 1, "the fixture must actually window"

    async with session_factory() as session:
        rows = await PacketRepository(session).list_items(limit=200)

    # The exact shape M4 consumes, built from what windowing actually stored:
    # several items, several extraction runs, one evidence id.
    items = [
        RelatableItem(
            item_id=item.id,
            kind=item.kind,
            role=ResolutionRole.CLAIMED_BY,
            text=item.grounding_quote,
            stance="asserted",
            claimed_by="someone",
            subject=None,
            grounding_quote=item.grounding_quote,
            extraction_run_id=run.id,
            evidence_id=evidence.id,
            resolution_id=uuid.uuid4(),
            source_label=evidence.source_label,
            captured_at=evidence.captured_at,
            evidence_created_at=evidence.created_at,
            position=item.position,
        )
        for item, run, evidence in rows
        if evidence.id == evidence_id
    ]
    assert len(items) > 1, "several items across several windows"
    assert len({item.extraction_run_id for item in items}) > 1, "drawn from different runs"

    selection = generate_candidates(items, max_pairs=100)

    assert selection.theoretical_pairs > 0, "pairs existed to be filtered"
    assert selection.pairs == [], "two windows of one source must never pair"
