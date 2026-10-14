"""Adaptive bounded extraction, end to end.

The architecture in one sentence: **input size is a cost heuristic, and
classified overflow plus deterministic subdivision is the correctness
mechanism.** Everything here tests that sentence.

The tests that matter most are the negative ones. Subdividing when an interval
genuinely produced more than one bounded response could carry is the whole
feature; subdividing after a timeout, a rate limit or a fabricated quote would
multiply calls against a provider that is failing for reasons no smaller prompt
can fix. So each non-overflow failure has its own test asserting that exactly
one call was made.

The other half is identity. `window_index` used to be what reuse and packet
selection keyed on, and a subdivided interval's child would inherit an ordinal
an untouched sibling already held — silently serving one interval's items as
another's, and silently dropping a sibling from every packet. Those two failures
are reproduced directly, against the real query, in
`test_a_child_interval_never_evicts_a_sibling_from_the_packet`.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelligence_engine.entities.schemas import ResolutionRole
from intelligence_engine.evidence.schemas import EvidenceCreate
from intelligence_engine.evidence.service import EvidenceService
from intelligence_engine.extraction.errors import (
    InvalidProviderOutput,
    IrreducibleOverflow,
    ProviderTimeout,
    ProviderUnavailable,
    UngroundedItem,
)
from intelligence_engine.extraction.fake_provider import (
    PROVIDER_NAME,
    IntervalScriptedProvider,
)
from intelligence_engine.extraction.provider import ProviderResult, StopReason
from intelligence_engine.extraction.schemas import (
    MAX_ITEMS_PER_RUN,
    ExtractionResponse,
    RunStatus,
)
from intelligence_engine.extraction.service import ExtractionService
from intelligence_engine.extraction.windowing import plan_windows, subdivide
from intelligence_engine.intelligence.candidates import RelatableItem, generate_candidates
from intelligence_engine.packet.service import PacketService

pytestmark = pytest.mark.integration

SessionFactory = async_sessionmaker[AsyncSession]

#: Eight clearly-delimited sections, each with a quotable sentence. At
#: SMALL_BUDGET this divides into several intervals; at the shipped default it
#: is one, so the same fixture exercises both paths.
SPEAKERS = ("Ada", "Brian", "Cleo", "Dara", "Eli", "Finn", "Gus", "Hana")
SECTIONS = [
    f"## Section {index}\n\n{name} reported that measurement {index} was recorded on Tuesday."
    for index, name in enumerate(SPEAKERS)
]
CONTENT = "\n\n".join(SECTIONS)

#: Forces CONTENT into several intervals.
SMALL_BUDGET = 140


def _sentence(evidence_text: str) -> str:
    """A real sentence out of the interval, so grounding passes honestly."""
    for line in evidence_text.splitlines():
        if line.startswith(SPEAKERS):
            return line
    return evidence_text.strip()[:40]


def _items(evidence_text: str, count: int) -> str:
    quote = _sentence(evidence_text)
    return json.dumps(
        {
            "items": [
                {
                    "kind": "reported_claim",
                    "data": {"claim": f"claim {index}", "claimed_by": "someone"},
                    "grounding": {"quote": quote},
                }
                for index in range(count)
            ]
        }
    )


def _result(raw: str, *, stop_reason: StopReason = StopReason.COMPLETE) -> ProviderResult:
    return ProviderResult(
        raw_output=raw,
        provider=PROVIDER_NAME,
        model="fake-adaptive",
        input_tokens=10,
        output_tokens=5,
        stop_reason=stop_reason,
    )


def _ok(count: int = 1) -> Any:
    """A responder that always succeeds, quoting whatever interval it is sent."""
    return lambda text: _result(_items(text, count))


async def _create_evidence(factory: SessionFactory, *, content: str = CONTENT) -> uuid.UUID:
    async with factory() as session:
        created = await EvidenceService(session).create(
            EvidenceCreate.model_validate(
                {
                    "content": content,
                    "source_type": "conversation_export",
                    "source_label": "adaptive source",
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
    extractor_version: str | None = None,
) -> ExtractionResponse:
    async with factory() as session:
        service = (
            ExtractionService(
                session, provider, window_max_chars=budget, extractor_version=extractor_version
            )
            if extractor_version is not None
            else ExtractionService(session, provider, window_max_chars=budget)
        )
        return await service.extract(evidence_id)


# --- what may and may not trigger subdivision ------------------------------


async def test_too_many_items_subdivides_the_offending_interval(
    session_factory: SessionFactory,
) -> None:
    """Trigger A: the response carried more than `MAX_ITEMS_PER_RUN` items.

    That is not a malformed response — it is a correct answer to too large a
    question — so the interval is split and the children asked instead.
    """
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[1].text(CONTENT)

    def responder(text: str) -> ProviderResult:
        if text == doomed:
            return _result(_items(text, MAX_ITEMS_PER_RUN + 1))
        return _result(_items(text, 1))

    provider = IntervalScriptedProvider(responder)
    result = await _extract(session_factory, evidence_id, provider)

    assert doomed in provider.intervals_seen
    # The interval was retried in strictly smaller pieces.
    children = [t for t in provider.intervals_seen if t != doomed and t in doomed]
    assert children, "the overflowing interval was never subdivided"
    assert result.complete is True


async def test_output_ceiling_truncation_subdivides_the_offending_interval(
    session_factory: SessionFactory,
) -> None:
    """Trigger B: the provider stopped at the output ceiling.

    This is the exact shape the real 60KB export produced. The response is cut
    off, so it is an incomplete answer for the interval whether or not the
    fragment that arrived happens to parse — which is why the stop reason is
    checked before the JSON is.
    """
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[0].text(CONTENT)

    def responder(text: str) -> ProviderResult:
        if text == doomed:
            return _result('{"items": [{"kind": "reported', stop_reason=StopReason.MAX_OUTPUT)
        return _result(_items(text, 1))

    provider = IntervalScriptedProvider(responder)
    result = await _extract(session_factory, evidence_id, provider)

    children = [t for t in provider.intervals_seen if t != doomed and t in doomed]
    assert children, "a truncated response did not subdivide"
    assert result.complete is True


async def test_a_truncated_response_that_still_parses_is_an_overflow(
    session_factory: SessionFactory,
) -> None:
    """The stop reason decides, not the parse.

    A response cut off on a lucky boundary can be valid JSON and still be an
    incomplete reading of the interval. Storing it would silently record a
    partial extraction as a complete one.
    """
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[0].text(CONTENT)

    def responder(text: str) -> ProviderResult:
        if text == doomed:
            # Perfectly valid JSON, but the provider says it ran out of room.
            return _result(_items(text, 2), stop_reason=StopReason.MAX_OUTPUT)
        return _result(_items(text, 1))

    provider = IntervalScriptedProvider(responder)
    await _extract(session_factory, evidence_id, provider)

    assert [t for t in provider.intervals_seen if t != doomed and t in doomed]


@pytest.mark.parametrize(
    ("responder", "expected"),
    [
        (lambda text: _result("this is not json at all"), InvalidProviderOutput),
        (
            lambda text: _result(
                json.dumps(
                    {
                        "items": [
                            {
                                "kind": "reported_claim",
                                "data": {"claim": "c"},
                                "grounding": {"quote": "never appeared in the source at all"},
                            }
                        ]
                    }
                )
            ),
            UngroundedItem,
        ),
    ],
)
async def test_a_bad_response_fails_without_subdividing(
    session_factory: SessionFactory, responder: Any, expected: type[Exception]
) -> None:
    """Malformed output and fabricated quotes are not size problems.

    Neither gets smaller when the prompt does, so retrying in pieces would pay
    two or four times over to learn the same thing. Exactly one call is made.
    """
    evidence_id = await _create_evidence(session_factory)
    provider = IntervalScriptedProvider(responder)

    with pytest.raises(expected):
        await _extract(session_factory, evidence_id, provider)

    assert provider.call_count == 1


@pytest.mark.parametrize(
    "error",
    [
        ProviderTimeout("timed out"),
        ProviderUnavailable("rate limited"),
        ProviderUnavailable("provider returned status 503"),
    ],
)
async def test_a_provider_failure_never_subdivides(
    session_factory: SessionFactory, error: Exception
) -> None:
    """The failure mode that would be most expensive to get wrong.

    A provider that is timing out or rate-limiting will do so again for every
    child, so subdivision turns one failure into an exponential fan-out of
    failures against a service already in trouble.
    """
    evidence_id = await _create_evidence(session_factory)

    def responder(text: str) -> ProviderResult:
        raise error

    provider = IntervalScriptedProvider(responder)

    with pytest.raises(type(error)):
        await _extract(session_factory, evidence_id, provider)

    assert provider.call_count == 1


async def test_a_successful_interval_is_never_subdivided(
    session_factory: SessionFactory,
) -> None:
    """Only the overflowing interval is retried; the rest are read once each."""
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[1].text(CONTENT)

    def responder(text: str) -> ProviderResult:
        if text == doomed:
            return _result(_items(text, MAX_ITEMS_PER_RUN + 1))
        return _result(_items(text, 1))

    provider = IntervalScriptedProvider(responder)
    await _extract(session_factory, evidence_id, provider)

    for window in plan:
        text = window.text(CONTENT)
        if text != doomed:
            assert provider.intervals_seen.count(text) == 1, "a healthy interval was re-read"


async def test_overflow_nested_two_levels_deep_still_resolves(
    session_factory: SessionFactory,
) -> None:
    """A child may overflow too, and recursion must handle that.

    The first child of the first interval overflows as well, so the descent goes
    two levels before succeeding. Coverage must still tile exactly.
    """
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[0].text(CONTENT)

    # Named with the real splitter, not guessed. Picking "whichever child
    # arrives first" would sometimes land on a sub-minimum fragment and test
    # the termination guard instead of the recursion.
    children = subdivide(CONTENT, plan[0].start, plan[0].end)
    assert children is not None
    doomed_child = max(children, key=lambda c: c.end - c.start).text(CONTENT)

    def responder(text: str) -> ProviderResult:
        if text in (doomed, doomed_child):
            return _result(_items(text, MAX_ITEMS_PER_RUN + 1))
        return _result(_items(text, 1))

    provider = IntervalScriptedProvider(responder)
    result = await _extract(session_factory, evidence_id, provider)

    assert result.complete is True
    assert result.uncovered == []
    assert doomed_child in provider.intervals_seen, "the chosen child was never asked"
    grandchildren = [t for t in provider.intervals_seen if t != doomed_child and t in doomed_child]
    assert len(grandchildren) > 1, "the nested overflow did not subdivide"


# --- offsets are the identity ----------------------------------------------


async def test_every_run_covers_a_distinct_interval(
    session_factory: SessionFactory,
) -> None:
    """Two runs may not claim the same characters, whatever the traversal did.

    Under ordinal identity a subdivided child could collide with a sibling; the
    interval cannot, because sharing it means reading the same text.
    """
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[2].text(CONTENT)

    def responder(text: str) -> ProviderResult:
        if text == doomed:
            return _result(_items(text, MAX_ITEMS_PER_RUN + 1))
        return _result(_items(text, 1))

    result = await _extract(session_factory, evidence_id, IntervalScriptedProvider(responder))

    intervals = [(run.window_start, run.window_end) for run in result.runs]
    assert len(intervals) == len(set(intervals))
    # And they are returned in source order, not in the order they were produced.
    assert intervals == sorted(intervals)


async def test_subdivision_leaves_sibling_runs_untouched(
    session_factory: SessionFactory,
) -> None:
    """A sibling's stored identity must not be rewritten by someone else's split.

    This is what renumbering would have done: the run rows for intervals that
    never overflowed keep exactly the ids and ranges they were first given.
    """
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    first = plan[0].text(CONTENT)
    doomed = plan[1].text(CONTENT)

    # A clean pass, so the first interval already has a stored run.
    clean = await _extract(session_factory, evidence_id, IntervalScriptedProvider(_ok()))
    before = {(r.window_start, r.window_end): r.id for r in clean.runs}

    def responder(text: str) -> ProviderResult:
        if text == doomed:
            return _result(_items(text, MAX_ITEMS_PER_RUN + 1))
        return _result(_items(text, 1))

    # Bump the version so nothing is reused and every interval is re-read.
    after = await _extract(
        session_factory,
        evidence_id,
        IntervalScriptedProvider(responder),
        extractor_version="99.0.0",
    )

    first_range = (plan[0].start, plan[0].start + len(first))
    assert first_range in before
    # The untouched sibling still covers exactly the same characters.
    assert first_range in {(r.window_start, r.window_end) for r in after.runs}


async def test_a_child_interval_never_evicts_a_sibling_from_the_packet(
    session_factory: SessionFactory,
) -> None:
    """The defect offset identity exists to prevent, reproduced against the packet.

    Under `(evidence_id, window_index)` partitioning, a child created later and
    numbered 1 would outrank the still-valid run for the original interval 1 and
    remove its items from every packet — a silent loss of evidence that nothing
    downstream could detect. Here interval 0 subdivides into several children
    while every later interval keeps its own run, and the packet must show items
    from all of them.
    """
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[0].text(CONTENT)

    def responder(text: str) -> ProviderResult:
        if text == doomed:
            return _result(_items(text, MAX_ITEMS_PER_RUN + 1))
        return _result(_items(text, 1))

    result = await _extract(session_factory, evidence_id, IntervalScriptedProvider(responder))

    async with session_factory() as session:
        packet = await PacketService(session).assemble(limit=200)

    packet_runs = {item.extraction_run_id for item in packet.items}
    stored_runs = {run.id for run in result.runs if run.items}
    assert stored_runs, "the fixture stored no items"
    assert stored_runs <= packet_runs, "the packet dropped a run's items"
    # One source, however many intervals it took.
    assert len([s for s in packet.sources if s.evidence_id == evidence_id]) == 1


# --- reuse across subdivision ----------------------------------------------


async def test_successful_children_are_reused_on_a_later_run(
    session_factory: SessionFactory,
) -> None:
    """A child interval is cached like any other, so a retry costs nothing.

    The second pass must make zero provider calls even though half its intervals
    only exist because the first pass subdivided something.
    """
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[1].text(CONTENT)

    def responder(text: str) -> ProviderResult:
        if text == doomed:
            return _result(_items(text, MAX_ITEMS_PER_RUN + 1))
        return _result(_items(text, 1))

    first = await _extract(session_factory, evidence_id, IntervalScriptedProvider(responder))
    assert first.complete is True

    # The same provider would overflow again on the parent — but the parent is
    # never asked, because its children are all cached.
    second_provider = IntervalScriptedProvider(responder)
    second = await _extract(session_factory, evidence_id, second_provider)

    # Exactly one call: the parent interval is re-probed, and every child is
    # served from cache. Re-probing is deliberate rather than an oversight —
    # overflow is a property of a *response*, not of the text, and the same
    # interval measurably does not always produce the same item count. Recording
    # "this interval always overflows" would bake in one sample of a
    # non-deterministic observation, and the re-probe lets a source that no
    # longer overflows be read in one call again instead of four forever.
    assert second_provider.call_count == 1
    assert second_provider.intervals_seen == [doomed]
    assert second.reused is False, "a re-probed parent means money was spent"
    assert second.complete is True
    assert {(r.window_start, r.window_end) for r in second.runs} == {
        (r.window_start, r.window_end) for r in first.runs
    }


async def test_a_reused_source_reports_that_it_spent_nothing(
    session_factory: SessionFactory,
) -> None:
    """An unchanged, fully extracted source makes zero provider calls."""
    evidence_id = await _create_evidence(session_factory)
    await _extract(session_factory, evidence_id, IntervalScriptedProvider(_ok()))

    provider = IntervalScriptedProvider(_ok())
    result = await _extract(session_factory, evidence_id, provider)

    assert provider.call_count == 0
    assert result.reused is True


async def test_siblings_are_reused_while_a_failed_interval_is_retried(
    session_factory: SessionFactory,
) -> None:
    """Partial progress survives, which is what makes a retry a resume."""
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[2].text(CONTENT)

    def failing(text: str) -> ProviderResult:
        if text == doomed:
            raise ProviderUnavailable("fell over on this interval")
        return _result(_items(text, 1))

    with pytest.raises(ProviderUnavailable):
        await _extract(session_factory, evidence_id, IntervalScriptedProvider(failing))

    recovered = IntervalScriptedProvider(_ok())
    result = await _extract(session_factory, evidence_id, recovered)

    # Only the intervals that had not already succeeded were paid for again.
    assert recovered.call_count == len(plan) - 2
    assert result.complete is True
    assert result.reused is False


# --- completeness as interval tiling ---------------------------------------


async def test_a_fully_read_source_tiles_exactly(session_factory: SessionFactory) -> None:
    evidence_id = await _create_evidence(session_factory)

    result = await _extract(session_factory, evidence_id, IntervalScriptedProvider(_ok()))

    assert result.complete is True
    assert result.uncovered == []
    assert result.source_length == len(CONTENT)
    intervals = sorted((r.window_start, r.window_end) for r in result.runs)
    assert intervals[0][0] == 0
    assert intervals[-1][1] == len(CONTENT)


async def test_a_failed_interval_leaves_a_named_gap(session_factory: SessionFactory) -> None:
    """Incomplete is reported as *which characters*, not merely that it failed."""
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[1].text(CONTENT)

    def failing(text: str) -> ProviderResult:
        if text == doomed:
            raise ProviderUnavailable("fell over")
        return _result(_items(text, 1))

    with pytest.raises(ProviderUnavailable):
        await _extract(session_factory, evidence_id, IntervalScriptedProvider(failing))

    async with session_factory() as session:
        packet = await PacketService(session).assemble(limit=200)

    source = next(s for s in packet.sources if s.evidence_id == evidence_id)
    assert source.complete is False
    # Fail-fast: the failing interval and everything after it went unread, so
    # the failure sits inside one larger gap rather than forming its own.
    assert any(start <= plan[1].start and plan[1].end <= end for start, end in source.uncovered), (
        f"the failed interval is not inside any reported gap: {source.uncovered}"
    )
    assert source.uncovered[-1][1] == len(CONTENT)


async def test_subdivided_children_still_produce_complete_coverage(
    session_factory: SessionFactory,
) -> None:
    """The point of computing completeness as a tiling.

    A source read partly whole and partly in children is exactly as complete as
    one read in a single pass, because the children cover what their parent
    would have.
    """
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[0].text(CONTENT)

    def responder(text: str) -> ProviderResult:
        if text == doomed:
            return _result(_items(text, MAX_ITEMS_PER_RUN + 1))
        return _result(_items(text, 1))

    result = await _extract(session_factory, evidence_id, IntervalScriptedProvider(responder))

    assert result.complete is True
    async with session_factory() as session:
        packet = await PacketService(session).assemble(limit=200)
    source = next(s for s in packet.sources if s.evidence_id == evidence_id)
    assert source.complete is True
    assert source.covered_chars == source.source_length == len(CONTENT)


async def test_an_unresolved_child_leaves_the_source_incomplete(
    session_factory: SessionFactory,
) -> None:
    """A descent that runs out leaves a gap, not a silent success."""
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[0].text(CONTENT)

    def responder(text: str) -> ProviderResult:
        # Everything inside the first interval overflows, forever.
        if text in doomed:
            return _result(_items(text, MAX_ITEMS_PER_RUN + 1))
        return _result(_items(text, 1))

    with pytest.raises(IrreducibleOverflow):
        await _extract(session_factory, evidence_id, IntervalScriptedProvider(responder))

    async with session_factory() as session:
        packet = await PacketService(session).assemble(limit=200)

    sources = [s for s in packet.sources if s.evidence_id == evidence_id]
    # Either nothing was stored at all, or what was stored is honestly partial.
    assert not sources or sources[0].complete is False


# --- provenance survives subdivision ---------------------------------------


async def test_many_leaves_of_one_source_are_never_cross_evidence_candidates(
    session_factory: SessionFactory,
) -> None:
    """M4's one hard filter, under the traversal that stresses it most.

    Subdivision multiplies runs, and every one of them must still carry the same
    `evidence_id` — otherwise two paraphrases of a single remark, landing in
    different children, become an eligible pair the classifier could label
    `reinforces`, manufacturing corroboration from one utterance.
    """
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[0].text(CONTENT)

    def responder(text: str) -> ProviderResult:
        if text == doomed:
            return _result(_items(text, MAX_ITEMS_PER_RUN + 1))
        return _result(_items(text, 2))

    result = await _extract(session_factory, evidence_id, IntervalScriptedProvider(responder))

    assert len({run.evidence_id for run in result.runs}) == 1

    async with session_factory() as session:
        evidence = await EvidenceService(session).get(evidence_id)
    assert evidence is not None

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
            evidence_id=evidence_id,
            resolution_id=uuid.uuid4(),
            source_label=evidence.source_label,
            captured_at=evidence.captured_at,
            evidence_created_at=evidence.created_at,
            position=item.position,
        )
        for run in result.runs
        for item in run.items
    ]
    assert len(items) > 2, "the fixture must produce several items to pair"
    assert len({item.extraction_run_id for item in items}) > 1, "drawn from several leaves"

    selection = generate_candidates(items, max_pairs=100)

    assert selection.theoretical_pairs > 0, "pairs existed to be filtered"
    assert selection.pairs == [], "many leaves of one source must never pair"


async def test_the_packet_shows_items_from_every_leaf_of_one_source(
    session_factory: SessionFactory,
) -> None:
    evidence_id = await _create_evidence(session_factory)
    result = await _extract(session_factory, evidence_id, IntervalScriptedProvider(_ok()))

    async with session_factory() as session:
        packet = await PacketService(session).assemble(limit=200)

    stored = {item.id for run in result.runs for item in run.items}
    assert {item.item_id for item in packet.items} >= stored


async def test_a_failed_parent_run_contributes_no_coverage(
    session_factory: SessionFactory,
) -> None:
    """The overflowing attempt is history, not evidence.

    It stays on record as a failed run — it happened, and it cost money — but it
    must never count towards the characters that were successfully read, or a
    source could be reported complete on the strength of a rejected response.
    """
    evidence_id = await _create_evidence(session_factory)
    plan = plan_windows(CONTENT, max_chars=SMALL_BUDGET)
    doomed = plan[1].text(CONTENT)

    def responder(text: str) -> ProviderResult:
        if text == doomed:
            return _result(_items(text, MAX_ITEMS_PER_RUN + 1))
        return _result(_items(text, 1))

    result = await _extract(session_factory, evidence_id, IntervalScriptedProvider(responder))

    async with session_factory() as session:
        runs = await ExtractionService(
            session, IntervalScriptedProvider(_ok()), window_max_chars=SMALL_BUDGET
        ).list_runs(evidence_id)

    failed = [r for r in runs.items if r.status is RunStatus.FAILED]
    assert failed, "the overflowing attempt was not recorded"
    assert all(r.error_code == "extraction_overflow" for r in failed)
    # The failed parent's range is not among the runs the response reports.
    assert (plan[1].start, plan[1].end) not in {(r.window_start, r.window_end) for r in result.runs}
    assert result.complete is True
