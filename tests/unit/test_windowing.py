"""The windowing algorithm, tested as the pure function it is.

The invariant every downstream guarantee rests on is the reconstruction: windows
are exact, contiguous, non-overlapping slices that tile the whole content. If
that ever stops being true, per-window grounding stops implying source-level
grounding, and the citation chain quietly becomes a claim rather than a proof.
So it is asserted directly, and asserted again at the scale of the 60KB export
that caused this module to exist, through a synthetic stand-in for it.
"""

from __future__ import annotations

import itertools

import pytest

from intelligence_engine.extraction.windowing import (
    MIN_SUBDIVISIBLE_CHARS,
    Window,
    coverage,
    plan_windows,
    subdivide,
)
from tests.synthetic_sources import LONG_TRANSCRIPT_CHARS, long_transcript


def _reconstruct(content: str, windows: tuple[Window, ...]) -> str:
    return "".join(window.text(content) for window in windows)


# --- the critical invariant ------------------------------------------------


@pytest.mark.parametrize("budget", [50, 120, 500, 5_000])
def test_windows_tile_the_content_exactly(budget: int) -> None:
    """Concatenating every window reproduces the source, byte for byte.

    Not "approximately" and not "modulo whitespace". Any gap loses evidence
    silently; any overlap double-counts it; any normalisation breaks the
    verbatim-quote guarantee that grounding depends on.
    """
    content = "\n\n".join(
        f"## Section {index}\n\nBody line for section {index}." for index in range(40)
    )

    windows = plan_windows(content, max_chars=budget)

    assert _reconstruct(content, windows) == content


def test_windows_are_contiguous_and_ordered() -> None:
    content = "\n\n".join(f"# Heading {index}\n\nText {index}" for index in range(30))

    windows = plan_windows(content, max_chars=200)

    assert [window.index for window in windows] == list(range(len(windows)))
    assert windows[0].start == 0
    assert windows[-1].end == len(content)
    for earlier, later in itertools.pairwise(windows):
        assert earlier.end == later.start, "a gap or overlap between windows"


def test_every_window_respects_the_budget() -> None:
    content = "\n\n".join(f"## S{index}\n\n{'x' * 300}" for index in range(20))

    windows = plan_windows(content, max_chars=1_000)

    assert all(window.end - window.start <= 1_000 for window in windows)


# --- determinism -----------------------------------------------------------


def test_planning_is_deterministic() -> None:
    """Same content, same budget, same windows — the precondition for reuse.

    Reuse is keyed on `(window_start, window_end, hash of that interval's
    text)`. If planning varied at all, a cached interval would be matched
    against a different slice and stale items would be served as current.
    """
    content = "\n\n".join(f"### Item {index}\n\ncontent {index}" for index in range(50))

    first = plan_windows(content, max_chars=400)
    second = plan_windows(content, max_chars=400)

    assert first == second


def test_a_different_budget_is_a_different_plan() -> None:
    content = "\n\n".join(f"## S{index}\n\nbody {index}" for index in range(40))

    assert plan_windows(content, max_chars=200) != plan_windows(content, max_chars=2_000)


# --- boundaries ------------------------------------------------------------


def test_headings_are_preferred_boundaries() -> None:
    """Windows start at a heading when the budget allows, not mid-sentence."""
    content = "\n\n".join(f"## Section {index}\n\n{'body ' * 30}" for index in range(12))

    windows = plan_windows(content, max_chars=600)

    starts = [window.text(content) for window in windows[1:]]
    assert all(text.startswith("## Section") for text in starts)


def test_paragraphs_are_the_fallback_when_there_are_no_headings() -> None:
    content = "\n\n".join(f"Paragraph number {index} with some text." for index in range(60))

    windows = plan_windows(content, max_chars=300)

    assert len(windows) > 1
    assert _reconstruct(content, windows) == content
    # No window begins mid-word, which is what a naive character cut would do.
    assert all(not window.text(content).startswith(" ") for window in windows)


def test_an_oversized_single_block_is_split_rather_than_emitted_whole() -> None:
    """One section larger than the budget must not defeat the budget.

    This is the case that would otherwise reproduce the original defect inside
    the fix: a single enormous section sent whole is the same too-large prompt.
    """
    huge = "## Enormous\n\n" + "\n".join(f"line {index}" for index in range(2_000))
    content = f"# Doc\n\nintro\n\n{huge}"

    windows = plan_windows(content, max_chars=1_000)

    assert all(window.end - window.start <= 1_000 for window in windows)
    assert _reconstruct(content, windows) == content


def test_a_block_with_no_break_at_all_still_terminates_within_budget() -> None:
    """The hard character cut, which has no preconditions and must always halt."""
    content = "#" + "x" * 5_000

    windows = plan_windows(content, max_chars=500)

    assert len(windows) >= 10
    assert all(window.end - window.start <= 500 for window in windows)
    assert _reconstruct(content, windows) == content


# --- the small-document path, which must not change ------------------------


def test_content_within_budget_is_exactly_one_window() -> None:
    """Most evidence is a note. A note must behave as it always has."""
    content = "# A short note\n\nSomething happened."

    windows = plan_windows(content, max_chars=6_000)

    assert windows == (Window(index=0, start=0, end=len(content)),)
    assert windows[0].text(content) == content


def test_empty_content_yields_one_empty_window() -> None:
    """Total, not partial: no input shape returns zero windows."""
    assert plan_windows("", max_chars=100) == (Window(index=0, start=0, end=0),)


def test_a_non_positive_budget_is_refused() -> None:
    """Loudly, because the alternative is a loop that never terminates."""
    with pytest.raises(ValueError, match="max_chars must be positive"):
        plan_windows("anything", max_chars=0)


# --- at the scale of the document that caused this to exist ----------------


def test_the_long_stand_in_keeps_the_shape_that_made_the_real_export_hard() -> None:
    """The synthetic stand-in is only useful while it keeps these properties."""
    content = long_transcript()

    assert len(content) == LONG_TRANSCRIPT_CHARS
    assert "\r" not in content
    assert sum(1 for char in content if ord(char) > 127) > 500
    assert len(content.encode("utf-8")) > len(content), "offsets must differ from bytes"
    assert content.count("\n\n") > 200
    assert max(len(line) for line in content.split("\n")) < 6_000


def test_a_long_export_tiles_exactly_and_stays_within_budget() -> None:
    """A 60,241-character export, at the shipped default budget.

    The measurement that matters is not the window count but that the document
    survives division unaltered.
    """
    content = long_transcript()

    windows = plan_windows(content, max_chars=6_000)

    assert len(windows) > 1, "the document that broke extraction must divide"
    assert _reconstruct(content, windows) == content
    assert all(window.end - window.start <= 6_000 for window in windows)


# --- subdivision: the correctness mechanism --------------------------------
#
# Sizing is a heuristic and was falsified three times against the real corpus.
# These tests cover what replaced it: an interval that overflows is split
# deterministically, and the split preserves every property the source-grounding
# argument depends on.


@pytest.mark.parametrize("budget", [50, 120, 500])
def test_children_tile_their_parent_exactly(budget: int) -> None:
    """The tiling invariant, one level down.

    A subdivided parent must be replaced by its children with no gap, no
    overlap and no reordering — otherwise coverage computed from leaves would
    disagree with the source the parent read.
    """
    content = "\n\n".join(f"## Section {index}\n\nbody text {index}" for index in range(30))
    parent = plan_windows(content, max_chars=budget)[0]

    children = subdivide(content, parent.start, parent.end)

    assert children is not None
    assert children[0].start == parent.start
    assert children[-1].end == parent.end
    for left, right in itertools.pairwise(children):
        assert left.end == right.start
    assert _reconstruct(content, children) == content[parent.start : parent.end]


def test_subdivision_produces_at_least_two_strictly_smaller_children() -> None:
    """Strict shrinkage is what makes the recursion terminate.

    A child equal to its parent would be re-sent unchanged, overflow again, and
    recurse forever at full price.
    """
    content = "\n\n".join(f"paragraph number {index} with some body" for index in range(40))

    children = subdivide(content, 0, len(content))

    assert children is not None
    assert len(children) >= 2
    assert all(child.end - child.start < len(content) for child in children)
    assert all(child.end > child.start for child in children)


def test_subdivision_is_deterministic() -> None:
    """Same parent, same children — the precondition for reusing a child."""
    content = "\n\n".join(f"## S{index}\n\nbody {index}" for index in range(25))

    first = subdivide(content, 0, len(content))
    second = subdivide(content, 0, len(content))

    assert first == second


def test_children_are_exact_substrings_of_the_original_source() -> None:
    """Grounding survives the recursion.

    A quote verified inside a child must be verbatim in the whole document, and
    that holds only if the child is an unaltered slice of it — no re-joining, no
    normalisation, no trimming at the split.
    """
    content = "\n\n".join(f"## S{index}\n\nquoted body {index}" for index in range(20))

    children = subdivide(content, 0, len(content))

    assert children is not None
    for child in children:
        assert child.text(content) == content[child.start : child.end]
        assert child.text(content) in content


def test_subdivision_refuses_an_interval_at_the_minimum_width() -> None:
    """The floor of the recursion, returned rather than raised.

    Below two minimum-length quotes there is nothing left to split into pieces
    that could each yield an item, so the caller is told to stop and report the
    region unread instead of descending forever.
    """
    content = "x" * 200

    assert subdivide(content, 0, MIN_SUBDIVISIBLE_CHARS) is None
    assert subdivide(content, 0, MIN_SUBDIVISIBLE_CHARS - 1) is None
    assert subdivide(content, 10, 10) is None


def test_subdivision_terminates_on_one_unbroken_line() -> None:
    """The pathological shape: no paragraph break, no line break, nothing.

    The hard character cut has no preconditions precisely so this case cannot
    hang. Repeated splitting must reach the minimum width in a bounded number of
    steps rather than stalling on an interval it cannot divide.
    """
    content = "x" * 5_000

    start, end, steps = 0, 5_000, 0
    while (children := subdivide(content, start, end)) is not None:
        steps += 1
        assert steps < 100, "subdivision failed to converge on an unbreakable line"
        start, end = children[0].start, children[0].end

    assert end - start <= MIN_SUBDIVISIBLE_CHARS


# --- coverage: processing completeness as a tiling -------------------------


def test_exact_tiling_is_complete() -> None:
    covered = coverage([(0, 10), (10, 25), (25, 40)], length=40)

    assert covered.is_complete
    assert covered.gaps == ()
    assert covered.overlaps == ()
    assert covered.covered_chars == 40


def test_coverage_is_independent_of_the_order_intervals_arrive_in() -> None:
    """Runs come back from the database in whatever order; the answer must not."""
    shuffled = coverage([(25, 40), (0, 10), (10, 25)], length=40)

    assert shuffled.is_complete


def test_a_gap_is_incomplete_and_named_exactly() -> None:
    """A bare "something failed" cannot be checked; a character range can."""
    covered = coverage([(0, 10), (25, 40)], length=40)

    assert not covered.is_complete
    assert covered.gaps == ((10, 25),)
    assert covered.covered_chars == 25


def test_a_missing_tail_is_a_gap() -> None:
    covered = coverage([(0, 10)], length=40)

    assert not covered.is_complete
    assert covered.gaps == ((10, 40),)


def test_no_intervals_at_all_is_one_gap_spanning_everything() -> None:
    covered = coverage([], length=40)

    assert not covered.is_complete
    assert covered.gaps == ((0, 40),)
    assert covered.covered_chars == 0


def test_overlapping_intervals_are_never_reported_complete() -> None:
    """Overlap means two current runs claim the same characters.

    The intervals on record are then not a partition of the source, so item
    counts drawn from them double-count. Reporting that as complete would be
    exactly the quiet miscount this mechanism exists to prevent — so it blocks
    completeness as firmly as a gap does, even though every character is
    touched.
    """
    covered = coverage([(0, 25), (20, 40)], length=40)

    assert covered.gaps == ()
    assert covered.overlaps == ((20, 25),)
    assert not covered.is_complete


def test_a_subdivided_parent_replaced_by_children_covers_what_the_parent_did() -> None:
    """The property that makes completeness survive subdivision.

    Coverage must give the same answer whether an interval succeeded whole or
    was split and its children succeeded, because they describe the same
    characters. If this failed, a source could be fully read and still be
    reported incomplete purely because of how it happened to be divided.
    """
    whole = coverage([(0, 20), (20, 60)], length=60)
    subdivided = coverage([(0, 20), (20, 35), (35, 48), (48, 60)], length=60)

    assert whole.is_complete
    assert subdivided.is_complete
    assert whole.covered_chars == subdivided.covered_chars


def test_an_empty_source_is_complete() -> None:
    """`[0, 0)` is tiled by the empty set, so empty evidence is not stuck."""
    assert coverage([(0, 0)], length=0).is_complete
    assert coverage([], length=0).is_complete


def test_a_negative_length_is_refused() -> None:
    with pytest.raises(ValueError, match="length must not be negative"):
        coverage([], length=-1)


def test_a_long_export_plan_tiles_under_coverage() -> None:
    """The planner and the completeness check must agree on a long document.

    They are separate computations — one divides, one verifies — and this is the
    only test that they answer the same way about a source the size of the one
    that caused both to exist.
    """
    content = long_transcript()
    windows = plan_windows(content, max_chars=6_000)

    covered = coverage(((w.start, w.end) for w in windows), length=len(content))

    assert covered.is_complete
    assert covered.covered_chars == len(content) == LONG_TRANSCRIPT_CHARS
