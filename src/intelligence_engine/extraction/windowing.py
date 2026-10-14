"""Deterministic division of one evidence item into extraction windows.

The defect this exists for, measured rather than argued: a 60,241-character
conversation export sent as one prompt returned `stop_reason: max_tokens` with
`output_tokens` landing exactly on the configured 8,000 ceiling, truncated
mid-string, and was correctly rejected as invalid JSON. The same response had
already emitted 69 items, so raising the output ceiling would only have moved
the failure to `MAX_ITEMS_PER_RUN = 50`. The constrained resource is *output*
volume, and sending less input at a time is the lever that reduces it.

## Sizing is a heuristic; overflow detection is the mechanism

The first version of this module treated the budget as a guarantee, and the
real corpus falsified that three separate times. At 12,000 characters one
window of Event 2 returned 39 items and the next exceeded 50. At 6,000 a window
still exceeded 50. Given a single structural block of 4,412 characters and
nothing else, the provider truncated at the 8,000-token ceiling. Item count is
driven by how densely a passage states things, which no character budget can
see, so **no budget exists that makes overflow impossible.**

So the budget stopped being a correctness argument. It is now purely a cost
heuristic: pack generously, because most intervals succeed and each call pays a
fixed prompt overhead. Correctness comes from `subdivide` — an interval that
overflows is split and its children retried, recursively — and from
`coverage`, which proves after the fact that the intervals that did succeed
tile the source exactly. That inversion is the architecture: sizing makes
extraction cheap, and classified overflow plus deterministic subdivision makes
it correct.

**A window is extraction mechanics, never an evidence source.** Windows carry no
identity, no provenance and no lifetime of their own; they are byte ranges over
one `EvidenceItem` that stays the single logical source. That distinction is the
whole reason this lives here instead of in M1: splitting the export into several
`evidence_items` rows would have handed M4's cross-evidence filter — whose one
hard test is `first.evidence_id == second.evidence_id` — a way to pair two
halves of a single conversation and call the agreement corroboration.

## The invariant everything else rests on

Windows are **exact, contiguous, non-overlapping slices that cover the whole
content**:

    "".join(content[w.start:w.end] for w in plan) == content

Nothing is reformatted, re-joined, normalised or summarised on the way to the
provider. That is what lets grounding stay honest: a quote verified verbatim
inside `content[start:end]` is verbatim inside `content` by construction, so
per-window grounding *implies* source-level grounding rather than approximating
it. `tests/unit/test_windowing.py` asserts the reconstruction directly.

## The algorithm

1. **Split at natural block boundaries.** Markdown ATX headings (`#` to `######`)
   start a new block, because a heading is where a document itself says one
   subject ends. Absent headings, blank-line-separated paragraphs are the
   fallback, so the strategy degrades to something sensible on plain text.
2. **Pack** consecutive whole blocks into a window while they fit the budget.
3. **Split oversized blocks** deterministically — paragraphs, then lines, then a
   hard character cut — so a single enormous block cannot defeat the budget.

Deliberately not semantic chunking: no embeddings, no topic modelling, no model
call. Windowing must be a pure function of `(content, budget)` or reuse cannot be
keyed on it, and anything involving a model is neither pure nor free.

The heading rule is generic. It keys on Markdown syntax, never on the shape of
any particular corpus — no `E001` ids, no per-document parser, no hand-curated
boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from intelligence_engine.extraction.schemas import MIN_QUOTE_LENGTH

#: The shortest interval `subdivide` will split. Derived, not chosen: every
#: extracted item must carry a grounding quote of at least `MIN_QUOTE_LENGTH`
#: characters, so an interval that cannot hold two such quotes cannot be split
#: into two pieces that both yield an item. Below this width, splitting stops
#: buying anything and the honest move is to report the region unread.
#:
#: It is a floor, not the usual stopping point. Halving from the 6,000-character
#: default budget reaches it in nine steps, and real overflow resolves in one or
#: two, so this bound exists to guarantee termination rather than to be met.
MIN_SUBDIVISIBLE_CHARS = 2 * MIN_QUOTE_LENGTH

#: ATX headings only. Setext underlining (`===`) is deliberately unsupported:
#: recognising it needs lookahead over the previous line, and every document
#: this was built for uses ATX. Unrecognised heading styles are not an error —
#: they simply fall through to paragraph splitting.
_HEADING = re.compile(r"^#{1,6} \S", re.MULTILINE)

#: A blank line, i.e. a paragraph break. Used only when no headings exist.
_PARAGRAPH = re.compile(r"\n[ \t]*\n")


@dataclass(frozen=True)
class Window:
    """One half-open character range `[start, end)` over the parent content.

    **`(start, end)` is the identity.** The range is what a stored run records,
    what reuse is keyed on, and what completeness is computed from, because it
    is the only thing about a window that stays true when the plan changes
    around it. An interval that overflows is replaced by children covering the
    same characters; its siblings are untouched and must remain reusable, which
    an ordinal cannot promise — renumbering would silently redefine work that
    was already paid for.

    `index` survives as a transient ordinal for ordering and logging within a
    single plan. It is deliberately not persisted, not part of any key, and not
    meaningful across plans: children of a subdivided interval start again at
    zero, and that is harmless precisely because nothing durable reads it.
    """

    index: int
    start: int
    end: int

    def text(self, content: str) -> str:
        """The exact substring. No normalisation, ever — see the module docstring."""
        return content[self.start : self.end]


def plan_windows(content: str, *, max_chars: int) -> tuple[Window, ...]:
    """Divide `content` into ordered windows of at most `max_chars` characters.

    Pure and total: the same `(content, max_chars)` always yields the same tuple,
    and every input yields at least one window. Content that already fits is
    returned as a single window spanning it, which is what keeps small evidence
    byte-identical to the pre-windowing behaviour.

    Raises:
        ValueError: if `max_chars` is not positive, since no division of a
            document into non-positive-sized pieces terminates.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not content:
        return (Window(index=0, start=0, end=0),)
    if len(content) <= max_chars:
        return (Window(index=0, start=0, end=len(content)),)

    bounds: list[tuple[int, int]] = []
    for block_start, block_end in _blocks(content):
        if block_end - block_start > max_chars:
            bounds.extend(_split_oversized(content, block_start, block_end, max_chars))
        else:
            bounds.append((block_start, block_end))

    return tuple(
        Window(index=index, start=start, end=end)
        for index, (start, end) in enumerate(_pack(bounds, max_chars))
    )


def subdivide(content: str, start: int, end: int) -> tuple[Window, ...] | None:
    """Split one overflowing interval into contiguous children, or `None`.

    Called only when `[start, end)` produced a *classified* overflow — the
    interval demanded more than one bounded response could carry. Size is not
    why this exists: three separate measurements showed input size does not
    bound output cardinality, so no budget can be chosen that makes overflow
    impossible. Detecting overflow and splitting the interval that caused it is
    the correctness mechanism; the initial budget is only a cost heuristic.

    Children are produced by the same weaker-boundary cascade `plan_windows`
    already uses for an over-budget block — paragraphs, then lines, then a hard
    character cut — by asking for pieces of at most half the parent. Halving is
    what makes the recursion terminate at a rate that can be stated: an
    interval of length *n* reaches the minimum in at most `log2(n)` steps.

    Guarantees, all asserted in `tests/unit/test_windowing.py`:

    - at least two children;
    - every child strictly shorter than the parent;
    - children are contiguous, non-overlapping, and tile `[start, end)` exactly;
    - each child is an exact substring range of the original content, so a
      quote grounded in a child is verbatim in the source by construction —
      the same argument that holds for a top-level window;
    - pure: the same `(content, start, end)` always yields the same children.

    Returns `None` when the interval is at or below `MIN_SUBDIVISIBLE_CHARS`,
    which is the caller's signal that the recursion has honestly run out and
    the region must be reported unread rather than split further.
    """
    if end - start <= MIN_SUBDIVISIBLE_CHARS:
        return None
    bounds = _split_oversized(content, start, end, (end - start) // 2)
    if len(bounds) < 2:  # pragma: no cover - arithmetic above forbids it
        return None
    return tuple(
        Window(index=index, start=child_start, end=child_end)
        for index, (child_start, child_end) in enumerate(bounds)
    )


@dataclass(frozen=True)
class Coverage:
    """How much of one source has been read, stated as character ranges.

    Processing completeness is a tiling question, and this is the whole of it:
    do the successfully processed intervals lay end to end over `[0, length)`
    with no gap and no double-counting? Asking it this way is what lets the
    answer survive arbitrary subdivision — a parent replaced by its children
    covers exactly what the parent did, so the tiling is unchanged by the
    recursion that produced it.

    `gaps` names the unread regions exactly, so an incomplete source can say
    *which* characters were never read rather than only that something failed.

    **This is processing completeness, never semantic completeness.** It says
    every character of the source belonged to an interval that was extracted
    successfully. It says nothing whatever about whether the model found every
    meaningful item inside those characters, and no wording built on it may
    imply that it does.
    """

    length: int
    gaps: tuple[tuple[int, int], ...]
    overlaps: tuple[tuple[int, int], ...]

    @property
    def is_complete(self) -> bool:
        """Exact tiling of `[0, length)`: nothing missing, nothing counted twice.

        Overlaps block completeness as firmly as gaps do. An overlap means two
        current runs claim the same characters, so the intervals on record are
        not a partition of the source and the item counts drawn from them
        double-count. Reporting that as complete would be the quiet miscount
        this whole mechanism exists to prevent.
        """
        return not self.gaps and not self.overlaps

    @property
    def covered_chars(self) -> int:
        """Characters inside some successful interval. Overlap counted once."""
        return self.length - sum(end - start for start, end in self.gaps)


def coverage(intervals: Iterable[tuple[int, int]], *, length: int) -> Coverage:
    """Reduce successful intervals to gaps and overlaps over `[0, length)`.

    Pure, total and order-independent: the caller hands over whatever ranges it
    has and gets back the same answer for any ordering of them. Empty input
    over a non-empty source is one gap spanning everything, which is the
    correct reading of "nothing has been read".

    A zero-length source is complete when nothing overlaps, because `[0, 0)` is
    tiled by the empty set. That keeps empty evidence — which `plan_windows`
    already returns a single empty window for — from being permanently
    incomplete.
    """
    if length < 0:
        raise ValueError("length must not be negative")

    ordered = sorted(intervals)
    gaps: list[tuple[int, int]] = []
    overlaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in ordered:
        if start > cursor:
            gaps.append((cursor, start))
        elif start < cursor:
            overlaps.append((start, min(cursor, end)))
        cursor = max(cursor, end)
    if cursor < length:
        gaps.append((cursor, length))
    return Coverage(length=length, gaps=tuple(gaps), overlaps=tuple(overlaps))


def _blocks(content: str) -> list[tuple[int, int]]:
    """Natural boundaries: heading-led sections, else paragraphs, else the whole.

    Returned ranges are contiguous and cover `content` exactly. Each heading
    match starts a block; the preamble before the first heading is its own block
    so nothing before it is lost.
    """
    starts = [match.start() for match in _HEADING.finditer(content)]
    if not starts:
        starts = _paragraph_starts(content)
    if not starts:
        return [(0, len(content))]

    if starts[0] != 0:
        starts.insert(0, 0)
    return [
        (start, starts[position + 1] if position + 1 < len(starts) else len(content))
        for position, start in enumerate(starts)
    ]


def _paragraph_starts(content: str) -> list[int]:
    """Offsets just after each blank line — the fallback boundary for plain text."""
    return [match.end() for match in _PARAGRAPH.finditer(content)]


def _split_oversized(content: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Cut one over-budget block into contiguous sub-ranges within budget.

    Three passes, each a strictly weaker boundary than the last: paragraph
    breaks, then line breaks, then a hard character cut that always terminates.
    A block of one enormous unbroken line is rare in real evidence and must
    still not hang, so the last pass has no preconditions.
    """
    ranges: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        limit = min(cursor + max_chars, end)
        if limit == end:
            ranges.append((cursor, end))
            break
        cut = _last_boundary(content, cursor, limit)
        ranges.append((cursor, cut))
        cursor = cut
    return ranges


def _last_boundary(content: str, start: int, limit: int) -> int:
    """The latest safe cut in `(start, limit]`, preferring the strongest break.

    Never returns `start`, so the caller always advances.
    """
    window = content[start:limit]
    for pattern in ("\n\n", "\n"):
        position = window.rfind(pattern)
        if position > 0:
            return start + position + len(pattern)
    return limit


def _pack(bounds: list[tuple[int, int]], max_chars: int) -> list[tuple[int, int]]:
    """Merge consecutive whole blocks while they fit, preserving contiguity.

    Blocks are only ever joined, never reordered or dropped, so the packed
    ranges still tile the content end to end.
    """
    packed: list[tuple[int, int]] = []
    for start, end in bounds:
        if packed and end - packed[-1][0] <= max_chars:
            packed[-1] = (packed[-1][0], end)
        else:
            packed.append((start, end))
    return packed
