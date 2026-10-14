"""Deterministic candidate generation. Pure functions, no database, no model.

Candidate generation and relation classification are two different problems with
different failure modes, and they are kept apart on purpose. This module answers
only: *which pairs are worth paying a model to look at?*

Naive all-pairs classification is forbidden. For N relatable items there are
N(N-1)/2 pairs — 100 items is 4,950 — and sending those to a model is both
ruinous and pointless, because the overwhelming majority are unrelated.

Three stages, all deterministic and all free:

1. **Filter** — a hard rule, applied for reasons, not for volume.
2. **Rank** — a lexicographic order over cheap signals already in the schema.
3. **Cap** — an absolute bound on how many pairs one analysis can cost.

The filter is *cross-evidence only*: two items extracted from the same evidence
item are never paired. This is the milestone's own definition rather than a
convenience — M4 is cross-evidence intelligence, and two adjacent sentences in
one note are already read together by anyone reading that note. The cost is real
and named: an intra-note qualification is invisible to V1.

The ranking is where the honesty matters. **Lexical overlap alone is not a
usable filter**, and this was measured rather than assumed. On the pair this
milestone exists for —

    "the fastest way to get good is to reverse-engineer a serious project"
    "the first six weeks were mostly reading, not writing"

— content-word overlap is exactly **zero**. A lexical threshold would discard the
single most valuable relation in the corpus. So overlap is a tiebreak, never a
gate, and the primary signal is structural: M2 already records a `stance`, and a
`recommended` claim paired with an `asserted` one is precisely the shape that
produces a qualification. That signal costs nothing, comes from the existing
schema, and does not depend on the two statements sharing vocabulary.

No embeddings, no vector store. The deferred-systems rule requires deterministic
narrowing to be *demonstrated* inadequate first; the place that would show is
recall at the cap, which the Gate B evaluation measures.
"""

from __future__ import annotations

import datetime as dt
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from intelligence_engine.entities.schemas import ResolutionRole
from intelligence_engine.extraction.schemas import ItemKind

#: Words carrying no topical signal. Deliberately small and fixed: a large
#: stopword list is a tuning knob, and this score is a tiebreak, not a gate.
_STOPWORD_TEXT = """
a an the and or but if then than that this these those of in on at to for from with
by as is are was were be been being do does did doing have has had having he she it
they them his her its their you your i me my we us our not no so up out over under
about into more most other some such only own same too very can will just should now
what which who whom when where why how all any both each few nor s t don
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class RelatableItem:
    """One grounded item, flattened with everything a relation needs.

    Carries its own provenance (`resolution_id`, `extraction_run_id`,
    `evidence_id`) so a candidate pair can become a fully provenanced relation
    without a second query, and its evidence's `source_label` / `captured_at` so
    the classifier can tell two occasions apart. That context is what stops a
    later revision of an earlier statement from reading as a contradiction.

    `claimed_by` and `subject` are two fields rather than one merged
    "attributed_to", and never both set. A reported claim has a speaker; an
    observation has a subject it is *about*. Collapsing them here would let the
    classifier read "her public profile lists Solidity" as something Zi Lan asserted,
    which is precisely the confusion the whole reported-claim design prevents.
    The distinction is enforced in the database by the role/kind CHECK; this
    keeps it true in memory as well.
    """

    item_id: uuid.UUID
    kind: ItemKind
    #: The role M3 actually asserted, carried rather than inferred from `kind`.
    #: The two are one-to-one today, and deriving one from the other is exactly
    #: how "this entity said it" came to be rendered as "this is about them".
    #: The connection shown to the classifier must be the asserted one.
    role: ResolutionRole
    text: str
    stance: str | None
    claimed_by: str | None
    subject: str | None
    grounding_quote: str
    extraction_run_id: uuid.UUID
    evidence_id: uuid.UUID
    resolution_id: uuid.UUID
    source_label: str | None
    captured_at: dt.datetime | None
    evidence_created_at: dt.datetime
    position: int


@dataclass(frozen=True, slots=True)
class CandidatePair:
    """Two items proposed for classification, in canonical order.

    `left.item_id < right.item_id` always, matching the database's CHECK, so the
    pair has one identity regardless of the order it was generated in.
    """

    left: RelatableItem
    right: RelatableItem
    lexical_overlap: float
    stance_complementary: bool

    @property
    def key(self) -> tuple[uuid.UUID, uuid.UUID]:
        return (self.left.item_id, self.right.item_id)


class CandidateSelection(NamedTuple):
    """What narrowing produced, and what it started from."""

    pairs: list[CandidatePair]
    theoretical_pairs: int
    filtered_pairs: int
    capped: bool


def content_words(text: str) -> frozenset[str]:
    """Lowercased alphanumeric tokens with stopwords removed."""
    return frozenset(word for word in _WORD_RE.findall(text.lower()) if word not in _STOPWORDS)


def lexical_overlap(left: RelatableItem, right: RelatableItem) -> float:
    """Jaccard similarity over content words of the claim text and its quote.

    A weak signal used only to break ties. Two claims can be deeply related and
    share no vocabulary at all; two claims can share a great deal of vocabulary
    and have no relationship worth surfacing. Both cases are real, and both are
    the classifier's problem rather than this function's.
    """
    left_words = content_words(f"{left.text} {left.grounding_quote}")
    right_words = content_words(f"{right.text} {right.grounding_quote}")
    if not left_words or not right_words:
        return 0.0
    union = left_words | right_words
    return len(left_words & right_words) / len(union)


def is_stance_complementary(left: RelatableItem, right: RelatableItem) -> bool:
    """True when exactly one side is offered as advice.

    Advice plus a report of what actually happened is the shape a qualification
    takes. Two recommendations, or two plain assertions, are more often either
    reinforcement or nothing — still worth classifying, just later in the queue.
    """
    stances = {left.stance, right.stance}
    return "recommended" in stances and len(stances) == 2


def canonical_pair(
    first: RelatableItem, second: RelatableItem
) -> tuple[RelatableItem, RelatableItem]:
    """Order two items so a pair has exactly one representation."""
    return (first, second) if first.item_id < second.item_id else (second, first)


def generate_candidates(items: Sequence[RelatableItem], *, max_pairs: int) -> CandidateSelection:
    """Narrow an entity's relatable items to a bounded, ranked set of pairs.

    Cheap in the way that matters: this is O(N²) in *comparisons*, which are
    string operations on a handful of rows, and O(1) in model calls. The cap is
    what bounds cost, and it is absolute — an entity with a hundred evidence
    items costs the same as one with ten.
    """
    if max_pairs < 0:
        raise ValueError("max_pairs must not be negative")

    count = len(items)
    theoretical = count * (count - 1) // 2

    scored: list[CandidatePair] = []
    for index, first in enumerate(items):
        for second in items[index + 1 :]:
            # The one hard filter. Same-evidence pairs are not cross-evidence
            # intelligence, and a reader of that note already has both lines.
            if first.evidence_id == second.evidence_id:
                continue
            left, right = canonical_pair(first, second)
            scored.append(
                CandidatePair(
                    left=left,
                    right=right,
                    lexical_overlap=lexical_overlap(left, right),
                    stance_complementary=is_stance_complementary(left, right),
                )
            )

    # Lexicographic, with no invented weights: advice/experience pairs first,
    # then vocabulary overlap, then a fully deterministic tiebreak so the same
    # inputs always produce the same batch - which is what makes reuse and the
    # evaluation reproducible.
    scored.sort(
        key=lambda pair: (
            not pair.stance_complementary,
            -pair.lexical_overlap,
            pair.left.evidence_created_at,
            pair.left.position,
            str(pair.left.item_id),
            str(pair.right.item_id),
        )
    )

    return CandidateSelection(
        pairs=scored[:max_pairs],
        theoretical_pairs=theoretical,
        filtered_pairs=len(scored),
        capped=len(scored) > max_pairs,
    )


def batched(pairs: Sequence[CandidatePair], *, size: int) -> list[list[CandidatePair]]:
    """Split pairs into bounded batches.

    Bounded on purpose in both directions: one pair per call wastes the prompt,
    and an enormous prompt trades a real quality risk for a small saving. Each
    batch is independently validated and independently persisted.
    """
    if size < 1:
        raise ValueError("batch size must be at least 1")
    return [list(pairs[start : start + size]) for start in range(0, len(pairs), size)]
