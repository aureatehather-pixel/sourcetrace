"""Gate B — measure what candidate narrowing keeps and what it throws away.

Costs nothing and needs no credentials: candidate generation is deterministic
software, which is the entire point of it. Run with:

    uv run python -m evaluation.run_candidate_eval

The fixture is a synthetic entity at a scale the real database has not reached
yet, because the question Gate B asks only becomes interesting once the cap
binds. It plants relations of two kinds:

- **valuable, low-overlap** — a recommendation in one note and the experience
  that qualifies it in another, sharing no content words. This is the shape of
  the one relation the milestone exists to find, and it is the shape a
  vocabulary-similarity filter destroys.
- **noisy, high-overlap** — pairs that share most of their vocabulary and have
  no useful relationship. These are what a similarity filter would rank first.

Both rankings are then run over the same pairs, so the reduction and the recall
are compared rather than asserted.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence

from intelligence_engine.entities.schemas import ResolutionRole
from intelligence_engine.extraction.schemas import ItemKind
from intelligence_engine.intelligence.candidates import (
    RelatableItem,
    canonical_pair,
    generate_candidates,
    is_stance_complementary,
    lexical_overlap,
)

PairKey = tuple[uuid.UUID, uuid.UUID]

_BASE = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

# Ten notes, four or five claims each. Sized so the cap genuinely binds.
_ADVICE = [
    "Rebuild something you already use rather than starting from a blank file.",
    "Ship the smallest thing that a stranger could actually run.",
    "Read the failing test before you read the stack trace.",
    "Interview the people who left, not only the people who stayed.",
]
_QUALIFICATIONS = [
    "The first six weeks were mostly reading, not writing.",
    "Almost nobody who tried it finished, and the ones who did already knew C.",
    "That only works when somebody else already wrote the failing test.",
    "Half of them would not say anything honest until months afterwards.",
]
_FILLER = [
    "The migration script for the reporting database runs in about four minutes.",
    "The migration script for the billing database runs in about six minutes.",
    "The migration script for the audit database runs in about five minutes.",
    "The conference moved to Pinehaven this year.",
    "The conference moved to Ipoh the year before.",
    "He prefers tabs and will not discuss it.",
    "Her editor crashes on generated protobuf files.",
    "The visa renewal comes up in September.",
    "The office is closed on Saturdays.",
    "Nobody has owned the onboarding flow since the reorg.",
]


def _stable_id(*parts: object) -> uuid.UUID:
    """A deterministic id, so Gate B reports the same numbers every run.

    The ranking's final tiebreak is the item id. With `uuid4()` the fixture's ids
    changed on every run, so the measured recall moved by a pair or two between
    runs and the comparison between rankings was not reproducible. Real items
    have fixed ids; the fixture should too.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, "|".join(str(part) for part in parts))


def _item(text: str, *, stance: str, evidence_index: int, position: int) -> RelatableItem:
    return RelatableItem(
        item_id=_stable_id("item", text, evidence_index, position),
        kind=ItemKind.REPORTED_CLAIM,
        # A claim reaches an entity as its speaker and carries no subject - the
        # invariant the database enforces with the role/kind CHECK. Neither field
        # participates in ranking, so the Gate B numbers below are unaffected by
        # them; they are set correctly so the fixture cannot model an item shape
        # the system would refuse to store.
        role=ResolutionRole.CLAIMED_BY,
        text=text,
        stance=stance,
        claimed_by="Subject",
        subject=None,
        grounding_quote=text.lower(),
        extraction_run_id=_stable_id("run", evidence_index),
        evidence_id=uuid.UUID(int=evidence_index),
        resolution_id=_stable_id("resolution", text, evidence_index, position),
        source_label=f"note {evidence_index}",
        captured_at=_BASE + dt.timedelta(days=evidence_index * 7),
        evidence_created_at=_BASE + dt.timedelta(days=evidence_index * 7),
        position=position,
    )


def build_profile() -> tuple[list[RelatableItem], set[PairKey]]:
    """A ten-note entity, with four known-valuable relations planted across notes.

    Each recommendation lands in one note and the experience qualifying it in a
    different one, so every planted relation is genuinely cross-evidence and
    genuinely low-overlap.
    """
    items: list[RelatableItem] = []
    valuable: set[PairKey] = set()

    for index, (advice, qualification) in enumerate(zip(_ADVICE, _QUALIFICATIONS, strict=True)):
        advice_item = _item(advice, stance="recommended", evidence_index=index, position=0)
        # Deliberately several notes later, never the same note.
        qualifying_item = _item(
            qualification, stance="asserted", evidence_index=index + 5, position=0
        )
        items.extend((advice_item, qualifying_item))
        left, right = canonical_pair(advice_item, qualifying_item)
        valuable.add((left.item_id, right.item_id))

    for index, text in enumerate(_FILLER * 3):
        items.append(_item(text, stance="asserted", evidence_index=index % 10, position=index + 1))

    return items, valuable


def _cross_evidence(items: Sequence[RelatableItem]) -> list[tuple[RelatableItem, RelatableItem]]:
    pairs: list[tuple[RelatableItem, RelatableItem]] = []
    for index, first in enumerate(items):
        for second in items[index + 1 :]:
            if first.evidence_id == second.evidence_id:
                continue
            pairs.append(canonical_pair(first, second))
    return pairs


def _lexical_only(items: Sequence[RelatableItem], *, max_pairs: int) -> list[PairKey]:
    """The ranking this design rejected, run for comparison.

    Same cross-evidence filter, ordered by vocabulary similarity alone. This is
    the obvious thing to build and the reason not to build it.
    """
    ordered = sorted(
        _cross_evidence(items),
        key=lambda pair: (
            -lexical_overlap(*pair),
            str(pair[0].item_id),
            str(pair[1].item_id),
        ),
    )
    return [(left.item_id, right.item_id) for left, right in ordered[:max_pairs]]


def _recency_tiebreak(items: Sequence[RelatableItem], *, max_pairs: int) -> list[PairKey]:
    """A considered-and-not-adopted alternative, kept so the choice is auditable.

    Same stance-first primary signal, but ties broken by newest evidence rather
    than by vocabulary overlap. It scores better on this fixture at the caps that
    matter — and it was still rejected, because the fixture plants every
    qualification in a *later* note than the advice it qualifies. Preferring this
    ranking on that evidence would be fitting to how the fixture was written
    rather than to anything about real data.

    The independent argument for it is real (new information is what you want
    checked against what you already know), so this is worth revisiting once an
    entity accumulates enough real notes for the cap to actually bind.
    """
    ordered = sorted(
        _cross_evidence(items),
        key=lambda pair: (
            not is_stance_complementary(*pair),
            -max(pair[0].evidence_created_at, pair[1].evidence_created_at).timestamp(),
            str(pair[0].item_id),
            str(pair[1].item_id),
        ),
    )
    return [(left.item_id, right.item_id) for left, right in ordered[:max_pairs]]


def main() -> None:
    items, valuable = build_profile()

    print(f"\n{'=' * 72}")  # noqa: T201
    print("GATE B - candidate narrowing")  # noqa: T201
    print("=" * 72)  # noqa: T201
    print(f"relatable items                 {len(items):>6}")  # noqa: T201

    uncapped = generate_candidates(items, max_pairs=10_000)
    print(f"theoretical pairs               {uncapped.theoretical_pairs:>6}")  # noqa: T201
    print(  # noqa: T201
        f"after cross-evidence filter     {uncapped.filtered_pairs:>6}"
        f"   ({_pct(uncapped.filtered_pairs, uncapped.theoretical_pairs)} of theoretical)"
    )

    print(f"\nplanted valuable relations      {len(valuable):>6}")  # noqa: T201
    retained_unfiltered = sum(1 for pair in uncapped.pairs if pair.key in valuable)
    print(f"  surviving the filter          {retained_unfiltered:>6}")  # noqa: T201

    total = len(valuable)
    print(  # noqa: T201
        f"\n{'cap':>5} {'cands':>6} {'reduction':>10}    "
        f"{'shipped':>9} {'lexical':>9} {'recency':>9}"
    )
    print("-" * 56)  # noqa: T201
    for cap in (10, 20, 40, 60, 100, 200):
        selection = generate_candidates(items, max_pairs=cap)
        shipped = sum(1 for pair in selection.pairs if pair.key in valuable)
        lexical = sum(1 for key in _lexical_only(items, max_pairs=cap) if key in valuable)
        recency = sum(1 for key in _recency_tiebreak(items, max_pairs=cap) if key in valuable)
        print(  # noqa: T201
            f"{cap:>5} {len(selection.pairs):>6} "
            f"{_pct(len(selection.pairs), uncapped.theoretical_pairs):>10}    "
            f"{shipped:>6}/{total:<2} {lexical:>6}/{total:<2} {recency:>6}/{total:<2}"
        )

    overlaps = [lexical_overlap(pair.left, pair.right) for pair in uncapped.pairs]
    planted = [
        lexical_overlap(pair.left, pair.right) for pair in uncapped.pairs if pair.key in valuable
    ]
    print(  # noqa: T201
        f"\nlexical overlap of the planted valuable pairs: {[round(value, 3) for value in planted]}"
    )
    print(  # noqa: T201
        f"highest lexical overlap anywhere in the candidate set: {max(overlaps):.3f}"
    )
    print(  # noqa: T201
        "\nThe planted relations score at or near zero on vocabulary similarity while "
        "unrelated\npairs score far higher. That is why overlap is a tiebreak here and "
        "never a gate."
    )


def _pct(part: int, whole: int) -> str:
    return f"{(part / whole * 100):.1f}%" if whole else "-"


if __name__ == "__main__":
    main()
