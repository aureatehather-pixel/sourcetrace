"""Loading and scoring the hand-labelled pair corpus.

Kept separate from the runner so the unit suite can validate the corpus itself —
coverage of every trap category, unique ids, `expected` always inside
`acceptable`, `forbidden` never overlapping it — without any provider.

Scoring is deliberately **not** one accuracy number. The whole point of M4's
error asymmetry is that the mistakes differ in cost, so they are counted
separately:

- `false_contradiction` — the worst routine failure. A pair whose forbidden set
  contains `contradicts`, answered `contradicts`.
- `false_relation` — a meaningful relation asserted where none was acceptable.
- `missed_relation` — `no_meaningful_relation` where a meaningful one was
  expected. The cheapest failure, and acceptable in V1.
- `wrong_meaningful` — a meaningful relation, but the wrong one (and not merely
  unlisted-but-defensible).
- `wrong_direction` — right label, backwards.

An aggregate agreement figure is reported too, but it is reported *after* the
categories, because it is the least informative number on the page.

## Why the aggregate is not merely uninformative but actively misleading

A classifier that answers `no_meaningful_relation` to every pair — knowing
nothing, reading nothing, costing nothing — scores **25 of 37** on this corpus
and lands in no failure category at all: no false contradictions, no false
relations, no wrong labels, no wrong directions, no malformed batches. It is a
`return` statement, and on the headline number it beats a bad model.

That is a property of the corpus, not a defect in it. Declining is the right
answer for most pairs of statements, so a corpus that reflects reality has a high
floor, and any measure that rewards agreement rewards abstention.

`abstention_baseline` computes that floor from the corpus itself rather than
from a number anyone chose, and `meaningful_hits` is the one measure abstention
cannot reach: pairs where declining is *not* acceptable, answered with the right
relation. The baseline scores exactly zero there by construction. Those two
functions are what let a gate be stated without inventing a threshold.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from intelligence_engine.intelligence.schemas import MEANINGFUL_RELATION_TYPES, RelationType

GOLD_SET_PATH = Path(__file__).resolve().parent / "relation_gold_set.json"

#: Every trap the corpus is required to exercise. A category dropping to zero
#: examples is a corpus regression, and a unit test fails on it.
REQUIRED_CATEGORIES = frozenset(
    {
        "clear_reinforcement",
        "clear_qualification",
        "clear_contradiction",
        "no_relation",
        "false_contradiction_trap",
        "attribution_trap",
        "trivial_similarity_trap",
        "useful_non_obvious",
        # Added from real M4 output, not invented: relations that are defensible
        # and worth nothing. Invariant 15 says these do not count as intelligence,
        # so the corpus has to contain some.
        "useless_relation",
    }
)

#: Words that turn an explanation into a causal claim. M4 does not do causal
#: inference, so any hit here is a finding regardless of the label being right.
CAUSAL_MARKERS = (
    "caused",
    "causes",
    "causing",
    "led to",
    "leads to",
    "resulted in",
    "results in",
    "because of",
    "due to",
    "as a result",
    "therefore",
    "consequently",
)


@dataclass(frozen=True, slots=True)
class GoldStatement:
    statement: str
    stance: str | None
    claimed_by: str | None
    quote: str
    context: str
    #: Defaults keep every pre-existing entry valid: the corpus was claim-only
    #: until observations became reachable through the `about` role.
    kind: str = "reported_claim"
    subject: str | None = None


@dataclass(frozen=True, slots=True)
class GoldPair:
    """One hand-labelled pair.

    `acceptable` exists because some pairs are genuinely borderline and a single
    forced label would be fake precision. `forbidden` is what the traps actually
    test: a false-contradiction trap is failed by answering `contradicts`, not by
    disagreeing with `expected`.
    """

    id: str
    category: str
    note: str
    expected: RelationType
    expected_direction: str | None
    acceptable: frozenset[RelationType]
    forbidden: frozenset[RelationType]
    a: GoldStatement
    b: GoldStatement
    #: The entity the pairs are analysed under, exactly as production names it.
    #: The default is deliberately anonymous, but a pair that only misbehaves when
    #: the classifier is told every statement belongs to ONE named person must say
    #: so, or it silently stops testing anything. The speaker/subject traps
    #: (gold_033-035) are precisely that shape.
    entity_label: str = "various speakers (evaluation corpus)"


@dataclass
class Outcome:
    """What the classifier did with one pair, and what that counts as."""

    pair: GoldPair
    predicted: RelationType | None
    direction: str | None
    explanation: str | None
    verdict: str
    causal_language: bool = False


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)

    def count(self, verdict: str) -> int:
        return sum(1 for outcome in self.outcomes if outcome.verdict == verdict)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def causal_explanations(self) -> list[Outcome]:
        return [outcome for outcome in self.outcomes if outcome.causal_language]

    def by_verdict(self, verdict: str) -> list[Outcome]:
        return [outcome for outcome in self.outcomes if outcome.verdict == verdict]


def load_gold_set(path: Path = GOLD_SET_PATH) -> list[GoldPair]:
    """Read and structurally validate the corpus.

    Raises rather than skipping a malformed entry: an evaluation fixture that
    quietly drops examples reports better numbers than the classifier earned.

    Every problem is collected across the whole corpus before raising, so a
    hand-grown corpus is repaired a handful of entries at a time instead of one
    problem at a time. The corpus is not usable until it is clean.
    """
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    pairs = [_to_pair(entry) for entry in raw["pairs"]]

    # Problems are collected as (corpus index, message) pairs so the final error
    # reads in corpus order. Duplicate ids are reported once per duplicated id,
    # anchored at that id's first occurrence, because they have no single pair
    # to attach to.
    problems: list[tuple[int, str]] = []

    first_occurrence: dict[str, int] = {}
    duplicated: dict[str, int] = {}
    for index, pair in enumerate(pairs):
        if pair.id in first_occurrence:
            duplicated[pair.id] = first_occurrence[pair.id]
        else:
            first_occurrence[pair.id] = index
    problems.extend(
        (anchor, f"Duplicate gold pair ids: {pair_id}") for pair_id, anchor in duplicated.items()
    )

    for index, pair in enumerate(pairs):
        if pair.expected not in pair.acceptable:
            problems.append((index, f"{pair.id}: expected label is not in its own acceptable set"))
        overlap = pair.acceptable & pair.forbidden
        if overlap:
            problems.append(
                (index, f"{pair.id}: labels are both acceptable and forbidden: {overlap}")
            )
        if pair.expected is RelationType.QUALIFIES and pair.expected_direction is None:
            problems.append((index, f"{pair.id}: expects 'qualifies' but states no direction"))

    if problems:
        problems.sort(key=lambda problem: problem[0])
        raise ValueError("\n".join(message for _, message in problems))

    return pairs


def score(pair: GoldPair, predicted: RelationType | None, direction: str | None) -> str:
    """Classify one result into an error category.

    Order matters: a false contradiction is reported as a false contradiction
    even though it is also a wrong label, because that is the failure the error
    asymmetry cares about.
    """
    if predicted is None:
        return "malformed"

    if predicted is RelationType.CONTRADICTS and RelationType.CONTRADICTS in pair.forbidden:
        return "false_contradiction"

    if predicted in pair.acceptable:
        if (
            predicted is RelationType.QUALIFIES
            and pair.expected_direction is not None
            and direction != pair.expected_direction
        ):
            return "wrong_direction"
        return "correct"

    if predicted is RelationType.NO_MEANINGFUL_RELATION:
        return "missed_relation"

    if RelationType.NO_MEANINGFUL_RELATION in pair.acceptable:
        # A meaningful relation asserted where declining was the right answer.
        return "false_relation"

    if predicted in MEANINGFUL_RELATION_TYPES:
        return "wrong_meaningful"

    return "wrong_meaningful"


#: Verdicts that are failures. `correct` is the only thing that is not one, and
#: `missed_relation` is included here even though it is named the cheapest
#: failure: cheap is not free, and the gate below compares *counts* against a
#: baseline rather than judging any of them acceptable in the abstract.
FAILURE_VERDICTS = (
    "false_contradiction",
    "false_relation",
    "wrong_meaningful",
    "wrong_direction",
    "missed_relation",
    "malformed",
)


def requires_meaningful(pair: GoldPair) -> bool:
    """Whether declining is unacceptable for this pair.

    The pairs abstention cannot answer. Read from `acceptable` rather than from
    `expected`, because a pair whose acceptable set contains both a relation and
    `no_meaningful_relation` is one the corpus itself declines to force — and
    counting it here would credit a classifier for a judgement the labellers
    would not make.
    """
    return RelationType.NO_MEANINGFUL_RELATION not in pair.acceptable


def meaningful_hits(report: Report) -> int:
    """Correct answers on the pairs where declining was not an option.

    The measure an always-abstain classifier scores zero on, by construction and
    not by tuning. Everything else this module counts can be gamed by refusing to
    answer; this cannot.
    """
    return sum(
        1
        for outcome in report.outcomes
        if outcome.verdict == "correct" and requires_meaningful(outcome.pair)
    )


def abstention_baseline(pairs: Sequence[GoldPair]) -> Report:
    """The corpus scored against a classifier that always declines.

    Not a hypothetical and not a hard-coded table: the real `score` function over
    the real corpus, so the floor moves when the corpus moves and cannot fall out
    of date. `direction` is `None` because declining has none, and an explanation
    is omitted for the same reason the real classifier omits one.
    """
    report = Report()
    for pair in pairs:
        report.outcomes.append(
            Outcome(
                pair=pair,
                predicted=RelationType.NO_MEANINGFUL_RELATION,
                direction=None,
                explanation=None,
                verdict=score(pair, RelationType.NO_MEANINGFUL_RELATION, None),
            )
        )
    return report


def gate_failures(report: Report, baseline: Report) -> list[str]:
    """Why this classifier is not usable, or an empty list.

    **Every threshold here comes from the baseline. None was chosen.** That is
    the whole reason this can be a gate on 37 hand-labelled examples without
    claiming a precision the corpus does not have:

    1. *No worse than abstention in any failure category.* Abstention scores zero
       in five of the six, so this is mostly "do no harm" — and it is stated
       against the baseline rather than against a literal zero so that it stays
       true if the corpus ever gains a pair abstention gets wrong.
    2. *Strictly better on `meaningful_hits`.* Abstention scores zero there, so
       this asks for at least one relation the corpus insists on. It is a low
       bar, deliberately: any higher number would be one invented here.

    What passing proves: the classifier beats a `return` statement on this
    corpus. What it does not prove: that the classifier is good, that 37
    hand-labelled pairs measure anything, or that the labels are right. n is
    small by design — this falsifies, it does not measure — and the labelling was
    done by the same people who wrote the system. A gate this shape can only ever
    reject; it cannot certify.
    """
    failures: list[str] = []
    for verdict in FAILURE_VERDICTS:
        actual, floor = report.count(verdict), baseline.count(verdict)
        if actual > floor:
            failures.append(
                f"{verdict}: {actual} vs {floor} for always-no_meaningful_relation "
                "(a classifier must not be worse than declining)"
            )

    hits, baseline_hits = meaningful_hits(report), meaningful_hits(baseline)
    if hits <= baseline_hits:
        failures.append(
            f"meaningful relations found: {hits} vs {baseline_hits} for "
            "always-no_meaningful_relation (must be strictly better; declining "
            "to answer is free and already scores the aggregate)"
        )
    return failures


def has_causal_language(explanation: str | None) -> bool:
    if not explanation:
        return False
    lowered = explanation.lower()
    return any(marker in lowered for marker in CAUSAL_MARKERS)


def categories(pairs: Sequence[GoldPair]) -> set[str]:
    return {pair.category for pair in pairs}


def _to_pair(entry: dict[str, Any]) -> GoldPair:
    return GoldPair(
        id=entry["id"],
        category=entry["category"],
        note=entry["note"],
        expected=RelationType(entry["expected"]),
        expected_direction=entry.get("expected_direction"),
        acceptable=frozenset(RelationType(value) for value in entry["acceptable"]),
        forbidden=frozenset(RelationType(value) for value in entry["forbidden"]),
        a=_to_statement(entry["a"]),
        b=_to_statement(entry["b"]),
        entity_label=entry.get("entity_label", "various speakers (evaluation corpus)"),
    )


def _to_statement(entry: dict[str, Any]) -> GoldStatement:
    return GoldStatement(
        statement=entry["statement"],
        stance=entry.get("stance"),
        claimed_by=entry.get("claimed_by"),
        quote=entry["quote"],
        context=entry["context"],
        kind=entry.get("kind", "reported_claim"),
        subject=entry.get("subject"),
    )
