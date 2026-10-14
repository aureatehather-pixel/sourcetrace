"""The abstention baseline, and what the classifier gate can honestly reject.

These tests exist because of a specific finding: an always-`no_meaningful_relation`
classifier — no model, no reading, no cost — scores 25 of 37 on the corpus and
lands in *no* failure category, while `run_classifier_eval` printed that result
and exited 0 whatever it said. Something described as a gate proved only that a
model had been reachable.

So the negative case is the point here. Every test below either shows the gate
rejecting something it must reject, or shows that a threshold was derived from
the corpus rather than chosen by someone.
"""

from __future__ import annotations

from evaluation.gold_set import (
    FAILURE_VERDICTS,
    Outcome,
    Report,
    abstention_baseline,
    gate_failures,
    load_gold_set,
    meaningful_hits,
    requires_meaningful,
)

from intelligence_engine.intelligence.schemas import RelationType


def _perfect_report() -> Report:
    """Every pair answered with its own `expected` label and direction."""
    report = Report()
    for pair in load_gold_set():
        report.outcomes.append(
            Outcome(
                pair=pair,
                predicted=pair.expected,
                direction=pair.expected_direction,
                explanation="a derived reading of the two supplied statements",
                verdict="correct",
            )
        )
    return report


# --- the finding itself ----------------------------------------------------


def test_abstention_scores_well_on_the_aggregate_and_that_is_the_problem() -> None:
    """Measured, not asserted from memory: this is why the aggregate is not a gate."""
    baseline = abstention_baseline(load_gold_set())

    assert baseline.count("correct") > baseline.total / 2


def test_abstention_is_clean_in_every_failure_category_except_missing_relations() -> None:
    """Declining cannot be wrong in any way the error asymmetry cares about.

    It cannot contradict, cannot assert a relation, cannot pick the wrong one,
    cannot point one backwards, and cannot fail validation. Every high-severity
    category a reviewer looks at first is zero for a `return` statement.
    """
    baseline = abstention_baseline(load_gold_set())

    for verdict in FAILURE_VERDICTS:
        if verdict == "missed_relation":
            continue
        assert baseline.count(verdict) == 0, verdict
    assert baseline.count("missed_relation") > 0


def test_the_gate_rejects_the_abstention_classifier() -> None:
    """The negative case. If this ever passes, the gate has stopped being one."""
    baseline = abstention_baseline(load_gold_set())

    failures = gate_failures(baseline, baseline)

    assert failures
    assert any("meaningful relations found" in failure for failure in failures)


# --- the one measure abstention cannot reach -------------------------------


def test_abstention_finds_no_relation_the_corpus_insists_on() -> None:
    """Zero by construction, which is what makes "strictly better" a real bar."""
    baseline = abstention_baseline(load_gold_set())

    assert meaningful_hits(baseline) == 0


def test_the_corpus_actually_contains_pairs_where_declining_is_unacceptable() -> None:
    """Otherwise the bar above would be unreachable rather than low.

    A corpus in which every pair permitted `no_meaningful_relation` would make
    the gate impossible to pass, and it would look like a strict gate instead of
    a broken one.
    """
    required = [pair for pair in load_gold_set() if requires_meaningful(pair)]

    assert len(required) > 0
    assert all(RelationType.NO_MEANINGFUL_RELATION not in pair.acceptable for pair in required)


def test_a_borderline_pair_is_not_counted_as_one_abstention_must_answer() -> None:
    """`requires_meaningful` reads `acceptable`, not `expected`, and it must.

    A pair whose acceptable set holds both a relation and `no_meaningful_relation`
    is one the labellers declined to force. Counting it would credit a classifier
    for a judgement the corpus itself refused to make.
    """
    borderline = [
        pair
        for pair in load_gold_set()
        if pair.expected in {RelationType.REINFORCES, RelationType.QUALIFIES}
        and RelationType.NO_MEANINGFUL_RELATION in pair.acceptable
    ]

    assert borderline, "the corpus no longer contains a borderline pair to test this with"
    assert not any(requires_meaningful(pair) for pair in borderline)


# --- what the gate accepts -------------------------------------------------


def test_the_gate_accepts_a_classifier_that_answers_every_pair_correctly() -> None:
    """A gate that rejects everything proves as little as one that accepts everything."""
    baseline = abstention_baseline(load_gold_set())

    assert gate_failures(_perfect_report(), baseline) == []


def test_one_false_contradiction_is_enough_to_reject() -> None:
    """The worst routine failure, and abstention makes none, so the floor is zero."""
    baseline = abstention_baseline(load_gold_set())
    report = _perfect_report()
    report.outcomes[0].verdict = "false_contradiction"

    failures = gate_failures(report, baseline)

    assert any(failure.startswith("false_contradiction") for failure in failures)


def test_a_classifier_that_finds_relations_but_also_contradicts_is_still_rejected() -> None:
    """Beating the baseline on the one positive measure does not buy a harm allowance."""
    baseline = abstention_baseline(load_gold_set())
    report = _perfect_report()
    for outcome in report.outcomes:
        if not requires_meaningful(outcome.pair):
            outcome.verdict = "false_contradiction"
            break

    assert meaningful_hits(report) > meaningful_hits(baseline)
    assert gate_failures(report, baseline)


def test_extra_missed_relations_are_rejected_against_the_baseline_not_against_zero() -> None:
    """The condition is baseline-relative, and this is what that buys.

    Abstention misses relations, so a literal "zero missed relations" rule would
    hold the classifier to a standard the baseline itself fails — a threshold
    nobody derived. Matching the baseline's count is fine; exceeding it is not.
    """
    baseline = abstention_baseline(load_gold_set())
    matching = _perfect_report()
    for outcome in matching.outcomes[: baseline.count("missed_relation")]:
        outcome.verdict = "missed_relation"

    assert not any(
        failure.startswith("missed_relation") for failure in gate_failures(matching, baseline)
    )

    worse = _perfect_report()
    for outcome in worse.outcomes[: baseline.count("missed_relation") + 1]:
        outcome.verdict = "missed_relation"

    assert any(failure.startswith("missed_relation") for failure in gate_failures(worse, baseline))


def test_the_baseline_is_computed_from_the_corpus_rather_than_recorded() -> None:
    """Every count in the gate moves with the corpus, so none can go stale.

    A hard-coded floor would be a number chosen once and then quietly wrong, and
    the whole claim of this gate is that no number in it was chosen.
    """
    gold = load_gold_set()
    baseline = abstention_baseline(gold)

    assert baseline.total == len(gold)
    assert all(
        outcome.predicted is RelationType.NO_MEANINGFUL_RELATION for outcome in baseline.outcomes
    )
    assert abstention_baseline(gold[:5]).total == 5
