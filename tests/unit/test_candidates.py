"""Candidate generation: bounded, deterministic, and honest about its blind spot."""

from __future__ import annotations

import pytest

from intelligence_engine.intelligence.candidates import (
    batched,
    canonical_pair,
    content_words,
    generate_candidates,
    is_stance_complementary,
    lexical_overlap,
)
from tests.intelligence_fixtures import (
    EVIDENCE_A_ID,
    EVIDENCE_B_ID,
    advice_item,
    make_item,
    mentoring_item,
    planner_item,
    reading_item,
)


class TestTheoreticalSpace:
    def test_theoretical_count_is_the_full_pair_space(self) -> None:
        items = [make_item(text=f"claim {index}", position=index) for index in range(10)]
        assert generate_candidates(items, max_pairs=100).theoretical_pairs == 45

    def test_all_pairs_classification_is_never_what_happens(self) -> None:
        """100 items is 4,950 pairs. Narrowing plus the cap must bound it hard."""
        items = [
            make_item(
                text=f"claim {index}",
                evidence_id=EVIDENCE_A_ID if index % 2 else EVIDENCE_B_ID,
                position=index,
            )
            for index in range(100)
        ]
        selection = generate_candidates(items, max_pairs=60)

        assert selection.theoretical_pairs == 4950
        assert len(selection.pairs) == 60
        assert selection.capped is True

    def test_empty_and_single_item_produce_nothing(self) -> None:
        assert generate_candidates([], max_pairs=10).pairs == []
        assert generate_candidates([advice_item()], max_pairs=10).pairs == []


class TestCrossEvidenceFilter:
    def test_same_evidence_pairs_are_excluded(self) -> None:
        """Two claims from one note are not cross-evidence intelligence."""
        selection = generate_candidates([advice_item(), planner_item()], max_pairs=10)

        assert selection.theoretical_pairs == 1
        assert selection.pairs == []

    def test_cross_evidence_pairs_survive(self) -> None:
        selection = generate_candidates([advice_item(), reading_item()], max_pairs=10)

        assert len(selection.pairs) == 1
        pair = selection.pairs[0]
        assert {pair.left.evidence_id, pair.right.evidence_id} == {EVIDENCE_A_ID, EVIDENCE_B_ID}

    def test_reduction_is_real_on_mixed_material(self) -> None:
        """Two notes, four claims each: 28 theoretical pairs narrow to 16."""
        items = [
            make_item(
                text=f"claim {index}",
                evidence_id=EVIDENCE_A_ID if index < 4 else EVIDENCE_B_ID,
                position=index,
            )
            for index in range(8)
        ]
        selection = generate_candidates(items, max_pairs=100)

        assert selection.theoretical_pairs == 28
        assert selection.filtered_pairs == 16


class TestCanonicalOrdering:
    def test_pairs_are_ordered_by_item_id(self) -> None:
        selection = generate_candidates([reading_item(), advice_item()], max_pairs=10)
        pair = selection.pairs[0]
        assert pair.left.item_id < pair.right.item_id

    def test_order_of_input_does_not_change_the_pair(self) -> None:
        left, right = advice_item(), reading_item()
        forwards = generate_candidates([left, right], max_pairs=10).pairs[0]
        backwards = generate_candidates([right, left], max_pairs=10).pairs[0]
        assert forwards.key == backwards.key

    def test_canonical_pair_is_stable(self) -> None:
        left, right = advice_item(), reading_item()
        assert canonical_pair(left, right) == canonical_pair(right, left)


class TestLexicalOverlap:
    def test_shared_vocabulary_scores_high(self) -> None:
        first = make_item(
            text="the migration script runs in four minutes", quote="migration script"
        )
        second = make_item(
            text="the migration script runs in five minutes", quote="migration script"
        )
        assert lexical_overlap(first, second) > 0.5

    def test_the_flagship_pair_scores_zero(self) -> None:
        """The measurement that decided the design.

        The single most valuable relation in the corpus shares no content word at
        all, so lexical overlap can only ever be a tiebreak. Used as a filter it
        would discard exactly the pair M4 exists to find.
        """
        assert lexical_overlap(advice_item(), reading_item()) == 0.0

    def test_stopwords_do_not_create_overlap(self) -> None:
        first = make_item(text="the and of in on at", quote="the and of")
        second = make_item(text="but if then than that", quote="but if then")
        assert content_words(first.text) == frozenset()
        assert lexical_overlap(first, second) == 0.0


class TestStanceSignal:
    def test_advice_plus_experience_is_complementary(self) -> None:
        assert is_stance_complementary(advice_item(), reading_item()) is True

    def test_two_assertions_are_not(self) -> None:
        assert is_stance_complementary(planner_item(), mentoring_item()) is False

    def test_two_recommendations_are_not(self) -> None:
        first = advice_item()
        second = reading_item(stance="recommended")
        assert is_stance_complementary(first, second) is False

    def test_complementary_pairs_rank_above_higher_overlap_ones(self) -> None:
        """The structural signal outranks vocabulary, on purpose.

        The advice/experience pair shares no content word; the other pair shares
        almost all of them. A vocabulary-first ranking would put the wrong one
        first, and at a binding cap that is the difference between finding the
        relation and never looking at it.
        """
        advice, reading = advice_item(), reading_item()
        overlapping_a = make_item(
            text="rebuilding a Postgres query planner taught him more",
            evidence_id=EVIDENCE_A_ID,
            position=1,
        )
        overlapping_b = make_item(
            text="rebuilding a Postgres query planner taught her more",
            evidence_id=EVIDENCE_B_ID,
            position=1,
        )
        assert lexical_overlap(overlapping_a, overlapping_b) > 0.5
        assert lexical_overlap(advice, reading) == 0.0

        selection = generate_candidates(
            [advice, reading, overlapping_a, overlapping_b], max_pairs=10
        )
        ranking = [pair.key for pair in selection.pairs]
        complementary = canonical_pair(advice, reading)
        overlapping = canonical_pair(overlapping_a, overlapping_b)

        assert ranking.index((complementary[0].item_id, complementary[1].item_id)) < ranking.index(
            (overlapping[0].item_id, overlapping[1].item_id)
        )


class TestDeterminism:
    def test_the_same_input_produces_the_same_order(self) -> None:
        """Reuse and the evaluation both depend on this."""
        items = [advice_item(), reading_item(), planner_item(), mentoring_item()]
        first = [pair.key for pair in generate_candidates(items, max_pairs=10).pairs]
        second = [pair.key for pair in generate_candidates(items, max_pairs=10).pairs]
        assert first == second

    def test_the_cap_takes_the_top_of_a_stable_order(self) -> None:
        items = [
            make_item(
                text=f"claim {index}",
                evidence_id=EVIDENCE_A_ID if index % 2 else EVIDENCE_B_ID,
                position=index,
            )
            for index in range(10)
        ]
        full = generate_candidates(items, max_pairs=100).pairs
        capped = generate_candidates(items, max_pairs=5).pairs
        assert [pair.key for pair in capped] == [pair.key for pair in full[:5]]


class TestBatching:
    def test_batches_are_bounded(self) -> None:
        items = [
            make_item(
                text=f"claim {index}",
                evidence_id=EVIDENCE_A_ID if index % 2 else EVIDENCE_B_ID,
                position=index,
            )
            for index in range(10)
        ]
        pairs = generate_candidates(items, max_pairs=25).pairs
        batches = batched(pairs, size=8)

        assert len(batches) == 4
        assert all(len(batch) <= 8 for batch in batches)
        assert sum(len(batch) for batch in batches) == 25

    def test_no_empty_trailing_batch(self) -> None:
        items = [advice_item(), reading_item()]
        assert len(batched(generate_candidates(items, max_pairs=10).pairs, size=8)) == 1

    def test_batch_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            batched([], size=0)

    def test_negative_cap_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not be negative"):
            generate_candidates([], max_pairs=-1)
