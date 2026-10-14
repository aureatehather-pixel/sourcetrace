"""The evaluation corpus is itself an asset, so it is tested.

The runner that calls a real model lives in `evaluation/` and never runs under
pytest — it costs money. What runs here is everything that can rot silently: a
category quietly emptying out, a label that is both acceptable and forbidden, a
`qualifies` example with no direction, a duplicated id.

The scoring function is tested too. It decides which failures are which, and the
whole error-asymmetry argument rests on it being right.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation.gold_set import (
    REQUIRED_CATEGORIES,
    categories,
    has_causal_language,
    load_gold_set,
    score,
)

from intelligence_engine.intelligence.schemas import RelationType

GOLD = load_gold_set()


class TestCorpusIntegrity:
    def test_it_loads(self) -> None:
        assert len(GOLD) >= 20, "the corpus should hold at least 20 hand-labelled pairs"

    def test_every_required_category_is_represented(self) -> None:
        """A trap category emptying to zero is a corpus regression."""
        missing = REQUIRED_CATEGORIES - categories(GOLD)
        assert missing == set(), f"no examples left for: {sorted(missing)}"

    def test_no_category_is_a_single_example(self) -> None:
        counts = dict.fromkeys(REQUIRED_CATEGORIES, 0)
        for pair in GOLD:
            counts[pair.category] += 1
        assert all(count >= 2 for count in counts.values()), counts

    def test_ids_are_unique(self) -> None:
        assert len({pair.id for pair in GOLD}) == len(GOLD)

    def test_expected_is_always_acceptable(self) -> None:
        for pair in GOLD:
            assert pair.expected in pair.acceptable, pair.id

    def test_acceptable_and_forbidden_never_overlap(self) -> None:
        for pair in GOLD:
            assert not (pair.acceptable & pair.forbidden), pair.id

    def test_every_qualifies_example_states_a_direction(self) -> None:
        for pair in GOLD:
            if pair.expected is RelationType.QUALIFIES:
                assert pair.expected_direction in {"a_qualifies_b", "b_qualifies_a"}, pair.id

    def test_every_pair_carries_both_statements_and_context(self) -> None:
        for pair in GOLD:
            for side in (pair.a, pair.b):
                assert side.statement.strip()
                assert side.quote.strip()
                assert side.context.strip(), (
                    f"{pair.id}: context is what defuses false contradictions"
                )

    def test_every_pair_explains_its_own_label(self) -> None:
        """A corpus nobody can audit is not evidence."""
        for pair in GOLD:
            assert len(pair.note) > 20, pair.id


class TestTrapsForbidTheTrap:
    def test_false_contradiction_traps_forbid_contradicts(self) -> None:
        traps = [pair for pair in GOLD if pair.category == "false_contradiction_trap"]
        assert traps
        for pair in traps:
            assert RelationType.CONTRADICTS in pair.forbidden, pair.id

    def test_trivial_similarity_traps_accept_declining(self) -> None:
        traps = [pair for pair in GOLD if pair.category == "trivial_similarity_trap"]
        assert traps
        for pair in traps:
            assert RelationType.NO_MEANINGFUL_RELATION in pair.acceptable, pair.id

    def test_no_relation_examples_forbid_every_meaningful_label(self) -> None:
        unrelated = [pair for pair in GOLD if pair.category == "no_relation"]
        assert unrelated
        for pair in unrelated:
            assert pair.forbidden == {
                RelationType.REINFORCES,
                RelationType.QUALIFIES,
                RelationType.CONTRADICTS,
            }, pair.id

    def test_the_flagship_qualification_is_present(self) -> None:
        """The pair the milestone exists for. If it leaves the corpus, say so loudly."""
        flagship = next(pair for pair in GOLD if pair.id == "gold_001")
        assert flagship.expected is RelationType.QUALIFIES
        assert flagship.expected_direction == "b_qualifies_a"
        assert RelationType.CONTRADICTS in flagship.forbidden


class TestScoring:
    @pytest.fixture
    def trap(self) -> object:
        return next(pair for pair in GOLD if pair.category == "false_contradiction_trap")

    def test_a_forbidden_contradiction_is_its_own_category(self, trap: object) -> None:
        assert score(trap, RelationType.CONTRADICTS, None) == "false_contradiction"  # type: ignore[arg-type]

    def test_an_acceptable_label_is_correct(self) -> None:
        pair = next(item for item in GOLD if item.id == "gold_001")
        assert score(pair, RelationType.QUALIFIES, "b_qualifies_a") == "correct"

    def test_a_backwards_direction_is_reported_separately(self) -> None:
        pair = next(item for item in GOLD if item.id == "gold_001")
        assert score(pair, RelationType.QUALIFIES, "a_qualifies_b") == "wrong_direction"

    def test_declining_a_real_relation_is_a_miss_not_a_false_relation(self) -> None:
        pair = next(item for item in GOLD if item.id == "gold_001")
        assert score(pair, RelationType.NO_MEANINGFUL_RELATION, None) == "missed_relation"

    def test_asserting_a_relation_where_declining_was_right_is_a_false_relation(self) -> None:
        pair = next(item for item in GOLD if item.category == "no_relation")
        assert score(pair, RelationType.REINFORCES, None) == "false_relation"

    def test_the_wrong_meaningful_label_is_its_own_category(self) -> None:
        pair = next(item for item in GOLD if item.id == "gold_001")
        assert score(pair, RelationType.REINFORCES, None) == "wrong_meaningful"

    def test_a_missing_prediction_is_malformed(self) -> None:
        pair = GOLD[0]
        assert score(pair, None, None) == "malformed"


class TestCausalDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "The first caused the second.",
            "Leaving the team led to the knowledge loss.",
            "The delay resulted in a missed deadline.",
            "This happened because of the migration.",
            "Therefore the second statement follows.",
        ],
    )
    def test_causal_phrasing_is_detected(self, text: str) -> None:
        assert has_causal_language(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "The second adds a time-phase condition to the first.",
            "Both describe the same six-week period from different angles.",
            None,
            "",
        ],
    )
    def test_non_causal_phrasing_is_not_flagged(self, text: str | None) -> None:
        assert has_causal_language(text) is False


def _valid_gold_entry(pair_id: str = "test_pair") -> dict[str, object]:
    """Build the smallest complete corpus entry accepted by the loader."""
    statement = {
        "statement": "One independently useful statement.",
        "stance": "asserted",
        "claimed_by": "Test author",
        "quote": "one independently useful statement",
        "context": "Unit test fixture",
    }
    return {
        "id": pair_id,
        "category": "no_relation",
        "note": "A deliberately minimal valid gold-set entry for loader tests.",
        "expected": "qualifies",
        "expected_direction": "b_qualifies_a",
        "acceptable": ["qualifies"],
        "forbidden": ["contradicts"],
        "a": statement,
        "b": statement.copy(),
    }


def _write_gold_set(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    """Write a temporary corpus and return its path."""
    path = tmp_path / "gold_set.json"
    path.write_text(json.dumps({"pairs": entries}), encoding="utf-8")
    return path


class TestLoaderRejectsMalformedCorpora:
    def test_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        entries = [_valid_gold_entry("duplicate"), _valid_gold_entry("duplicate")]

        with pytest.raises(ValueError, match="Duplicate gold pair ids"):
            load_gold_set(_write_gold_set(tmp_path, entries))

    def test_expected_label_outside_acceptable_is_rejected(self, tmp_path: Path) -> None:
        entry = _valid_gold_entry()
        entry["acceptable"] = ["reinforces"]

        with pytest.raises(
            ValueError,
            match="expected label is not in its own acceptable set",
        ):
            load_gold_set(_write_gold_set(tmp_path, [entry]))

    def test_acceptable_and_forbidden_overlap_is_rejected(self, tmp_path: Path) -> None:
        entry = _valid_gold_entry()
        # The label has to be one already in `acceptable`, or there is no overlap
        # to detect and the test proves nothing.
        entry["forbidden"] = ["qualifies"]

        with pytest.raises(ValueError, match="both acceptable and forbidden"):
            load_gold_set(_write_gold_set(tmp_path, [entry]))

    def test_qualifies_without_direction_is_rejected(self, tmp_path: Path) -> None:
        entry = _valid_gold_entry()
        del entry["expected_direction"]

        with pytest.raises(ValueError, match="states no direction"):
            load_gold_set(_write_gold_set(tmp_path, [entry]))

    def test_valid_entry_loads_cleanly(self, tmp_path: Path) -> None:
        pairs = load_gold_set(_write_gold_set(tmp_path, [_valid_gold_entry()]))

        assert len(pairs) == 1


class TestLoaderCollectsEveryProblem:
    """The loader collects every problem across the corpus before raising.

    A hand-grown corpus is repaired a handful of entries at a time; raising on
    the first problem found makes every fix a separate round-trip.
    """

    def test_it_reports_every_problem_in_one_error(self, tmp_path: Path) -> None:
        duplicate = _valid_gold_entry("shared_id")
        bad_expected = _valid_gold_entry("expected_not_acceptable")
        bad_expected["acceptable"] = ["reinforces"]
        no_direction = _valid_gold_entry("qualifies_no_direction")
        del no_direction["expected_direction"]

        entries = [duplicate, duplicate, bad_expected, no_direction]

        with pytest.raises(ValueError) as exc_info:
            load_gold_set(_write_gold_set(tmp_path, entries))

        message: str = exc_info.value.args[0]
        assert "Duplicate gold pair ids" in message
        assert "expected label is not in its own acceptable set" in message
        assert "states no direction" in message
        assert "shared_id" in message
        assert "expected_not_acceptable" in message
        assert "qualifies_no_direction" in message

    def test_a_repeated_id_is_reported_once_however_many_copies(self, tmp_path: Path) -> None:
        """Anchored at the first occurrence, not re-armed by later copies.

        Tracking the anchor in a field that also carried an "already reported"
        sentinel made the fourth copy of an id report the same problem twice.
        """
        entries = [_valid_gold_entry("repeated") for _ in range(5)]

        with pytest.raises(ValueError) as exc_info:
            load_gold_set(_write_gold_set(tmp_path, entries))

        message: str = exc_info.value.args[0]
        assert message.count("Duplicate gold pair ids") == 1, message
