"""The classifier trust boundary.

Model output is untrusted input. These tests are the negative cases: each one
proves a specific malformed response is *rejected*, not coerced, dropped or
quietly accepted.
"""

from __future__ import annotations

import json

import pytest

from intelligence_engine.intelligence.errors import InvalidRelationOutput
from intelligence_engine.intelligence.schemas import (
    MAX_EXPLANATION_LENGTH,
    RelationDirection,
    RelationType,
)
from intelligence_engine.intelligence.validation import bind_results, parse_output
from tests.intelligence_fixtures import (
    advice_item,
    classifier_response,
    contradicts,
    make_pair,
    mentoring_item,
    no_relation,
    planner_item,
    qualifies,
    reading_item,
    reinforces,
)


def one_batch() -> dict[str, object]:
    return {"pair_01": make_pair(advice_item(), reading_item())}


def two_batch() -> dict[str, object]:
    return {
        "pair_01": make_pair(advice_item(), reading_item()),
        "pair_02": make_pair(planner_item(), mentoring_item()),
    }


class TestStructuralParsing:
    def test_a_well_formed_response_parses(self) -> None:
        output = parse_output(classifier_response(qualifies("pair_01")))
        assert output.results[0].relation_type is RelationType.QUALIFIES

    def test_non_json_is_rejected(self) -> None:
        with pytest.raises(InvalidRelationOutput):
            parse_output("Here are the relationships I found:")

    def test_markdown_fenced_json_is_rejected(self) -> None:
        with pytest.raises(InvalidRelationOutput):
            parse_output('```json\n{"results": []}\n```')

    def test_an_unknown_relation_type_is_rejected(self) -> None:
        raw = json.dumps(
            {"results": [{"pair_01": "x", "relation_type": "causes", "explanation": "a" * 30}]}
        )
        with pytest.raises(InvalidRelationOutput):
            parse_output(raw)

    def test_a_causal_relation_type_cannot_be_smuggled_in(self) -> None:
        """The vocabulary is bounded; `causes` is not in it and never will be by accident."""
        raw = classifier_response(
            {"pair_id": "pair_01", "relation_type": "causes", "explanation": "a" * 30}
        )
        with pytest.raises(InvalidRelationOutput):
            parse_output(raw)

    def test_extra_keys_are_rejected_rather_than_dropped(self) -> None:
        raw = classifier_response(
            {
                "pair_id": "pair_01",
                "relation_type": "reinforces",
                "explanation": "a" * 30,
                "confidence": 0.91,
            }
        )
        with pytest.raises(InvalidRelationOutput):
            parse_output(raw)

    def test_an_over_long_explanation_is_rejected(self) -> None:
        raw = classifier_response(
            {
                "pair_id": "pair_01",
                "relation_type": "reinforces",
                "explanation": "a" * (MAX_EXPLANATION_LENGTH + 1),
            }
        )
        with pytest.raises(InvalidRelationOutput):
            parse_output(raw)

    def test_the_error_summary_does_not_echo_the_payload(self) -> None:
        """Claim text must not leak into logs or error columns."""
        private_text = "a thing somebody said in confidence"
        raw = classifier_response(
            {"pair_id": "pair_01", "relation_type": "invented", "explanation": private_text}
        )
        with pytest.raises(InvalidRelationOutput) as caught:
            parse_output(raw)
        assert private_text not in str(caught.value)


class TestBatchIdentity:
    def test_results_bind_to_their_supplied_pairs(self) -> None:
        batch = two_batch()
        bound = bind_results(
            parse_output(classifier_response(qualifies("pair_01"), no_relation("pair_02"))),
            batch,  # type: ignore[arg-type]
        )
        assert [result.relation_type for _pair, result, _dir in bound] == [
            RelationType.QUALIFIES,
            RelationType.NO_MEANINGFUL_RELATION,
        ]

    def test_results_are_returned_in_supplied_order_not_model_order(self) -> None:
        """Persistence must not depend on how the model chose to order its answer."""
        batch = two_batch()
        bound = bind_results(
            parse_output(classifier_response(no_relation("pair_02"), qualifies("pair_01"))),
            batch,  # type: ignore[arg-type]
        )
        assert [result.pair_id for _pair, result, _dir in bound] == ["pair_01", "pair_02"]

    def test_an_unknown_pair_id_is_rejected(self) -> None:
        with pytest.raises(InvalidRelationOutput, match="not supplied"):
            bind_results(
                parse_output(classifier_response(qualifies("pair_99"))),
                one_batch(),  # type: ignore[arg-type]
            )

    def test_a_duplicate_pair_id_is_rejected(self) -> None:
        with pytest.raises(InvalidRelationOutput, match="more than once"):
            bind_results(
                parse_output(classifier_response(qualifies("pair_01"), reinforces("pair_01"))),
                one_batch(),  # type: ignore[arg-type]
            )

    def test_an_incomplete_batch_is_rejected(self) -> None:
        """Silently dropping a pair would mean paying for it again next time."""
        with pytest.raises(InvalidRelationOutput, match="not answered"):
            bind_results(
                parse_output(classifier_response(qualifies("pair_01"))),
                two_batch(),  # type: ignore[arg-type]
            )

    def test_an_empty_response_for_a_non_empty_batch_is_rejected(self) -> None:
        with pytest.raises(InvalidRelationOutput, match="not answered"):
            bind_results(
                parse_output(classifier_response()),
                one_batch(),  # type: ignore[arg-type]
            )


class TestDirectionSemantics:
    def test_qualifies_maps_a_to_the_canonical_left(self) -> None:
        bound = bind_results(
            parse_output(classifier_response(qualifies("pair_01", direction="a_qualifies_b"))),
            one_batch(),  # type: ignore[arg-type]
        )
        assert bound[0][2] is RelationDirection.LEFT_TO_RIGHT

    def test_qualifies_maps_b_to_the_canonical_right(self) -> None:
        bound = bind_results(
            parse_output(classifier_response(qualifies("pair_01", direction="b_qualifies_a"))),
            one_batch(),  # type: ignore[arg-type]
        )
        assert bound[0][2] is RelationDirection.RIGHT_TO_LEFT

    def test_symmetric_relations_are_stored_symmetric(self) -> None:
        bound = bind_results(
            parse_output(classifier_response(reinforces("pair_01"))),
            one_batch(),  # type: ignore[arg-type]
        )
        assert bound[0][2] is RelationDirection.SYMMETRIC

    def test_qualifies_without_a_direction_is_rejected(self) -> None:
        raw = classifier_response(
            {"pair_id": "pair_01", "relation_type": "qualifies", "explanation": "a" * 30}
        )
        with pytest.raises(InvalidRelationOutput, match="states no direction"):
            bind_results(parse_output(raw), one_batch())  # type: ignore[arg-type]

    def test_a_direction_on_a_symmetric_relation_is_rejected(self) -> None:
        raw = classifier_response(
            {
                "pair_id": "pair_01",
                "relation_type": "contradicts",
                "direction": "a_qualifies_b",
                "explanation": "a" * 30,
            }
        )
        with pytest.raises(InvalidRelationOutput, match="symmetric"):
            bind_results(parse_output(raw), one_batch())  # type: ignore[arg-type]


class TestExplanationContract:
    def test_a_meaningful_relation_without_an_explanation_is_rejected(self) -> None:
        raw = classifier_response({"pair_id": "pair_01", "relation_type": "reinforces"})
        with pytest.raises(InvalidRelationOutput, match="no explanation"):
            bind_results(parse_output(raw), one_batch())  # type: ignore[arg-type]

    def test_no_relation_with_an_explanation_is_rejected(self) -> None:
        """An explanation attached to "no relation" is unlabelled speculation."""
        raw = classifier_response(
            {
                "pair_id": "pair_01",
                "relation_type": "no_meaningful_relation",
                "explanation": "They are both loosely about learning to program.",
            }
        )
        with pytest.raises(InvalidRelationOutput, match="supplies an explanation"):
            bind_results(parse_output(raw), one_batch())  # type: ignore[arg-type]

    def test_no_relation_without_an_explanation_is_accepted(self) -> None:
        bound = bind_results(
            parse_output(classifier_response(no_relation("pair_01"))),
            one_batch(),  # type: ignore[arg-type]
        )
        assert bound[0][1].explanation is None

    def test_contradiction_keeps_its_explanation(self) -> None:
        bound = bind_results(
            parse_output(classifier_response(contradicts("pair_01"))),
            one_batch(),  # type: ignore[arg-type]
        )
        assert bound[0][1].relation_type is RelationType.CONTRADICTS
        assert bound[0][1].explanation
