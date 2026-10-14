"""The JSON Schema sent to the provider.

Two of these pins exist because the live Messages API returned 400s until the
normalisations were applied (M2 learnings 1 and 2). They are pinned so nobody
"tidies" them back in and rediscovers the same failure with credentials.
"""

from __future__ import annotations

from typing import Any

import jsonschema

from intelligence_engine.intelligence.schemas import (
    UNSUPPORTED_SCHEMA_KEYWORDS,
    RelationOutput,
    wire_schema,
)
from tests.intelligence_fixtures import classifier_response, no_relation, qualifies


def _walk(node: Any) -> list[Any]:
    found = [node]
    if isinstance(node, dict):
        for value in node.values():
            found.extend(_walk(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk(item))
    return found


def _keys(schema: dict[str, Any]) -> set[str]:
    return {key for node in _walk(schema) if isinstance(node, dict) for key in node}


class TestApiCompatibility:
    def test_no_keyword_the_api_rejects_survives(self) -> None:
        assert _keys(wire_schema()) & UNSUPPORTED_SCHEMA_KEYWORDS == set()

    def test_titles_are_dropped(self) -> None:
        """Noise in a constraint, and paid for as input tokens on every call."""
        assert "title" not in _keys(wire_schema())

    def test_the_schema_is_a_flat_object_not_a_discriminated_union(self) -> None:
        """Materially cheaper than M2's eight-variant schema, and one less thing to normalise."""
        assert "oneOf" not in _keys(wire_schema())
        assert "discriminator" not in _keys(wire_schema())

    def test_it_is_valid_json_schema(self) -> None:
        jsonschema.Draft202012Validator.check_schema(wire_schema())


class TestSchemaCost:
    def test_class_docstrings_do_not_reach_the_wire(self) -> None:
        """Engineering prose is billed as input on every call.

        Field descriptions written for the model survive; module documentation
        about version bumps and rejected designs does not.
        """
        schema = wire_schema()
        assert "description" not in schema
        assert all("description" not in definition for definition in schema["$defs"].values())

    def test_field_descriptions_written_for_the_model_survive(self) -> None:
        properties = wire_schema()["$defs"]["RelationResult"]["properties"]
        assert "Required for `qualifies`" in properties["direction"]["description"]
        assert "Derived reasoning, not a quote" in properties["explanation"]["description"]

    def test_the_schema_stays_small(self) -> None:
        """A guard, not a target. It was 2,483 characters before the strip."""
        assert len(str(wire_schema())) < 1_400


class TestSchemaAndValidationAgree:
    def test_a_response_valid_against_the_wire_schema_also_parses(self) -> None:
        payload = {"results": [qualifies("pair_01"), no_relation("pair_02")]}
        jsonschema.validate(payload, wire_schema())
        assert len(RelationOutput.model_validate(payload).results) == 2

    def test_the_relation_vocabulary_reaches_the_wire(self) -> None:
        rendered = str(wire_schema())
        for value in ("reinforces", "qualifies", "contradicts", "no_meaningful_relation"):
            assert value in rendered

    def test_no_causal_relation_type_is_offered_to_the_model(self) -> None:
        """The vocabulary the model may choose from contains nothing causal."""
        offered = wire_schema()["$defs"]["RelationType"]["enum"]
        assert offered == ["reinforces", "qualifies", "contradicts", "no_meaningful_relation"]
        for forbidden in ("causes", "caused_by", "leads_to", "explains"):
            assert forbidden not in offered

    def test_no_confidence_field_is_offered_to_the_model(self) -> None:
        """Nothing calibrates a self-rating, so the model is never asked for one."""
        rendered = str(wire_schema()).lower()
        assert "confidence" not in rendered
        assert "probability" not in rendered
        assert "score" not in rendered


class TestBoundsHoldAfterNormalisation:
    def test_the_batch_bound_is_enforced_on_arrival(self) -> None:
        """`maxItems` cannot be sent, so the guarantee moves to validation."""
        oversized = classifier_response(*[no_relation(f"pair_{n:02d}") for n in range(1, 30)])
        assert "maxItems" not in str(wire_schema())
        try:
            RelationOutput.model_validate_json(oversized)
        except Exception as exc:
            assert "too_long" in str(exc) or "at most" in str(exc)
        else:
            raise AssertionError("an over-long batch response should be rejected")
