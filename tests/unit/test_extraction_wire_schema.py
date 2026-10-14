"""The schema sent to the provider must match the schema we validate against.

If these drift, the model is constrained to produce one shape while the
application accepts another — every run would fail validation, or worse, a
constraint would silently permit something we cannot store.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest

from intelligence_engine.extraction.errors import ExtractionOverflow
from intelligence_engine.extraction.schemas import (
    MAX_ITEMS_PER_RUN,
    UNSUPPORTED_SCHEMA_KEYWORDS,
    ItemKind,
    wire_schema,
)
from intelligence_engine.extraction.validation import parse_output
from tests.extraction_fixtures import QUOTE_CLAIM, claim_item, payload_with_items, valid_payload


def _walk(node: Any) -> list[Any]:
    found = [node]
    if isinstance(node, dict):
        for value in node.values():
            found.extend(_walk(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_walk(value))
    return found


def test_schema_is_valid_json_schema() -> None:
    jsonschema.Draft202012Validator.check_schema(wire_schema())


def test_discriminator_is_stripped() -> None:
    """An OpenAPI extension, not JSON Schema. Strict validators reject it."""
    assert "discriminator" not in json.dumps(wire_schema())


def test_titles_are_stripped() -> None:
    """Noise in a constraint, and paid for on every request as input tokens."""
    assert '"title"' not in json.dumps(wire_schema())


def test_every_object_forbids_additional_properties() -> None:
    """Without this the model can attach fields we would then have to ignore."""
    for node in _walk(wire_schema()):
        if isinstance(node, dict) and node.get("type") == "object":
            assert node.get("additionalProperties") is False, node


def test_all_item_kinds_are_offered_to_the_model() -> None:
    schema = wire_schema()
    variants = schema["properties"]["items"]["items"]["anyOf"]
    definitions = schema["$defs"]

    offered = {
        definitions[ref["$ref"].removeprefix("#/$defs/")]["properties"]["kind"]["const"]
        for ref in variants
    }
    assert offered == {kind.value for kind in ItemKind}


def test_wire_schema_avoids_keywords_the_provider_rejects() -> None:
    """Learned from a live 400, pinned so it cannot be reintroduced.

    The Messages API structured-output subset rejects `maxItems` on arrays and
    `oneOf` entirely. Both once appeared here and cost a failed live run.
    """
    serialised = json.dumps(wire_schema())
    for keyword in UNSUPPORTED_SCHEMA_KEYWORDS:
        assert f'"{keyword}"' not in serialised, keyword


def test_the_union_is_expressed_as_anyof_not_oneof() -> None:
    """`oneOf` is rejected by the API; `anyOf` is equivalent here.

    Every variant pins a distinct `kind` const, so at most one can match and
    "at least one" and "exactly one" mean the same thing.
    """
    item_schema = wire_schema()["properties"]["items"]["items"]
    assert "anyOf" in item_schema
    assert "oneOf" not in item_schema


def test_item_count_bound_is_enforced_by_validation_not_the_wire_schema() -> None:
    """The bound survives the schema keyword being unsupported.

    `maxItems` cannot be sent, so an over-long response is rejected on arrival
    instead of never being produced. The guarantee is identical; only where it
    is enforced changed.

    The model is therefore never told the bound exists, which is why exceeding
    it cannot be prevented — only detected. That is precisely what makes it a
    subdivision trigger rather than a constraint: the response is rejected as
    `ExtractionOverflow`, and the interval is asked again in smaller pieces.
    """
    assert "maxItems" not in wire_schema()["properties"]["items"]

    too_many = payload_with_items(*[claim_item() for _ in range(MAX_ITEMS_PER_RUN + 1)])
    with pytest.raises(ExtractionOverflow):
        parse_output(json.dumps(too_many))


def test_a_payload_valid_under_the_wire_schema_also_passes_pydantic() -> None:
    """The two halves of the contract agree on what is acceptable."""
    payload = valid_payload()

    jsonschema.validate(payload, wire_schema())
    assert len(parse_output(json.dumps(payload)).items) == 4


@pytest.mark.parametrize(
    "payload",
    [
        payload_with_items(
            {
                "kind": "personality_assessment",
                "data": {"summary": "x"},
                "grounding": {"quote": QUOTE_CLAIM},
            }
        ),
        payload_with_items(claim_item(stance="certain")),
        {"items": [], "commentary": "extra"},
    ],
)
def test_payloads_pydantic_rejects_are_also_rejected_by_the_wire_schema(
    payload: dict[str, Any],
) -> None:
    """Neither half is a weaker gate than the other."""
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, wire_schema())
