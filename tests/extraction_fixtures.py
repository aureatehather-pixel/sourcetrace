"""Shared realistic extraction fixtures.

The evidence text is the real M1 smoke-test material, not "hello world", so the
grounding quotes below are genuine substrings of it. Tests that mutate a quote
are therefore testing the real check, not an artefact of a toy string.
"""

from __future__ import annotations

import json
from typing import Any

EVIDENCE_CONTENT = (
    "Engineer at DevGuild told me serious projects mattered more than LeetCode. "
    "His argument was that grinding algorithm puzzles proves you can pass a filter, "
    "but shipping something real proves you can own an outcome. "
    "He said the people he actually hires are the ones who can walk him through a "
    "system they built and explain what broke. "
    "I noticed I could not describe my own project without reading my notes."
)

# Every quote below is a verbatim substring of EVIDENCE_CONTENT.
QUOTE_CLAIM = "serious projects mattered more than LeetCode"
QUOTE_HIRING = (
    "the people he actually hires are the ones who can walk him through a system they built"
)
QUOTE_OBSERVATION = "I could not describe my own project without reading my notes"
QUOTE_ENTITY = "Engineer at DevGuild"


def valid_payload() -> dict[str, Any]:
    """A well-formed extraction result for EVIDENCE_CONTENT.

    Note the epistemics: the engineer's advice is a `reported_claim` with
    `claimed_by` set, never an `observation`. The only `observation` is
    something the author states about themselves.
    """
    return {
        "items": [
            {
                "kind": "reported_claim",
                "data": {
                    "claim": "Serious projects matter more than LeetCode for getting hired.",
                    "claimed_by": "An engineer at DevGuild",
                    "stance": "asserted",
                },
                "grounding": {"quote": QUOTE_CLAIM},
            },
            {
                "kind": "reported_claim",
                "data": {
                    "claim": "He hires people who can walk him through a system they built.",
                    "claimed_by": "An engineer at DevGuild",
                    "stance": "asserted",
                },
                "grounding": {"quote": QUOTE_HIRING},
            },
            {
                "kind": "entity_mention",
                "data": {
                    "name": "DevGuild",
                    "entity_type": "organization",
                    "described_as": "Where the engineer was encountered",
                },
                "grounding": {"quote": QUOTE_ENTITY},
            },
            {
                "kind": "observation",
                "data": {
                    "observation": ("I could not describe my own project without reading my notes.")
                },
                "grounding": {"quote": QUOTE_OBSERVATION},
            },
        ]
    }


def valid_output_json() -> str:
    return json.dumps(valid_payload())


def payload_with_items(*items: dict[str, Any]) -> dict[str, Any]:
    return {"items": list(items)}


def claim_item(
    *,
    quote: str = QUOTE_CLAIM,
    claim: str = "Serious projects matter more than LeetCode.",
    stance: str = "asserted",
) -> dict[str, Any]:
    return {
        "kind": "reported_claim",
        "data": {"claim": claim, "claimed_by": "An engineer", "stance": stance},
        "grounding": {"quote": quote},
    }
