"""One note carrying all eight extraction kinds, for the packet tests.

Existing fixtures cover `entity_mention`, `reported_claim` and `observation`,
because those are the only kinds M3 and M4 can act on. The packet exists
precisely to reach the other five, so a fixture that stopped where M4 stops would
test around the change rather than through it.

Every `quote` below is a verbatim substring of `EVIDENCE_CONTENT`, so the real
M2 grounding check runs rather than being bypassed.
"""

from __future__ import annotations

from typing import Any

EVIDENCE_CONTENT = (
    "Notes after the DevGuild lab. Aisyah told me that judging weighted the demo far "
    "more than the code, and she recommended writing the test suite before the pitch "
    "deck. Watching the repository, Aisyah scaffolded the whole backend in one commit. "
    "We shipped a project called Fernvault, a receipt-matching service. "
    "The lab itself, Emberline DevGuild Lab 1, ran over a single weekend last spring. "
    "There is an opening to take ownership of the settlement module next cycle if I ask "
    "for it. I want future learning to start from a real project and its broken parts "
    "rather than from a tutorial. The team kept referring to the Postgres row-locking "
    "docs at https://example.invalid/locking throughout."
)

QUOTE_MENTION = "Aisyah told me that judging weighted the demo"
QUOTE_CLAIM = "judging weighted the demo far more than the code"
QUOTE_ADVICE = "writing the test suite before the pitch deck"
QUOTE_OBSERVATION = "Aisyah scaffolded the whole backend in one commit"
QUOTE_PROJECT = "We shipped a project called Fernvault, a receipt-matching service"
QUOTE_EVENT = "Emberline DevGuild Lab 1, ran over a single weekend last spring"
QUOTE_OPPORTUNITY = "an opening to take ownership of the settlement module next cycle"
QUOTE_IDEA = "future learning to start from a real project and its broken parts"
QUOTE_RESOURCE = "the Postgres row-locking docs at https://example.invalid/locking"


def all_kinds_payload() -> dict[str, Any]:
    """One extraction run containing exactly one item of each of the eight kinds.

    Ordered so `position` is stable and so the packet's ordering assertions have
    something to be wrong about: the packet kinds are deliberately not contiguous
    in this list.
    """
    return {
        "items": [
            {
                "kind": "entity_mention",
                "data": {
                    "name": "Aisyah",
                    "entity_type": "person",
                    "described_as": None,
                },
                "grounding": {"quote": QUOTE_MENTION},
            },
            {
                "kind": "reported_claim",
                "data": {
                    "claim": "Judging weighted the demo far more than the code.",
                    "claimed_by": "Aisyah",
                    "stance": "asserted",
                },
                "grounding": {"quote": QUOTE_CLAIM},
            },
            {
                "kind": "project",
                "data": {
                    "name": "Fernvault",
                    "description": "A receipt-matching service shipped at the lab.",
                },
                "grounding": {"quote": QUOTE_PROJECT},
            },
            {
                "kind": "observation",
                "data": {
                    "observation": "The whole backend was scaffolded in one commit.",
                    "subject": "Aisyah",
                },
                "grounding": {"quote": QUOTE_OBSERVATION},
            },
            {
                "kind": "event",
                "data": {
                    "name": "Emberline DevGuild Lab 1",
                    "occurred_at_text": "last spring",
                },
                "grounding": {"quote": QUOTE_EVENT},
            },
            {
                "kind": "resource",
                "data": {
                    "title": "Postgres row-locking docs",
                    "url": "https://example.invalid/locking",
                    "resource_type": "article",
                },
                "grounding": {"quote": QUOTE_RESOURCE},
            },
            {
                "kind": "opportunity",
                "data": {
                    "summary": "Take ownership of the settlement module next cycle.",
                },
                "grounding": {"quote": QUOTE_OPPORTUNITY},
            },
            {
                "kind": "idea",
                "data": {
                    "idea": (
                        "Future learning should start from a real project and its broken parts."
                    ),
                },
                "grounding": {"quote": QUOTE_IDEA},
            },
        ]
    }


SECOND_CONTENT = (
    "A week later Aisyah said the settlement module is still unclaimed, and that "
    "nobody has written the integration tests either."
)

SECOND_QUOTE_MENTION = "A week later Aisyah said"
SECOND_QUOTE_CLAIM = "the settlement module is still unclaimed"
SECOND_QUOTE_IDEA = "nobody has written the integration tests either"


def second_payload() -> dict[str, Any]:
    """A second note, so ordering across evidence items is testable."""
    return {
        "items": [
            {
                "kind": "entity_mention",
                "data": {"name": "Aisyah", "entity_type": "person", "described_as": None},
                "grounding": {"quote": SECOND_QUOTE_MENTION},
            },
            {
                "kind": "reported_claim",
                "data": {
                    "claim": "The settlement module is still unclaimed.",
                    "claimed_by": "Aisyah",
                    "stance": "asserted",
                },
                "grounding": {"quote": SECOND_QUOTE_CLAIM},
            },
            {
                "kind": "idea",
                "data": {"idea": "Nobody has written the integration tests."},
                "grounding": {"quote": SECOND_QUOTE_IDEA},
            },
        ]
    }


def reduced_payload() -> dict[str, Any]:
    """A smaller run over `EVIDENCE_CONTENT`, as a re-extraction at a new version.

    Its quotes are verbatim in the *same* evidence, so `verify_grounding` passes
    for real. A payload borrowed from another note would be rejected at ingest,
    which is the check working rather than a fixture problem.
    """
    return {
        "items": [
            {
                "kind": "reported_claim",
                "data": {
                    "claim": "Judging weighted the demo far more than the code.",
                    "claimed_by": "Aisyah",
                    "stance": "asserted",
                },
                "grounding": {"quote": QUOTE_CLAIM},
            },
            {
                "kind": "idea",
                "data": {
                    "idea": (
                        "Future learning should start from a real project and its broken parts."
                    ),
                },
                "grounding": {"quote": QUOTE_IDEA},
            },
        ]
    }
