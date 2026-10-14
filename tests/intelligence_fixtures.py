"""Shared M4 fixtures: relatable items and classifier responses.

Built on the same Zilan material as `entity_fixtures`, because the relation M4
exists to find is in that material and a toy fixture would test around it rather
than through it.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from typing import Any

from intelligence_engine.entities.schemas import ResolutionRole
from intelligence_engine.extraction.schemas import ItemKind
from intelligence_engine.intelligence.candidates import CandidatePair, RelatableItem

EVIDENCE_A_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
EVIDENCE_B_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")

_BASE = dt.datetime(2026, 2, 11, 12, 0, tzinfo=dt.UTC)


def make_item(
    *,
    item_id: uuid.UUID | None = None,
    text: str,
    stance: str | None = "asserted",
    claimed_by: str | None = "Zilan",
    quote: str | None = None,
    evidence_id: uuid.UUID = EVIDENCE_A_ID,
    source_label: str = "Kestrel Python meetup",
    position: int = 0,
    days: int = 0,
    kind: ItemKind = ItemKind.REPORTED_CLAIM,
    subject: str | None = None,
) -> RelatableItem:
    """One relatable item. Ids default to fresh uuid4 so pairs are distinct.

    The `claimed_by` / `subject` invariant is asserted here rather than assumed.
    A fixture that can build an item with both set would let a test pass against
    a state the database and the service layer both forbid, which is worse than
    no fixture at all.
    """
    if kind is ItemKind.OBSERVATION:
        assert claimed_by is None, "an observation has a subject, never a speaker"
        assert stance is None, "an observation is not offered as a stance"
    else:
        assert subject is None, f"{kind.value} carries claimed_by, not subject"

    return RelatableItem(
        item_id=item_id or uuid.uuid4(),
        kind=kind,
        # Mirrors the DB role/kind CHECK, so a fixture cannot render a connection
        # the database would refuse to store.
        role=(ResolutionRole.ABOUT if kind is ItemKind.OBSERVATION else ResolutionRole.CLAIMED_BY),
        text=text,
        stance=stance,
        claimed_by=claimed_by,
        subject=subject,
        grounding_quote=quote or text.lower(),
        extraction_run_id=uuid.uuid4(),
        evidence_id=evidence_id,
        resolution_id=uuid.uuid4(),
        source_label=source_label,
        captured_at=_BASE + dt.timedelta(days=days),
        evidence_created_at=_BASE + dt.timedelta(days=days),
        position=position,
    )


def make_observation(
    *,
    text: str,
    subject: str | None = "Zilan",
    **kwargs: object,
) -> RelatableItem:
    """An observation: a subject it is about, and no speaker at all."""
    return make_item(
        text=text,
        kind=ItemKind.OBSERVATION,
        stance=None,
        claimed_by=None,
        subject=subject,
        **kwargs,  # type: ignore[arg-type]
    )


ADVICE = "The fastest way to get good is to reverse-engineer a serious project."
READING = "The first six weeks were mostly reading, not writing."
PLANNER = "He rebuilt a small Postgres query planner over three months."
MENTORING = "He is now mentoring two juniors at Glyde."
PROFILE = "Her public profile listed experience as a club secretary."


def _build(defaults: dict[str, Any], overrides: dict[str, Any]) -> RelatableItem:
    """Overrides replace defaults rather than colliding with them."""
    return make_item(**{**defaults, **overrides})


def advice_item(**overrides: Any) -> RelatableItem:
    return _build(
        {
            "text": ADVICE,
            "stance": "recommended",
            "quote": "the fastest way to get good is to reverse-engineer a serious project",
            "evidence_id": EVIDENCE_A_ID,
            "source_label": "Kestrel Python meetup",
            "position": 0,
        },
        overrides,
    )


def reading_item(**overrides: Any) -> RelatableItem:
    return _build(
        {
            "text": READING,
            "stance": "asserted",
            "quote": "the first six weeks were mostly reading, not writing",
            "evidence_id": EVIDENCE_B_ID,
            "source_label": "Coffee, Brookside",
            "position": 0,
            "days": 38,
        },
        overrides,
    )


def planner_item(**overrides: Any) -> RelatableItem:
    return _build(
        {
            "text": PLANNER,
            "quote": "he rebuilt a small Postgres query planner over three months",
            "evidence_id": EVIDENCE_A_ID,
            "source_label": "Kestrel Python meetup",
            "position": 1,
        },
        overrides,
    )


def mentoring_item(**overrides: Any) -> RelatableItem:
    return _build(
        {
            "text": MENTORING,
            "quote": "he is now mentoring two juniors at Glyde",
            "evidence_id": EVIDENCE_B_ID,
            "source_label": "Coffee, Brookside",
            "position": 1,
            "days": 38,
        },
        overrides,
    )


def profile_observation_item(**overrides: Any) -> RelatableItem:
    """An observation about the entity. Nobody asserted it."""
    return make_observation(
        text=PROFILE,
        quote="her public profile listed experience as a secretary",
        evidence_id=EVIDENCE_B_ID,
        source_label="Public profile",
        **overrides,
    )


def make_pair(left: RelatableItem, right: RelatableItem) -> CandidatePair:
    """A pair in canonical order, as candidate generation would produce it."""
    ordered = (left, right) if left.item_id < right.item_id else (right, left)
    return CandidatePair(
        left=ordered[0], right=ordered[1], lexical_overlap=0.0, stance_complementary=True
    )


def classifier_response(*results: dict[str, Any]) -> str:
    """A well-formed classifier response body."""
    return json.dumps({"results": list(results)})


def qualifies(pair_id: str, *, direction: str = "b_qualifies_a") -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "relation_type": "qualifies",
        "direction": direction,
        "explanation": (
            "The second statement adds a time-phase condition to the broader "
            "recommendation: the rapid building it advises was preceded by six "
            "weeks spent mostly reading."
        ),
    }


def reinforces(pair_id: str) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "relation_type": "reinforces",
        "explanation": (
            "The second statement supplies a concrete instance of the practice "
            "the first one recommends, rather than restating it."
        ),
    }


def contradicts(pair_id: str) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "relation_type": "contradicts",
        "explanation": (
            "The two statements describe the same period in ways that cannot both hold as written."
        ),
    }


def no_relation(pair_id: str) -> dict[str, Any]:
    return {"pair_id": pair_id, "relation_type": "no_meaningful_relation"}
