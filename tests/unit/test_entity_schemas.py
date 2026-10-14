"""Entity vocabulary and wire-contract pins.

The load-bearing test here is the first one: M3's entity types must stay exactly
the types an M2 `entity_mention` can carry. An entity type nothing can ever be
resolved to is furniture; a mention type with no entity to resolve to is a gap.
Both would only be discovered much later, in data.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from intelligence_engine.entities.schemas import (
    ENTITY_TYPE_VALUES,
    EntityCreate,
    EntityProfile,
    EntityType,
    ResolutionCreate,
    ResolutionMethod,
    ResolutionOutcome,
    ResolutionRole,
    ResolutionSupersede,
    UnresolvedReason,
)


def test_entity_types_are_exactly_the_m2_entity_mention_types() -> None:
    assert tuple(member.value for member in EntityType) == ENTITY_TYPE_VALUES


def test_no_speculative_entity_types_were_invented() -> None:
    """`project`, `resource` and `event` are M2 *item kinds*, not mention types.

    They have their own payload shapes and no `entity_mention` can ever name
    one, so an entity of that type could never be resolved to.
    """
    values = {member.value for member in EntityType}
    assert values.isdisjoint({"project", "resource", "event", "opportunity", "idea"})


def test_resolution_methods_are_categories_not_scores() -> None:
    """No confidence field exists anywhere in the resolution contract."""
    assert {member.value for member in ResolutionMethod} == {"exact_match", "manual"}

    fields = (
        set(ResolutionCreate.model_fields)
        | set(ResolutionSupersede.model_fields)
        | set(ResolutionOutcome.model_fields)
    )
    assert not any("confidence" in name or "score" in name for name in fields)


def test_unresolved_is_a_first_class_outcome() -> None:
    outcome = ResolutionOutcome(status="unresolved", reason=UnresolvedReason.AMBIGUOUS_CANDIDATES)

    assert outcome.resolution is None
    assert outcome.reason is UnresolvedReason.AMBIGUOUS_CANDIDATES


def test_entity_create_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        EntityCreate.model_validate(
            {"entity_type": "person", "canonical_name": "Zilan", "confidence": 0.93}
        )


def test_entity_create_rejects_a_blank_name() -> None:
    with pytest.raises(ValidationError):
        EntityCreate.model_validate({"entity_type": "person", "canonical_name": ""})


def test_entity_create_rejects_an_unknown_type() -> None:
    with pytest.raises(ValidationError):
        EntityCreate.model_validate({"entity_type": "spaceship", "canonical_name": "X"})


def test_resolution_request_defaults_to_deterministic_matching() -> None:
    request = ResolutionCreate.model_validate({"role": "mention"})

    assert request.entity_id is None
    assert request.role is ResolutionRole.MENTION


def test_supersede_may_name_no_replacement() -> None:
    """Detaching is a legitimate correction: 'not that entity, and I don't know which'."""
    request = ResolutionSupersede.model_validate({"note": "wrong Alex Lee"})

    assert request.entity_id is None


def test_profile_carries_no_narrative_field() -> None:
    """A profile is grounded lines with provenance, never generated biography."""
    forbidden = {"summary", "biography", "bio", "narrative", "description", "about"}

    assert forbidden.isdisjoint(EntityProfile.model_fields)


def test_every_profile_entry_type_requires_full_provenance() -> None:
    from intelligence_engine.entities.schemas import ProfileAttributedClaim, ProfileMention

    required = {
        "resolution_id",
        "extraction_item_id",
        "extraction_run_id",
        "evidence_id",
        "grounding_quote",
    }
    for model in (ProfileMention, ProfileAttributedClaim):
        assert required <= set(model.model_fields)
        assert all(model.model_fields[name].is_required() for name in required)


def test_profile_rejects_an_entry_missing_its_evidence_id() -> None:
    """Provenance is not optional; a line with no source cannot be constructed."""
    from intelligence_engine.entities.schemas import ProfileMention

    with pytest.raises(ValidationError):
        ProfileMention.model_validate(
            {
                "resolution_id": uuid.uuid4(),
                "extraction_item_id": uuid.uuid4(),
                "extraction_run_id": uuid.uuid4(),
                "grounding_quote": "Zilan told me",
                "method": "manual",
                "resolved_at": "2026-08-27T00:00:00+00:00",
                "name_as_mentioned": "Zilan",
                "mentioned_as_type": "person",
                "described_as": None,
            }
        )
