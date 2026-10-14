"""Entity, resolution and profile schemas.

Three groups live here:

1. **Vocabularies** — the bounded enums that both the ORM and the API use.
2. **API request/response models** — the wire contract.
3. **Profile read model** — the shape assembled from evidence, extraction items
   and active resolutions at request time. It is a *view*, never a stored blob:
   see `EntityProfile`.

Epistemic note that shapes all three: nothing here carries a confidence number.
A resolution is either asserted or it is not, and *how* it was decided is
recorded as a `ResolutionMethod` category. There is no calibration behind a
probability in this system, so there is no probability.
"""

from __future__ import annotations

import datetime as dt
import enum
import uuid
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from intelligence_engine.extraction.schemas import EntityMentionData

# The entity types M3 supports are exactly the types an M2 `entity_mention` can
# carry. Derived from the M2 model rather than retyped, so the two cannot drift:
# an entity type nothing can ever be resolved to would be furniture, and a
# mention type with no entity to resolve to would be a gap.
ENTITY_TYPE_VALUES: tuple[str, ...] = tuple(
    get_args(EntityMentionData.model_fields["entity_type"].annotation)
)


class EntityType(enum.StrEnum):
    """Mirrors `EntityMentionData.entity_type`. Pinned by a unit test."""

    PERSON = "person"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    PLACE = "place"
    OTHER = "other"


class ResolutionRole(enum.StrEnum):
    """*What* about the source item is being said to refer to the entity.

    - `mention` — the item is an `entity_mention`; the thing it names is this
      entity.
    - `claimed_by` — the item is a `reported_claim`; the person or source its
      `claimed_by` string attributes the claim to is this entity.
    - `about` — the item is an `observation`; the subject the evidence says it
      concerns is this entity. It asserts *only* that the observation is about
      the entity. It never says the entity made, endorsed or is the source of
      the observation; that is what `claimed_by` is for, and the two must not be
      read as interchangeable.

    Three roles, each valid for exactly one item kind. Without the role, a
    resolution row could not say whether it meant "this claim is *about* the
    entity" or "the entity *made* this claim" — a distinction the whole
    reported-claim design exists to protect.
    """

    MENTION = "mention"
    CLAIMED_BY = "claimed_by"
    ABOUT = "about"


class ResolutionMethod(enum.StrEnum):
    """How the assertion was arrived at. A category, never a score."""

    EXACT_MATCH = "exact_match"
    MANUAL = "manual"


class UnresolvedReason(enum.StrEnum):
    """Why no resolution was asserted. Unresolved is a valid, final state."""

    NO_MATCHABLE_NAME = "no_matchable_name"
    NO_CANDIDATE = "no_candidate"
    AMBIGUOUS_CANDIDATES = "ambiguous_candidates"
    DETACHED_BY_CORRECTION = "detached_by_correction"


MAX_CANONICAL_NAME_LENGTH = 200
MAX_DISAMBIGUATOR_LENGTH = 300
MAX_NOTE_LENGTH = 500

CanonicalName = Annotated[str, Field(min_length=1, max_length=MAX_CANONICAL_NAME_LENGTH)]
Disambiguator = Annotated[str, Field(min_length=1, max_length=MAX_DISAMBIGUATOR_LENGTH)]
Note = Annotated[str, Field(min_length=1, max_length=MAX_NOTE_LENGTH)]


# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------


class EntityCreate(BaseModel):
    """Creating an entity is a human assertion that this identity exists.

    Nothing creates entities as a side effect of ingesting evidence or running
    extraction. That is the main defence against entity pollution: a thousand
    evidence items produce zero entities until somebody decides one is real.
    """

    model_config = ConfigDict(extra="forbid")

    entity_type: EntityType
    canonical_name: CanonicalName = Field(
        description="Display name. Deliberately NOT unique - there are many John Smiths."
    )
    disambiguator: Disambiguator | None = Field(
        default=None,
        description=(
            "Human-written note that tells this entity apart from a same-named "
            "one, e.g. 'the backend engineer from DevGuild'. Never generated."
        ),
    )


class EntityRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    entity_type: EntityType
    canonical_name: str
    normalized_name: str
    disambiguator: str | None
    created_at: dt.datetime


class EntityPage(BaseModel):
    items: list[EntityRead]
    next_cursor: str | None = Field(
        default=None, description="Pass as ?cursor= to fetch the next page. Null when exhausted."
    )


# --------------------------------------------------------------------------
# Resolutions
# --------------------------------------------------------------------------


class ResolutionCreate(BaseModel):
    """Ask for one extraction item to be connected to an entity.

    Two modes, chosen by whether `entity_id` is supplied:

    - **supplied** — a human is asserting the link. Recorded as `manual`.
    - **omitted** — run the conservative deterministic matcher. It resolves only
      on exactly one candidate and otherwise reports why it declined.
    """

    model_config = ConfigDict(extra="forbid")

    role: ResolutionRole
    entity_id: uuid.UUID | None = Field(
        default=None,
        description="Omit to attempt deterministic exact matching instead.",
    )
    note: Note | None = Field(default=None, description="Why, for the audit trail.")


class ResolutionSupersede(BaseModel):
    """Correct an existing resolution without rewriting it.

    `entity_id` supplied re-points the item at a different entity. `entity_id`
    omitted detaches it entirely — the operator is saying "this is not that
    entity" without claiming to know which entity it is. The item returns to
    unresolved, which is a legitimate answer.
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: uuid.UUID | None = None
    note: Note | None = None


class EntityResolutionRead(BaseModel):
    """One resolution assertion, active or superseded."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    extraction_item_id: uuid.UUID
    source_item_kind: str
    entity_id: uuid.UUID
    role: ResolutionRole
    method: ResolutionMethod
    is_active: bool
    superseded_by_id: uuid.UUID | None
    superseded_at: dt.datetime | None
    note: str | None
    created_at: dt.datetime


class ResolutionOutcome(BaseModel):
    """What a resolve or supersede attempt actually did.

    `unresolved` is a success, not an error: the system looked and honestly
    declined. `candidate_entity_ids` shows what the matcher saw, so an operator
    facing `ambiguous_candidates` can immediately pick the right one by hand.
    """

    status: Literal["resolved", "unresolved"]
    reason: UnresolvedReason | None = None
    candidate_entity_ids: list[uuid.UUID] = Field(default_factory=list)
    resolution: EntityResolutionRead | None = None
    superseded: EntityResolutionRead | None = Field(
        default=None, description="The resolution this call deactivated, if any."
    )


class ResolutionHistory(BaseModel):
    """Every resolution ever asserted for one extraction item, newest first."""

    items: list[EntityResolutionRead]


# --------------------------------------------------------------------------
# Profile read model
# --------------------------------------------------------------------------


class ProfileSource(BaseModel):
    """One evidence item that currently contributes to this profile."""

    evidence_id: uuid.UUID
    source_type: str | None
    source_label: str | None
    source_url: str | None
    captured_at: dt.datetime | None
    created_at: dt.datetime


class _ProfileEntry(BaseModel):
    """Provenance carried by every profile line, without exception.

    Four identifiers and a verbatim quote. Any line in a profile can be traced
    back to the exact evidence item, the exact extraction run that produced it,
    and the exact resolution that decided it belongs here.
    """

    resolution_id: uuid.UUID
    extraction_item_id: uuid.UUID
    extraction_run_id: uuid.UUID
    evidence_id: uuid.UUID
    grounding_quote: str
    method: ResolutionMethod
    resolved_at: dt.datetime


class ProfileMention(_ProfileEntry):
    """The entity was named here, and how the evidence described it."""

    name_as_mentioned: str
    mentioned_as_type: EntityType
    described_as: str | None


class ProfileAttributedClaim(_ProfileEntry):
    """Something this entity is recorded as having said.

    Still a `reported_claim`. Attributing it to a resolved entity does not
    promote it to fact - `claim` remains what was asserted, `stance` remains how
    it was offered, and the grounding quote remains the source text.
    """

    claim: str
    claimed_by: str | None
    stance: str | None


class ProfileSubjectObservation(_ProfileEntry):
    """Something the evidence records as observed *about* this entity.

    Distinct from `ProfileAttributedClaim` in the one way that matters: nobody
    asserted this. The entity is the subject of the observation, not its source.
    "Her profile lists Rust" belongs here; "she told me she knows Rust" is a
    claim. Collapsing the two would turn what someone noticed into what someone
    said, which is the failure the whole reported-claim design prevents.

    `subject` is the wording the evidence used, kept beside the resolved entity
    rather than replaced by it: the resolution can be corrected later, and the
    original phrasing is what a reader needs to audit that correction.
    """

    observation: str
    subject: str | None


class EntityProfile(BaseModel):
    """Everything the system currently believes about one entity, and why.

    Assembled by query from `entities`, `entity_resolutions`, `extraction_items`
    and `evidence_items` on every request. Nothing here is stored: a cached
    profile is a copy that drifts, and this system's whole value is that derived
    information can always be regenerated from evidence that never changed.

    Note what is deliberately absent: no biography, no summary, no inferred
    attributes, and nothing drawn from an evidence item merely because the
    entity also appeared in it. Co-occurrence is not attribution; concluding
    that every claim in an evidence item belongs to every person mentioned in it
    is precisely the unsupported leap M3 exists to avoid. A line appears here
    only because a resolution says it belongs.
    """

    entity: EntityRead
    evidence_count: int
    sources: list[ProfileSource]
    mentions: list[ProfileMention]
    attributed_claims: list[ProfileAttributedClaim]
    #: Observations the evidence says are about this entity. Never merged with
    #: `attributed_claims`: what was said and what was noticed stay separate.
    subject_observations: list[ProfileSubjectObservation]
