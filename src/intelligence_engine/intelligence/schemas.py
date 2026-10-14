"""Relation vocabulary, classifier output schemas, and the intelligence read model.

Three groups, the same split M2 and M3 use:

1. **Vocabularies** — bounded enums shared by the ORM, the classifier and the API.
2. **Classifier output schemas** (`RelationOutput` and friends) — the strict parse
   applied to whatever the model returns. A trust boundary: unknown relation
   types, missing fields, wrong types and extra keys all fail rather than being
   coerced or dropped.
3. **API / read-model schemas** — what the HTTP layer returns.

Epistemic note that shapes all three: a relation is a **new assertion**, derived
from two grounded items and capable of being wrong. It therefore carries no
confidence number (nothing in this system calibrates one), no causal vocabulary,
and an `explanation` that is explicitly *derived reasoning* — never presented as
a grounding quote.
"""

from __future__ import annotations

import datetime as dt
import enum
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from intelligence_engine.entities.schemas import (
    EntityRead,
    ProfileAttributedClaim,
    ResolutionRole,
)
from intelligence_engine.extraction.schemas import ItemKind

# --------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------


class RelationType(enum.StrEnum):
    """The bounded set of relations M4 knows how to assert.

    Deliberately four. Every addition is a schema change and a version bump,
    because a vocabulary that grows quietly makes old stored relations mean
    something they did not mean when they were written.

    - `reinforces` — one item materially supports, independently aligns with or
      strengthens the interpretation of the other.
    - `qualifies` — one item adds an important limitation, condition, phase,
      scope or nuance to the other. **Not** a weak contradiction.
    - `contradicts` — the two cannot straightforwardly both hold under the same
      relevant interpretation and context, or one explicitly disputes the other.
    - `no_meaningful_relation` — nothing useful enough to surface. A correct and
      frequently correct answer, not a failure.

    There is deliberately no `causes`, `leads_to` or `explains`. Causal inference
    from two co-stored statements is exactly the invention this layer exists to
    refuse.
    """

    REINFORCES = "reinforces"
    QUALIFIES = "qualifies"
    CONTRADICTS = "contradicts"
    NO_MEANINGFUL_RELATION = "no_meaningful_relation"


#: Relations worth showing a human. `no_meaningful_relation` is persisted (it is
#: an expensive negative result worth caching) but never surfaced as intelligence.
MEANINGFUL_RELATION_TYPES = frozenset(
    {RelationType.REINFORCES, RelationType.QUALIFIES, RelationType.CONTRADICTS}
)

#: Relation types whose meaning depends on which item is which.
DIRECTIONAL_RELATION_TYPES = frozenset({RelationType.QUALIFIES})


class RelationDirection(enum.StrEnum):
    """Which way a directional relation points.

    Pairs are stored in canonical order (`left_extraction_item_id <
    right_extraction_item_id`, enforced by a CHECK constraint) so that
    "A relates to B" and "B relates to A" cannot both be stored as separate rows.
    Canonical ordering makes column position meaningless as a carrier of meaning,
    so direction gets its own column.

    `symmetric` is the only legal value for `reinforces`, `contradicts` and
    `no_meaningful_relation`; `qualifies` may never use it. Enforced in the
    database, not only here.
    """

    SYMMETRIC = "symmetric"
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"


class RelationStatus(enum.StrEnum):
    """Whether this assertion is still served.

    `rejected` is the whole review mechanism. A human who sees a wrong relation
    marks it rejected: it stops appearing as intelligence, it stays in the table
    as a record of what the system asserted and when, and — importantly — it
    still satisfies the reuse lookup, so the next analysis does **not** pay a
    model call to regenerate the same wrong answer. Bumping `CLASSIFIER_VERSION`
    is what re-opens the question.
    """

    CURRENT = "current"
    REJECTED = "rejected"


class AnalysisStatus(enum.StrEnum):
    """Lifecycle of one analyse-relations operation. Mirrors M2's `RunStatus`."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


#: Item kinds that carry an assertion, and can therefore stand in a relation.
#: `entity_mention`, `project`, `resource`, `event`, `opportunity` and `idea` name
#: things; naming a thing cannot reinforce, qualify or contradict anything.
#: Both kinds are now genuinely reachable: the `about` role closed the M3 gap this
#: constraint was deliberately left wider than, rather than narrowed to hide.
RELATABLE_ITEM_KINDS: frozenset[ItemKind] = frozenset(
    {ItemKind.REPORTED_CLAIM, ItemKind.OBSERVATION}
)

#: The resolution roles that put an assertion in front of M4, and the *only*
#: roles that do. `mention` is excluded on purpose: naming an entity is not an
#: assertion about it, and co-occurrence is not attribution.
#:
#: This is one constant rather than three literals because it is read in three
#: places that must agree — candidate loading, the reuse/serve join, and the CLI's
#: decision to run an analysis at all. Two of those three fail *silently* when
#: they disagree: relations get classified and persisted, then never served and
#: never reused, so every run re-pays for them while the operator sees nothing.
RELATABLE_ROLES: frozenset[ResolutionRole] = frozenset(
    {ResolutionRole.CLAIMED_BY, ResolutionRole.ABOUT}
)

MIN_EXPLANATION_LENGTH = 20
MAX_EXPLANATION_LENGTH = 400
MAX_REJECTION_NOTE_LENGTH = 500
#: Hard ceiling on results in one classifier response. Enforced after parsing, not
#: in the wire schema: the Messages API rejects `maxItems` (M2 learning 1).
MAX_RESULTS_PER_BATCH = 20


# --------------------------------------------------------------------------
# Classifier output — the trust boundary
# --------------------------------------------------------------------------


class StrictModel(BaseModel):
    """Base for everything parsed out of a classifier response."""

    model_config = ConfigDict(extra="forbid", strict=False, frozen=True)


class RelationResult(StrictModel):
    """One classified pair, as the model returned it.

    `pair_id` binds the result to a specific input pair. Every batch supplies
    explicit ids and every returned id is checked against the supplied set, so a
    model that reorders, invents or drops a pair fails validation instead of
    silently attaching an explanation to the wrong two claims.
    """

    pair_id: str = Field(min_length=1, max_length=16)
    relation_type: RelationType
    direction: Literal["a_qualifies_b", "b_qualifies_a"] | None = Field(
        default=None,
        description=(
            "Required for `qualifies` and forbidden otherwise. Says which of the "
            "two supplied items is the one adding the limitation."
        ),
    )
    explanation: str | None = Field(
        default=None,
        min_length=MIN_EXPLANATION_LENGTH,
        max_length=MAX_EXPLANATION_LENGTH,
        description=(
            "Derived reasoning, not a quote. May reference only the supplied "
            "grounded material. Omitted for no_meaningful_relation."
        ),
    )


class RelationOutput(StrictModel):
    """The complete validated classifier response for one batch."""

    results: list[RelationResult] = Field(max_length=MAX_RESULTS_PER_BATCH)


UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({"maxItems", "minItems", "oneOf", "discriminator"})

_DROP_KEYS = frozenset({"discriminator", "title", "maxItems"})
_RENAME_KEYS = {"oneOf": "anyOf"}


def _normalise(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            _RENAME_KEYS.get(key, key): _normalise(value)
            for key, value in node.items()
            if key not in _DROP_KEYS
        }
    if isinstance(node, list):
        return [_normalise(item) for item in node]
    return node


def _strip_class_docstrings(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove the `description` Pydantic derives from a class docstring.

    Only class-level ones: the root object and each `$defs` entry. Per-field
    `Field(description=...)` values live on properties and survive untouched,
    because those are written *for the model* and earn their tokens.

    The docstrings in this module are engineering documentation — why the
    vocabulary is bounded, when to bump a version, which invention the design
    refuses. None of that helps a classifier and all of it is billed as input on
    every call; measured at ~1.3KB of the ~2.5KB schema before this ran.

    It is also the right separation. The schema is a *shape* constraint; the
    meaning of `reinforces` versus `qualifies` belongs in `SYSTEM_PROMPT`, where
    it is written in wording aimed at the model rather than at a future
    maintainer. Two copies of the vocabulary's definition would be two things
    free to disagree.
    """
    schema.pop("description", None)
    for definition in schema.get("$defs", {}).values():
        if isinstance(definition, dict):
            definition.pop("description", None)
    return schema


def wire_schema() -> dict[str, Any]:
    """The JSON Schema sent to the provider to constrain its output.

    Derived from `RelationOutput` so the constraint the model is given and the
    validation applied afterwards cannot drift apart. Three normalisations, all
    pinned by `tests/unit/test_relation_wire_schema.py`:

    - `title` is dropped — noise in a constraint, paid for as input tokens;
    - `maxItems` is dropped — the API rejects it outright (M2 learning 1). The
      bound still holds: `RelationOutput.results` enforces it on arrival, so an
      over-long response is rejected rather than never produced;
    - class-docstring `description`s are dropped — see `_strip_class_docstrings`.

    `oneOf`/`discriminator` never appear here because the output is a flat object
    rather than a discriminated union — one fewer thing to normalise, and
    materially cheaper in input tokens than M2's eight-variant schema.
    """
    normalised: dict[str, Any] = _normalise(RelationOutput.model_json_schema())
    return _strip_class_docstrings(normalised)


# --------------------------------------------------------------------------
# API schemas
# --------------------------------------------------------------------------


class RelationSide(BaseModel):
    """One grounded item behind a derived relation, with its full provenance.

    Four identifiers and a verbatim quote, so a reader can walk any relation back
    to the evidence it was derived from without trusting the relation.
    """

    extraction_item_id: uuid.UUID
    extraction_run_id: uuid.UUID
    evidence_id: uuid.UUID
    resolution_id: uuid.UUID
    item_kind: ItemKind
    text: str
    stance: str | None
    #: Who asserted it. Set only for `reported_claim`; never both with `subject`.
    claimed_by: str | None
    #: Who the evidence says it is about. Set only for `observation`. Kept
    #: separate from `claimed_by` so a reader of the API can never mistake an
    #: observation about someone for something they said.
    subject: str | None
    grounding_quote: str
    source_label: str | None
    captured_at: dt.datetime | None


class IntelligenceRelationRead(BaseModel):
    """One derived relation, with everything needed to audit or reproduce it."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    analysis_run_id: uuid.UUID
    entity_id: uuid.UUID
    left_extraction_item_id: uuid.UUID
    right_extraction_item_id: uuid.UUID
    relation_type: RelationType
    direction: RelationDirection
    explanation: str | None
    classifier_name: str
    classifier_version: str
    schema_version: int
    provider: str | None
    model: str | None
    input_fingerprint: str
    status: RelationStatus
    rejected_at: dt.datetime | None
    rejection_note: str | None
    created_at: dt.datetime


class DerivedRelation(BaseModel):
    """A relation as the intelligence view presents it: assertion plus both sources.

    `explanation` is derived reasoning. It is deliberately a different field name
    from `grounding_quote`, which appears only inside `left`/`right`, so nothing
    in this response can be mistaken for something the evidence actually said.
    """

    relation: IntelligenceRelationRead
    left: RelationSide
    right: RelationSide
    reading: str = Field(
        description=(
            "Human-readable direction, e.g. 'right qualifies left'. Rendered from "
            "relation_type and direction; carries no information of its own."
        )
    )


class EntityIntelligence(BaseModel):
    """Everything derived about one entity, and the grounded material underneath it.

    `grounded_claims` is the **M3 profile's own** attributed-claim list, assembled
    from the same read model rather than re-queried here, so the two views cannot
    disagree about what the evidence says.

    `derived_relations` holds only relations that are current, produced by the
    running classifier version, meaningful, and whose *both* sides still have an
    active resolution to this entity. An M3 correction therefore propagates into
    M4 automatically, with M4 writing nothing.

    Deliberately absent: any generated biography, narrative or summary of the
    entity. A schema test forbids those field names outright, as it does for the
    M3 profile.
    """

    entity: EntityRead
    grounded_claims: list[ProfileAttributedClaim]
    derived_relations: list[DerivedRelation]


class AnalysisTelemetry(BaseModel):
    """What one analyse-relations operation actually cost.

    Enough to answer "how much model work did this require", and no more. Not a
    billing system: no prices, no currency, no rate table. Prices change and do
    not belong in a domain model.
    """

    #: The item-loading bound, reported before the pair bounds so the two are
    #: never read as one number. `relatable_items_total` is what the entity has;
    #: `relatable_items_loaded` is what the analysis actually saw. When they
    #: differ, every pair figure below describes the loaded subset only, and
    #: `relatable_items_truncated` is what says so.
    relatable_items_total: int = Field(
        description="Relatable items this entity has, counted before any limit."
    )
    relatable_items_loaded: int = Field(
        description="Relatable items the analysis actually loaded and paired over."
    )
    relatable_items_truncated: bool = Field(
        description=(
            "Whether the item limit discarded evidence. When true, every pair "
            "count below is over the loaded subset, not the entity's population."
        )
    )

    theoretical_pairs: int = Field(
        description=(
            "n*(n-1)/2 over the relatable items that were *loaded*. Not over the "
            "entity's population: see relatable_items_truncated."
        )
    )
    candidate_pairs: int = Field(description="Survived deterministic narrowing.")
    reused_pairs: int = Field(description="Already settled under this classifier version.")
    classified_pairs: int = Field(description="Actually sent to the model.")
    batch_count: int
    external_call_count: int
    new_relations: int = Field(description="Rows written, including no_meaningful_relation.")
    new_meaningful_relations: int
    input_tokens: int | None
    output_tokens: int | None


class AnalysisResponse(BaseModel):
    """The outcome of an analyse-relations call.

    `performed_calls` is part of the contract rather than a debug field: the
    caller has to be able to tell a fully-reused analysis from one that spent
    money, exactly as M2's `reused` does.
    """

    performed_calls: bool
    analysis_run_id: uuid.UUID | None
    entity_id: uuid.UUID
    classifier_name: str
    classifier_version: str
    schema_version: int
    telemetry: AnalysisTelemetry
    relations: list[IntelligenceRelationRead]


class RelationReject(BaseModel):
    """Mark a derived relation wrong.

    The note is for the audit trail and is never populated from model output.
    """

    model_config = ConfigDict(extra="forbid")

    note: str | None = Field(default=None, min_length=1, max_length=MAX_REJECTION_NOTE_LENGTH)
