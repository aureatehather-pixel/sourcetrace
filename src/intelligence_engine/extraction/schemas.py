"""Structured extraction schemas.

Two distinct groups live here:

1. **Provider output schemas** (`ExtractionOutput` and friends) — the strict
   parse applied to whatever the model returns. This is a trust boundary:
   model output is untrusted input, exactly like an HTTP request body. Unknown
   item kinds, missing fields, wrong types and extra keys all fail rather than
   being coerced or dropped.
2. **API schemas** (`ExtractionRunRead`, `ExtractionItemRead`) — what the HTTP
   layer returns.

Epistemic note: the kinds are deliberately shaped so the model cannot collapse
"someone said X" into "X is true". A `reported_claim` records an attributed
assertion; an `observation` records something the evidence directly states was
seen. They are different tables rows with different fields, not one "fact" kind
with a flag.
"""

from __future__ import annotations

import datetime as dt
import enum
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Bounded on purpose. Adding a kind is a schema change and needs a version bump.
KIND_REPORTED_CLAIM = "reported_claim"
KIND_OBSERVATION = "observation"
KIND_ENTITY_MENTION = "entity_mention"
KIND_PROJECT = "project"
KIND_RESOURCE = "resource"
KIND_EVENT = "event"
KIND_OPPORTUNITY = "opportunity"
KIND_IDEA = "idea"


class ItemKind(enum.StrEnum):
    """The bounded set of things M2 knows how to extract."""

    REPORTED_CLAIM = KIND_REPORTED_CLAIM
    OBSERVATION = KIND_OBSERVATION
    ENTITY_MENTION = KIND_ENTITY_MENTION
    PROJECT = KIND_PROJECT
    RESOURCE = KIND_RESOURCE
    EVENT = KIND_EVENT
    OPPORTUNITY = KIND_OPPORTUNITY
    IDEA = KIND_IDEA


class RunStatus(enum.StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# Grounding quotes are bounded so extraction items stay pointers into the
# evidence rather than copies of it.
MIN_QUOTE_LENGTH = 8
MAX_QUOTE_LENGTH = 400
MAX_ITEMS_PER_RUN = 50


class StrictModel(BaseModel):
    """Base for everything parsed out of a provider response."""

    model_config = ConfigDict(extra="forbid", strict=False, frozen=True)


# --------------------------------------------------------------------------
# Per-kind payloads. Deliberately small: a few fields each, no nesting.
# Every optional field means "the evidence did not say", never "unknown value".
# --------------------------------------------------------------------------


class ReportedClaimData(StrictModel):
    """Something a person or source asserted. NOT a statement of fact."""

    claim: str = Field(min_length=1, max_length=2000)
    claimed_by: str | None = Field(
        default=None,
        max_length=200,
        description="Who asserted it, as described in the evidence. Null if unattributed.",
    )
    stance: Literal["asserted", "recommended", "questioned", "disputed"] | None = None


class ObservationData(StrictModel):
    """Something the evidence records as directly observed.

    `subject` is the person or organisation as the evidence words it, not a
    resolved identity. It is `None` when the evidence does not say.
    """

    observation: str = Field(min_length=1, max_length=2000)
    subject: str | None = Field(
        default=None,
        max_length=200,
        description="Who the observation is about, as worded in the evidence. Null if unstated.",
    )


class EntityMentionData(StrictModel):
    """A named entity the evidence mentions. Not a resolved identity."""

    name: str = Field(min_length=1, max_length=200)
    entity_type: Literal["person", "organization", "product", "place", "other"]
    described_as: str | None = Field(
        default=None,
        max_length=300,
        description="How the evidence describes them. Not inferred biography.",
    )


class ProjectData(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


class ResourceData(StrictModel):
    """A referenced artefact: article, tool, book, repository."""

    title: str | None = Field(default=None, max_length=300)
    url: str | None = Field(default=None, max_length=2000)
    resource_type: Literal["article", "tool", "book", "repository", "other"]


class EventData(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    occurred_at_text: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "The date/time exactly as the evidence words it. Deliberately text, "
            "not a parsed timestamp: parsing 'last spring' invents precision."
        ),
    )


class OpportunityData(StrictModel):
    """A possibility the evidence explicitly raises. Not one inferred by the model."""

    summary: str = Field(min_length=1, max_length=1000)


class IdeaData(StrictModel):
    idea: str = Field(min_length=1, max_length=2000)


class Grounding(StrictModel):
    """What in the source text supports this item.

    A verbatim quote is the simplest mechanism that actually works: a human can
    read it, and the service can verify it character-for-character against the
    evidence. Offsets would be more precise and far more fragile.
    """

    quote: str = Field(min_length=MIN_QUOTE_LENGTH, max_length=MAX_QUOTE_LENGTH)


class _ItemBase(StrictModel):
    """Common to every kind: what in the source supports it.

    There is deliberately no confidence score. A model's self-rating is not a
    calibrated quantity - nothing measures it against outcomes, so "high" and
    "low" carry no defensible meaning and would read as precision the system
    does not have. Uncertainty is represented where it actually exists: in the
    evidence's own wording, preserved verbatim in the extracted text and in
    `stance` for reported claims.
    """

    grounding: Grounding


class ReportedClaimItem(_ItemBase):
    kind: Literal[ItemKind.REPORTED_CLAIM]
    data: ReportedClaimData


class ObservationItem(_ItemBase):
    kind: Literal[ItemKind.OBSERVATION]
    data: ObservationData


class EntityMentionItem(_ItemBase):
    kind: Literal[ItemKind.ENTITY_MENTION]
    data: EntityMentionData


class ProjectItem(_ItemBase):
    kind: Literal[ItemKind.PROJECT]
    data: ProjectData


class ResourceItem(_ItemBase):
    kind: Literal[ItemKind.RESOURCE]
    data: ResourceData


class EventItem(_ItemBase):
    kind: Literal[ItemKind.EVENT]
    data: EventData


class OpportunityItem(_ItemBase):
    kind: Literal[ItemKind.OPPORTUNITY]
    data: OpportunityData


class IdeaItem(_ItemBase):
    kind: Literal[ItemKind.IDEA]
    data: IdeaData


ExtractedItem = Annotated[
    ReportedClaimItem
    | ObservationItem
    | EntityMentionItem
    | ProjectItem
    | ResourceItem
    | EventItem
    | OpportunityItem
    | IdeaItem,
    Field(discriminator="kind"),
]


class ExtractionOutput(StrictModel):
    """The complete validated provider response."""

    items: list[ExtractedItem] = Field(max_length=MAX_ITEMS_PER_RUN)


# --------------------------------------------------------------------------
# API response schemas
# --------------------------------------------------------------------------


class ExtractionItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: ItemKind
    payload: dict[str, Any]
    grounding_quote: str
    position: int
    created_at: dt.datetime


class ExtractionRunRead(BaseModel):
    """One extraction attempt, with everything needed to reproduce or audit it."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    evidence_id: uuid.UUID
    input_content_hash: str
    extractor_name: str
    extractor_version: str
    schema_version: int
    status: RunStatus
    provider: str | None
    model: str | None
    started_at: dt.datetime
    completed_at: dt.datetime | None
    error_code: str | None
    error_message: str | None
    input_tokens: int | None
    output_tokens: int | None
    #: The half-open character range of the source this run read, and the
    #: durable identity of the work. No ordinal accompanies it: intervals
    #: subdivide, and any number attached to one would be redefined by a
    #: sibling's subdivision.
    window_start: int
    window_end: int
    created_at: dt.datetime
    items: list[ExtractionItemRead] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    """Everything one `extract` call did to one evidence item.

    Source-level rather than run-level, because one evidence item can need
    several runs. `runs` holds one entry per successfully read interval, in
    source order, and a document read in a single pass yields a list of one.

    `reused` is part of the contract, not a debug field: the caller needs to be
    able to tell a cache hit from a fresh extraction. It is true only when
    *every* interval was reused, so a partially cached source honestly reports
    that it spent money.

    `complete` is the answer to the only question that matters downstream — may
    this source be read as fully extracted? — and it is computed as a tiling:
    true exactly when the intervals that succeeded lay end to end over the whole
    source with no gap and no overlap. Deriving it from coverage rather than
    from a count is what makes it survive subdivision, since children cover
    precisely what their parent did.

    **`complete` means processing completeness and nothing more.** Every
    character of the source belonged to an interval that was extracted
    successfully. It carries no claim that the model found every meaningful item
    within those characters, and no caller may present it as one.
    """

    reused: bool
    complete: bool
    #: Characters in the source, so a caller can size `uncovered` against it.
    source_length: int
    #: Character ranges no successful interval covers — the source's unread
    #: regions, stated exactly rather than as a count of failures. Empty when
    #: `complete`.
    uncovered: list[tuple[int, int]] = Field(default_factory=list)
    runs: list[ExtractionRunRead]

    @property
    def items(self) -> list[ExtractionItemRead]:
        """Every item from every interval, in source order then position order."""
        return [item for run in self.runs for item in run.items]


class ExtractionRunList(BaseModel):
    items: list[ExtractionRunRead]


def wire_schema() -> dict[str, Any]:
    """The JSON Schema sent to the provider to constrain its output.

    Derived from `ExtractionOutput` so the constraint the model is given and
    the validation applied afterwards cannot drift apart.

    Four normalisations. The last two were not guesses - the live Messages API
    rejected the raw Pydantic schema with a 400 until they were applied, and
    `tests/unit/test_extraction_wire_schema.py` now pins both so they cannot be
    "tidied" back in:

    - `discriminator` is dropped. It is an OpenAPI extension, not JSON Schema.
      The `const` on each variant's `kind` already discriminates the union.
    - `title` is dropped. Noise in a constraint, and paid for as input tokens.
    - `maxItems` is dropped. The API answers:
      *"For 'array' type, property 'maxItems' is not supported"*. The bound
      still holds - `ExtractionOutput.items` enforces it on validation, so an
      over-long response is rejected on arrival rather than never produced.
    - `oneOf` becomes `anyOf`. The API answers:
      *"Schema type 'oneOf' is not supported"*. Equivalent here: every variant
      pins a different `kind` const, so at most one can ever match and
      "at least one" and "exactly one" coincide.

    The schema is a hint, never a guarantee. Output is still validated on
    arrival - see `validation.parse_output`.
    """
    normalised: dict[str, Any] = _normalise(ExtractionOutput.model_json_schema())
    return normalised


# Keywords the Messages API structured-output subset rejects outright.
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
