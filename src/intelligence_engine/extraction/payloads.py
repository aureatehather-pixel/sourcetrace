"""Reading a stored extraction payload back, according to its own kind.

M2 writes eight different payload shapes into one JSONB column. Anything that
later wants to *display* a stored item — rather than act on one particular kind
of it — has to turn that column back into text without guessing which shape it
holds. This module is that step, and it lives in `extraction/` because the
payload models it dispatches over are extraction's own.

Two properties are load-bearing:

**Total over `ItemKind`.** Every kind M2 can write, this can read. A reader that
covers a subset would push the "which kinds are allowed here?" question into the
caller, and that question has different answers in different places — which is
exactly how a filter and a dispatch drift apart.

**Nothing is invented.** The payload is re-validated through the same Pydantic
model that wrote it, so JSONB stored under an older schema version fails loudly
here rather than yielding a silently wrong statement. Fields the evidence did not
supply are simply absent from `attributes`; none of them acquire a default, a
placeholder, or an inferred value.

`claimed_by` and `subject` stay two fields and are never both set, for the reason
the whole reported-claim design exists: a claim has a speaker, an observation has
a subject it is *about*, and collapsing them turns "her profile lists Solidity"
into something she asserted.

This is deliberately **not** `intelligence.service._read_assertion`, and does not
replace it. That function is total over exactly the two kinds M4 permits and
raises on anything else *on purpose* — the raise is a drift alarm telling M4 that
its kind filter and its dispatch have come apart. Widening it would delete the
alarm. Two functions with different totality, because they answer different
questions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from intelligence_engine.extraction.errors import ExtractionError
from intelligence_engine.extraction.schemas import (
    EntityMentionData,
    EventData,
    IdeaData,
    ItemKind,
    ObservationData,
    OpportunityData,
    ProjectData,
    ReportedClaimData,
    ResourceData,
)


class UnreadablePayload(ExtractionError):
    """A stored payload could not be read back under the current schema.

    Loud rather than skipped, for the reason every other assembly error in this
    system is loud: dropping the item would remove grounded material from a
    derived view without telling anyone, and a reader cannot detect an absence.
    """

    error_code = "unreadable_payload"


@dataclass(frozen=True, slots=True)
class ItemStatement:
    """One stored item's payload, read as text plus whatever else it recorded.

    `text` is the item's primary content as the evidence worded it: the claim,
    the observation, the idea, the name of the project. `attributes` carries the
    kind's remaining fields as ordered `(name, value)` pairs, present only when
    the evidence actually supplied them.
    """

    text: str
    stance: str | None = None
    claimed_by: str | None = None
    subject: str | None = None
    attributes: tuple[tuple[str, str], ...] = field(default=())


def _attributes(*pairs: tuple[str, str | None]) -> tuple[tuple[str, str], ...]:
    """Keep the pairs the evidence supplied, in declaration order, and no others."""
    return tuple((name, value) for name, value in pairs if value is not None and value != "")


def read_statement(kind: ItemKind, payload: object, *, item_label: str) -> ItemStatement:
    """Read one payload through the M2 model for `kind`.

    `item_label` is folded into the error message so a failure names the row that
    caused it. It is caller-supplied rather than derived, because this function
    takes a payload, not an ORM object, and therefore stays usable without a
    database session.
    """
    reader = _READERS[kind]
    try:
        return reader(payload)
    except ValidationError as exc:
        raise UnreadablePayload(f"{item_label} payload is not a readable {kind.value}") from exc


def _reported_claim(payload: object) -> ItemStatement:
    data = ReportedClaimData.model_validate(payload)
    return ItemStatement(text=data.claim, stance=data.stance, claimed_by=data.claimed_by)


def _observation(payload: object) -> ItemStatement:
    data = ObservationData.model_validate(payload)
    # No stance: an observation is not offered as advice or as a position, so
    # inventing one would state something the evidence never did.
    return ItemStatement(text=data.observation, subject=data.subject)


def _entity_mention(payload: object) -> ItemStatement:
    data = EntityMentionData.model_validate(payload)
    return ItemStatement(
        text=data.name,
        attributes=_attributes(
            ("entity_type", data.entity_type),
            ("described_as", data.described_as),
        ),
    )


def _project(payload: object) -> ItemStatement:
    data = ProjectData.model_validate(payload)
    return ItemStatement(
        text=data.name,
        attributes=_attributes(("description", data.description)),
    )


def _resource(payload: object) -> ItemStatement:
    data = ResourceData.model_validate(payload)
    # A resource may have no title at all; its type is the only always-present
    # field, so it becomes the text rather than a title being manufactured.
    return ItemStatement(
        text=data.title or data.url or data.resource_type,
        attributes=_attributes(
            ("resource_type", data.resource_type),
            ("url", data.url),
        ),
    )


def _event(payload: object) -> ItemStatement:
    data = EventData.model_validate(payload)
    # `occurred_at_text` stays text. It is the date as the evidence worded it,
    # and parsing "last spring" into a timestamp would invent precision.
    return ItemStatement(
        text=data.name,
        attributes=_attributes(("occurred_at_text", data.occurred_at_text)),
    )


def _opportunity(payload: object) -> ItemStatement:
    data = OpportunityData.model_validate(payload)
    return ItemStatement(text=data.summary)


def _idea(payload: object) -> ItemStatement:
    data = IdeaData.model_validate(payload)
    return ItemStatement(text=data.idea)


#: Exhaustive over `ItemKind`. Pinned by a unit test, so adding a kind to M2
#: without teaching this module to read it fails the suite rather than raising a
#: `KeyError` in front of an operator.
_READERS: dict[ItemKind, Callable[[object], ItemStatement]] = {
    ItemKind.REPORTED_CLAIM: _reported_claim,
    ItemKind.OBSERVATION: _observation,
    ItemKind.ENTITY_MENTION: _entity_mention,
    ItemKind.PROJECT: _project,
    ItemKind.RESOURCE: _resource,
    ItemKind.EVENT: _event,
    ItemKind.OPPORTUNITY: _opportunity,
    ItemKind.IDEA: _idea,
}
