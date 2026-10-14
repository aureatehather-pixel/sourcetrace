"""What a grounded evidence packet is, and which kinds may enter one.

The one constant in this file that matters most is `PACKET_ITEM_KINDS`, and the
most important thing about it is what it is **not**.

## Packet inclusion is not M4 relation eligibility

`intelligence.schemas.RELATABLE_ITEM_KINDS` is `{reported_claim, observation}`
and answers a different question: *which items can stand on one side of an
asserted relation?* Naming a project cannot reinforce, qualify or contradict
anything, so `project` is correctly excluded there — and that exclusion is
enforced in PostgreSQL, rendered into CHECK constraints on
`intelligence_relations` (`intelligence/models.py`). Widening it is a schema
change, not a filter change.

Packet inclusion answers: *which stored items carry grounded material a reader or
a model should be shown?* An `idea` cannot participate in a relation, and is
still among the most load-bearing evidence there is for a decision — the
experiment found exactly that.

The two sets are therefore kept **structurally independent**: this constant is
declared here, nothing in `packet/` imports `RELATABLE_ITEM_KINDS`, and the
packet's read path goes to `extraction_items` directly rather than through
`intelligence`'s repository. A test pins both the value of
`RELATABLE_ITEM_KINDS` and the absence of the import, so widening one set cannot
silently widen the other.

## Why `entity_mention` and `resource` are out

They name things without recording anything about them. An `entity_mention` is
M3's raw material — "this note said the word 'Zilan'" — and the entity context
it produces already travels on the items it resolved. A `resource` is a pointer
to an artefact, not a statement the artefact makes. Including either would pad
the packet with rows that cannot support a conclusion, which is the failure mode
this system calls fake precision.
"""

from __future__ import annotations

import datetime as dt
import uuid

from pydantic import BaseModel, ConfigDict

from intelligence_engine.entities.schemas import ResolutionRole
from intelligence_engine.extraction.schemas import ItemKind

#: Item kinds that carry grounded material worth putting in front of a reader.
#:
#: Independent of `RELATABLE_ITEM_KINDS` by construction — see the module
#: docstring. Six of M2's eight kinds; `entity_mention` and `resource` name
#: things rather than recording anything about them.
PACKET_ITEM_KINDS: frozenset[ItemKind] = frozenset(
    {
        ItemKind.REPORTED_CLAIM,
        ItemKind.OBSERVATION,
        ItemKind.IDEA,
        ItemKind.OPPORTUNITY,
        ItemKind.PROJECT,
        ItemKind.EVENT,
    }
)

#: A pathological corpus should not turn one command into an unbounded read. The
#: real corpus is ~46 items; this is a ceiling, not a target.
DEFAULT_PACKET_LIMIT = 200
MAX_PACKET_LIMIT = 1000


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class EntityContext(FrozenModel):
    """An entity M3 asserted this item connects to, and in what role.

    Present only where a resolution actually exists. Four of the six packet
    kinds cannot carry one at all: `entity_resolutions` has a CHECK binding each
    role to exactly one item kind, so `idea`, `opportunity`, `project` and
    `event` have no role available to them. That is why this is context attached
    to an item rather than the axis items are selected on — selecting by entity
    would silently drop the four kinds a decision most depends on.

    `role` is carried, never derived from `kind`. Deriving one from the other is
    precisely how "this entity said it" came to be rendered as "this is about
    them".
    """

    entity_id: uuid.UUID
    canonical_name: str
    role: ResolutionRole
    resolution_id: uuid.UUID


class PacketItem(FrozenModel):
    """One grounded item, carrying everything a citation of it needs.

    `item_id` is the identifier a downstream answer must cite, and the one
    `citations.validate_citations` checks membership against. It is the
    `extraction_items` primary key, so a cited id resolves to a real row rather
    than to a number this layer invented.

    `claimed_by` and `subject` stay separate and are never both set, exactly as
    they are in `RelatableItem` and in the database's role/kind CHECK.
    """

    item_id: uuid.UUID
    kind: ItemKind
    text: str
    stance: str | None
    claimed_by: str | None
    subject: str | None
    #: Remaining payload fields the evidence actually supplied, in the order the
    #: M2 model declares them. Absent fields are absent, never defaulted.
    attributes: tuple[tuple[str, str], ...]
    grounding_quote: str
    extraction_item_position: int
    extraction_run_id: uuid.UUID
    evidence_id: uuid.UUID
    source_label: str | None
    captured_at: dt.datetime | None
    entities: tuple[EntityContext, ...]


class PacketSource(FrozenModel):
    """One evidence item contributing to this packet.

    **One entry per evidence item, never per extraction interval.** A large
    export is read in several intervals, but it remains one conversation, one
    provenance and one source; rendering the interval count as a source count
    would tell a reader that six independent documents agree when one document
    was read six times. The coverage fields describe that mechanism without ever
    being mistaken for it.
    """

    evidence_id: uuid.UUID
    source_type: str | None
    source_label: str | None
    source_url: str | None
    captured_at: dt.datetime | None
    created_at: dt.datetime
    item_count: int

    #: Characters in the source, and how many of them fall inside an interval
    #: with a current succeeded run. Stated in characters rather than as a
    #: count of intervals because intervals subdivide: their number changes
    #: while the source does not.
    source_length: int = 0
    covered_chars: int = 0
    #: The exact character ranges no successful interval covers. Empty when the
    #: source was read whole; otherwise these are the regions PIE knows it did
    #: not read, named rather than merely counted.
    uncovered: tuple[tuple[int, int], ...] = ()

    @property
    def complete(self) -> bool:
        """Whether every character of this source falls in a successful interval.

        **Processing completeness only.** It means no region of the source went
        unread. It carries no claim that the model found every meaningful item
        inside the regions it did read, and the renderer must not word it as if
        it did.
        """
        return not self.uncovered


class EvidencePacket(FrozenModel):
    """Every grounded item currently stored, in a deterministic order.

    Ordered by evidence arrival, then evidence id, then item position — the same
    order M4 loads candidates in, so two readers of the same database state see
    the same packet.

    `truncated` is stated rather than implied. A packet that quietly stopped at a
    limit would be a compression that lost material invisibly, which is the exact
    failure this package exists to prevent.
    """

    items: tuple[PacketItem, ...]
    sources: tuple[PacketSource, ...]
    total_items_available: int
    truncated: bool

    @property
    def item_ids(self) -> frozenset[uuid.UUID]:
        """The closed set a citation must belong to."""
        return frozenset(item.item_id for item in self.items)
