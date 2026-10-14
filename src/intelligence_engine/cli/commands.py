"""The CLI's command vocabulary — one frozen dataclass per operation.

Parsing produces one of these and nothing else. Everything downstream (the
orchestrator, the renderers) consumes typed values, so no part of the system
below `args.py` ever inspects `sys.argv`, coerces a string to a UUID, or decides
what a missing flag means. That split is what makes the argument surface
testable without a database and the orchestration testable without a shell.

Two rules the shapes encode:

- **Nothing here defaults a provenance field.** `source_type`, `captured_at` and
  friends are `None` when the operator did not say, and `None` reaches the
  database as "the evidence did not say". Guessing `captured_at` from a file's
  mtime would be a fabricated observation timestamp, which is exactly the kind of
  fake precision this system refuses everywhere else.
- **No command creates an entity implicitly.** `EntityNewCommand` is the only way
  an entity comes into existence, and `ProcessCommand.interactive` gates the only
  path that can reach it without a separate invocation — by asking a human.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

#: Roles a resolution may assert. Mirrors `entities.schemas.ResolutionRole`;
#: kept as literals here so `args.py` needs no domain import to validate input.
ROLE_VALUES: tuple[str, ...] = ("mention", "claimed_by")

#: Entity types. Mirrors `entities.schemas.EntityType`. Pinned by a unit test.
ENTITY_TYPE_VALUES: tuple[str, ...] = (
    "person",
    "organization",
    "product",
    "place",
    "other",
)

#: What `pie show` can display.
SHOW_TARGETS: tuple[str, ...] = ("evidence", "items", "entity", "intelligence")

MAX_LIST_LIMIT = 500


class CliUsageError(Exception):
    """The operator's input was wrong.

    Raised instead of `argparse`'s `SystemExit` so parsing is a pure function of
    its arguments and can be tested directly. `main()` turns it into a message on
    stderr and exit status 2.
    """


@dataclass(frozen=True)
class EvidenceMetadata:
    """Provenance supplied on the command line, all of it optional.

    Every field is `None`/empty when unstated. Nothing infers a value.
    """

    source_type: str | None = None
    source_label: str | None = None
    source_url: str | None = None
    captured_at: dt.datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessCommand:
    """The happy path: ingest, extract, resolve, analyse, report.

    `interactive` decides what happens at the two points where the deterministic
    matcher honestly refuses to guess (`no_candidate`, `ambiguous_candidates`).
    True asks the operator; False prints the exact follow-up command and moves
    on. It never resolves the ambiguity by itself in either mode.
    """

    path: Path
    meta: EvidenceMetadata
    analyze: bool = True
    interactive: bool = True
    force_new: bool = False
    as_json: bool = False


@dataclass(frozen=True)
class IngestCommand:
    """Evidence only. Stops before extraction."""

    path: Path
    meta: EvidenceMetadata
    force_new: bool = False
    as_json: bool = False


@dataclass(frozen=True)
class ExtractCommand:
    evidence_id: uuid.UUID
    as_json: bool = False


@dataclass(frozen=True)
class EntityNewCommand:
    entity_type: str
    canonical_name: str
    disambiguator: str | None = None
    as_json: bool = False


@dataclass(frozen=True)
class EntityListCommand:
    entity_type: str | None = None
    limit: int | None = None
    as_json: bool = False


@dataclass(frozen=True)
class ResolveCommand:
    """`entity_id` omitted runs the deterministic matcher; supplied asserts by hand."""

    extraction_item_id: uuid.UUID
    role: Literal["mention", "claimed_by"]
    entity_id: uuid.UUID | None = None
    note: str | None = None
    as_json: bool = False


@dataclass(frozen=True)
class AnalyzeCommand:
    entity_id: uuid.UUID
    as_json: bool = False


@dataclass(frozen=True)
class PacketCommand:
    """Emit the grounded evidence packet. Read-only, and no model is called.

    Deliberately has no topic or entity argument. The packet's value is
    completeness, and the cheap filter available — lexical overlap with a topic
    string — is the one `candidates.py` measured and rejected as a gate. `limit`
    is a ceiling that the packet reports hitting, never a selection.
    """

    limit: int | None = None
    as_json: bool = False


@dataclass(frozen=True)
class ValidateCommand:
    """Check an answer's citations against a freshly assembled packet.

    The packet is re-read from the database rather than taken from a file, so an
    answer is always checked against what PIE actually holds now. Validating
    against a rendered packet that could have been edited in between would check
    the answer against itself.

    `limit` omitted means **the whole corpus**, not a default page size. That is
    the correction of a real defect: this command used to inherit
    `DEFAULT_PACKET_LIMIT`, so an answer written against a 496-item packet was
    checked against the first 200 items, and uniqueness — a packet-wide property
    — was computed over a subset. A fragment colliding only with item 350 passed.
    Supplying `limit` narrows the scope deliberately, to reproduce the exact
    packet an answer was written against; the audit output then says so.
    """

    path: Path
    limit: int | None = None
    as_json: bool = False


@dataclass(frozen=True)
class ShowCommand:
    target: Literal["evidence", "items", "entity", "intelligence"]
    target_id: uuid.UUID
    as_json: bool = False


@dataclass(frozen=True)
class RejectCommand:
    relation_id: uuid.UUID
    note: str | None = None
    as_json: bool = False


Command = (
    ProcessCommand
    | IngestCommand
    | ExtractCommand
    | EntityNewCommand
    | EntityListCommand
    | ResolveCommand
    | AnalyzeCommand
    | PacketCommand
    | ValidateCommand
    | ShowCommand
    | RejectCommand
)
