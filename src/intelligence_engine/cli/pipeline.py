"""The orchestration behind `pie process`: one note, all the way through.

This module is the only place that knows the normal order of the M1-M4 loop.
It adds no persistence and no domain logic of its own — every write goes through
the same service methods the HTTP routes call, so the two entry points cannot
come to mean different things.

Three orchestration decisions worth stating, because each of them is a judgement
rather than plumbing:

**Mentions are resolved before attributions.** An `entity_mention` names a thing
and carries its type; a `reported_claim`'s `claimed_by` is an attribution string
that often names nobody. Resolving mentions first means an entity created for
"Zilan" in this run is already on file when the claim attributed to "Zilan" is
reached, so one decision settles both. Order changes nothing about what is true —
each resolution is independent — it only avoids asking the same question twice.

**Nothing creates an entity from an attribution or subject string.** When a
`claimed_by` or an `observation.subject` finds no candidate, the offered answers
are "use one of these" and "skip", never "create". `claimed_by = "Engineer at
DevGuild"` is useful attribution that identifies nobody, and `subject = "I"` is
an honest subject naming nobody on file; manufacturing a person from either is
precisely the failure M3 exists to prevent. Creating an identity stays a
separate, deliberate act.

**Analysis follows attribution, not ingestion.** Only entities that gained an
assertion in this run are analysed - a claim attributed to them, or an
observation the evidence says is about them. Those are exactly `RELATABLE_ROLES`,
the same set M4 loads and serves through, so the three sites cannot drift into a
state where work is done and never shown. Analysing an entity that merely got
named would spend a model call on nothing.

Re-running `pie process` over the same file is safe and close to free: identical
content is recognised rather than duplicated, extraction reuses its stored run,
and settled relation pairs are reused. What it will not do is silently redo a
decision a human already made — an item that already has an active resolution is
reported as such and left alone, because changing one is a correction and
corrections go through supersession.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from intelligence_engine.cli.commands import CliUsageError, EvidenceMetadata
from intelligence_engine.cli.context import Services
from intelligence_engine.cli.decisions import (
    Candidate,
    CreateEntity,
    Decider,
    Question,
    Skip,
    UseEntity,
)
from intelligence_engine.entities.errors import ResolutionAlreadyExists
from intelligence_engine.entities.matching import matchable_name
from intelligence_engine.entities.schemas import (
    EntityCreate,
    EntityRead,
    EntityType,
    ResolutionCreate,
    ResolutionRole,
    UnresolvedReason,
)
from intelligence_engine.evidence.schemas import EvidenceCreate, EvidenceRead
from intelligence_engine.extraction.schemas import ExtractionItemRead, ExtractionRunRead, ItemKind
from intelligence_engine.intelligence.schemas import (
    RELATABLE_ROLES,
    AnalysisResponse,
    EntityIntelligence,
)

#: Which role applies to which extracted kind. Mirrors `matching.ROLE_SOURCE_KINDS`.
RESOLVABLE: dict[ItemKind, ResolutionRole] = {
    ItemKind.ENTITY_MENTION: ResolutionRole.MENTION,
    ItemKind.REPORTED_CLAIM: ResolutionRole.CLAIMED_BY,
    ItemKind.OBSERVATION: ResolutionRole.ABOUT,
}

#: Mentions first — see the module docstring.
KIND_ORDER: dict[ItemKind, int] = {
    ItemKind.ENTITY_MENTION: 0,
    ItemKind.REPORTED_CLAIM: 1,
    ItemKind.OBSERVATION: 2,
}

Outcome = Literal[
    "resolved",
    "created",
    "already_resolved",
    "declined",
    "unnamed",
]


@dataclass(frozen=True)
class EvidenceStep:
    evidence: EvidenceRead
    reused: bool


@dataclass(frozen=True)
class ExtractionStep:
    """One evidence item's extraction, however many intervals it took.

    `runs` holds one entry per successfully read interval, in source order — a
    single entry for anything read in one pass, which is most notes. `complete`
    is false when any region of the source went unread, `uncovered` names those
    regions exactly, and the renderer says so rather than letting a partial
    extraction read like a whole one.
    """

    runs: tuple[ExtractionRunRead, ...]
    reused: bool
    complete: bool
    uncovered: tuple[tuple[int, int], ...] = ()

    @property
    def items(self) -> list[ExtractionItemRead]:
        return [item for run in self.runs for item in run.items]


@dataclass(frozen=True)
class ResolutionRecord:
    """What happened to one extraction item, and what a human could do next."""

    extraction_item_id: uuid.UUID
    kind: ItemKind
    role: ResolutionRole
    name: str
    outcome: Outcome
    entity_id: uuid.UUID | None = None
    entity_name: str | None = None
    reason: UnresolvedReason | None = None
    candidates: tuple[Candidate, ...] = ()
    follow_up: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisStep:
    entity: EntityRead
    response: AnalysisResponse
    intelligence: EntityIntelligence


@dataclass
class ProcessResult:
    evidence: EvidenceStep
    extraction: ExtractionStep | None = None
    resolutions: list[ResolutionRecord] = field(default_factory=list)
    analyses: list[AnalysisStep] = field(default_factory=list)
    analysis_requested: bool = True

    @property
    def newly_attributed_entities(self) -> list[uuid.UUID]:
        seen: dict[uuid.UUID, None] = {}
        for record in self.resolutions:
            if (
                record.role in RELATABLE_ROLES
                and record.entity_id is not None
                and record.outcome in {"resolved", "created"}
            ):
                seen.setdefault(record.entity_id, None)
        return list(seen)


async def ingest(
    services: Services,
    *,
    content: str,
    meta: EvidenceMetadata,
    force_new: bool,
) -> EvidenceStep:
    """Persist the note, or recognise that it is already on file.

    The duplicate check is a convenience for a human re-running a command, not a
    uniqueness rule: the store still accepts identical content from two sources,
    because that genuinely is two observations. What it refuses to do is create a
    second record by accident when somebody runs the same command twice.
    """
    if not force_new:
        existing = await services.evidence.find_by_content(content)
        if len(existing) == 1:
            return EvidenceStep(evidence=existing[0], reused=True)
        if len(existing) > 1:
            ids = ", ".join(str(item.id) for item in existing)
            raise CliUsageError(
                f"this content is already on file {len(existing)} times ({ids}). "
                f"Which one is meant is a judgement, so nothing is assumed: pass "
                f"--force-new to record another observation, or work from one of "
                f"those ids directly."
            )

    created = await services.evidence.create(
        EvidenceCreate(
            content=content,
            source_type=meta.source_type,
            source_label=meta.source_label,
            source_url=meta.source_url,
            captured_at=meta.captured_at,
            metadata=dict(meta.metadata),
        )
    )
    return EvidenceStep(evidence=created, reused=False)


async def resolve_items(
    services: Services,
    *,
    items: Sequence[ExtractionItemRead],
    source_label: str | None,
    decider: Decider,
) -> list[ResolutionRecord]:
    """Attempt a resolution for every item that offers a name, asking when unsure."""
    ordered = sorted(
        (item for item in items if item.kind in RESOLVABLE),
        key=lambda item: (KIND_ORDER[ItemKind(item.kind)], item.position),
    )
    return [
        await _resolve_one(services, item=item, source_label=source_label, decider=decider)
        for item in ordered
    ]


async def _resolve_one(
    services: Services,
    *,
    item: ExtractionItemRead,
    source_label: str | None,
    decider: Decider,
) -> ResolutionRecord:
    kind = ItemKind(item.kind)
    role = RESOLVABLE[kind]
    name = matchable_name(kind, item.payload)

    if not name or not name.strip():
        return ResolutionRecord(
            extraction_item_id=item.id,
            kind=kind,
            role=role,
            name="",
            outcome="unnamed",
            reason=UnresolvedReason.NO_MATCHABLE_NAME,
        )

    try:
        outcome = await services.entities.resolve(item.id, ResolutionCreate(role=role))
    except ResolutionAlreadyExists:
        return await _describe_existing(services, item=item, kind=kind, role=role, name=name)

    if outcome.status == "resolved" and outcome.resolution is not None:
        entity = await services.entities.get_entity(outcome.resolution.entity_id)
        return ResolutionRecord(
            extraction_item_id=item.id,
            kind=kind,
            role=role,
            name=name,
            outcome="resolved",
            entity_id=entity.id,
            entity_name=entity.canonical_name,
        )

    if outcome.reason is UnresolvedReason.NO_MATCHABLE_NAME:
        return ResolutionRecord(
            extraction_item_id=item.id,
            kind=kind,
            role=role,
            name=name,
            outcome="unnamed",
            reason=outcome.reason,
        )

    candidates: tuple[Candidate, ...] = tuple(
        [
            Candidate.from_entity(await services.entities.get_entity(entity_id))
            for entity_id in outcome.candidate_entity_ids
        ]
    )
    entity_type = _mention_type(kind, item)

    declined = ResolutionRecord(
        extraction_item_id=item.id,
        kind=kind,
        role=role,
        name=name,
        outcome="declined",
        reason=outcome.reason,
        candidates=candidates,
        follow_up=_follow_up(
            item_id=item.id,
            role=role,
            name=name,
            entity_type=entity_type,
            candidates=candidates,
        ),
    )

    # An entity can only be created from something that states its own type, and
    # only an `entity_mention` does. `claimed_by = "Engineer at DevGuild"` names
    # nobody and says nothing about what kind of thing it is, so with no candidate
    # to choose from there is no question worth asking - the only answer available
    # would be "skip". This single condition is what stops a job title from
    # becoming a person; there is deliberately not a second flag saying the same
    # thing, because two guards for one rule are two things free to disagree.
    if not candidates and entity_type is None:
        return declined

    answer = decider.decide(
        Question(
            name=name,
            entity_type=entity_type or EntityType.OTHER,
            grounding_quote=item.grounding_quote,
            source_label=source_label,
            extraction_item_id=item.id,
            role=role.value,
            candidates=candidates,
        )
    )

    if isinstance(answer, Skip):
        return declined

    if isinstance(answer, CreateEntity):
        if entity_type is None:
            return declined
        entity = await services.entities.create_entity(
            EntityCreate(
                entity_type=entity_type,
                canonical_name=answer.canonical_name,
                disambiguator=answer.disambiguator,
            )
        )
        note = "created via pie process"
        chosen = "created"
    elif isinstance(answer, UseEntity):
        entity = await services.entities.get_entity(answer.entity_id)
        note = "chosen via pie process"
        chosen = "resolved"
    else:
        raise AssertionError(f"unhandled decision: {answer!r}")

    asserted = await services.entities.resolve(
        item.id, ResolutionCreate(role=role, entity_id=entity.id, note=note)
    )
    if asserted.status != "resolved":
        return declined
    return ResolutionRecord(
        extraction_item_id=item.id,
        kind=kind,
        role=role,
        name=name,
        outcome="created" if chosen == "created" else "resolved",
        entity_id=entity.id,
        entity_name=entity.canonical_name,
    )


async def _describe_existing(
    services: Services,
    *,
    item: ExtractionItemRead,
    kind: ItemKind,
    role: ResolutionRole,
    name: str,
) -> ResolutionRecord:
    """Report the decision a human already made, without touching it."""
    history = await services.entities.list_resolutions(item.id)
    active = next(
        (row for row in history.items if row.is_active and row.role is role),
        None,
    )
    entity_name: str | None = None
    if active is not None:
        entity_name = (await services.entities.get_entity(active.entity_id)).canonical_name
    return ResolutionRecord(
        extraction_item_id=item.id,
        kind=kind,
        role=role,
        name=name,
        outcome="already_resolved",
        entity_id=active.entity_id if active else None,
        entity_name=entity_name,
    )


def _mention_type(kind: ItemKind, item: ExtractionItemRead) -> EntityType | None:
    """A mention states its own type; an attribution string states nothing."""
    if kind is not ItemKind.ENTITY_MENTION:
        return None
    raw = item.payload.get("entity_type")
    return EntityType(raw) if isinstance(raw, str) else None


def _follow_up(
    *,
    item_id: uuid.UUID,
    role: ResolutionRole,
    name: str,
    entity_type: EntityType | None,
    candidates: tuple[Candidate, ...],
) -> tuple[str, ...]:
    """The exact commands that would settle this, for the unattended case."""
    if candidates:
        return tuple(
            f"pie resolve {item_id} --role {role.value} --entity {candidate.entity_id}"
            f"  # {candidate.canonical_name}"
            for candidate in candidates
        )
    if entity_type is not None:
        return (
            f'pie entity new --type {entity_type.value} --name "{name}"',
            f"pie resolve {item_id} --role {role.value}",
        )
    return (
        f'# "{name}" names nobody on file. Create the identity deliberately, then:',
        f"pie resolve {item_id} --role {role.value}",
    )


async def analyse(services: Services, entity_ids: Sequence[uuid.UUID]) -> list[AnalysisStep]:
    """Derive relations for each entity that gained an attributed claim."""
    steps: list[AnalysisStep] = []
    for entity_id in entity_ids:
        response = await services.intelligence.analyze(entity_id)
        steps.append(
            AnalysisStep(
                entity=await services.entities.get_entity(entity_id),
                response=response,
                intelligence=await services.intelligence.get_intelligence(entity_id),
            )
        )
    return steps


async def process(
    services: Services,
    *,
    content: str,
    meta: EvidenceMetadata,
    force_new: bool,
    analyze: bool,
    decider: Decider,
) -> ProcessResult:
    """Ingest, extract, resolve, analyse. Each stage commits before the next runs."""
    evidence = await ingest(services, content=content, meta=meta, force_new=force_new)
    result = ProcessResult(evidence=evidence, analysis_requested=analyze)

    extraction = await services.extraction.extract(evidence.evidence.id)
    result.extraction = ExtractionStep(
        runs=tuple(extraction.runs),
        reused=extraction.reused,
        complete=extraction.complete,
        uncovered=tuple(extraction.uncovered),
    )

    result.resolutions = await resolve_items(
        services,
        items=extraction.items,
        source_label=evidence.evidence.source_label,
        decider=decider,
    )

    if analyze:
        result.analyses = await analyse(services, result.newly_attributed_entities)
    return result
