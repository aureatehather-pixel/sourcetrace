"""Turning results into something a person will actually read.

Pure functions from typed results to text. Nothing here queries, decides or
mutates, so the output format can be changed and tested without a database.

Two constraints shape the format:

**ASCII only.** A Windows console commonly runs a legacy code page, where
writing a tick or an em dash raises `UnicodeEncodeError` and takes the whole
command down at the last step, after the model calls have been paid for.
Decoration is not worth that, so the markers are ASCII.

**Every id is printed in full.** The ids are not decoration either: they are the
arguments to the next command. Truncating them to look tidy would mean going back
to the database to copy one out, which is the friction this CLI exists to remove.

The human format leads with what happened and shows provenance underneath it.
`--json` exists for scripting and emits the same facts without the formatting.
"""

from __future__ import annotations

import json
import textwrap
import uuid
from collections.abc import Iterable, Sequence
from typing import Any

from intelligence_engine.cli.pipeline import ProcessResult, ResolutionRecord
from intelligence_engine.entities.schemas import (
    EntityPage,
    EntityProfile,
    EntityRead,
    ResolutionOutcome,
)
from intelligence_engine.evidence.schemas import EvidenceRead
from intelligence_engine.extraction.schemas import ExtractionItemRead, ExtractionRunRead
from intelligence_engine.intelligence.schemas import (
    AnalysisResponse,
    DerivedRelation,
    EntityIntelligence,
)

WIDTH = 88

#: One ASCII marker per outcome, so a run can be skimmed down the left edge.
MARKERS: dict[str, str] = {
    "resolved": "[ok]",
    "created": "[new]",
    "already_resolved": "[==]",
    "declined": "[??]",
    "unnamed": "[--]",
}


def render_process(result: ProcessResult) -> str:
    lines: list[str] = []
    lines += _evidence_section(result)
    lines += _extraction_section(result)
    lines += _resolution_section(result)
    lines += _intelligence_section(result)
    return "\n".join(lines)


def _evidence_section(result: ProcessResult) -> list[str]:
    item = result.evidence.evidence
    verb = "already on file" if result.evidence.reused else "ingested"
    provenance = (
        " / ".join(part for part in (item.source_type, item.source_label) if part)
        or "no provenance recorded"
    )
    return [
        "EVIDENCE",
        f"  {verb}  {item.id}",
        f"  {provenance}",
        f"  {len(item.content)} characters, sha256 {item.content_hash[:16]}...",
        "",
    ]


def _extraction_section(result: ProcessResult) -> list[str]:
    if result.extraction is None:
        return []
    step = result.extraction
    runs = step.runs
    if not runs:
        return []
    verb = "reused" if step.reused else "extracted"
    items = step.items
    first = runs[0]

    input_tokens = sum(run.input_tokens or 0 for run in runs)
    output_tokens = sum(run.output_tokens or 0 for run in runs)
    tokens = (
        f"  {input_tokens} in / {output_tokens} out" if input_tokens and not step.reused else ""
    )

    lines = ["EXTRACTION"]
    if len(runs) > 1:
        # Intervals are mechanics, so they are reported as one source read in
        # several passes — never as several sources.
        lines.append(f"  {verb}  {len(runs)} intervals, one source")
    else:
        lines.append(f"  {verb}  run {first.id}")
    lines.append(f"  {len(items)} items: {_kind_counts(items)}")
    lines.append(
        f"  {first.extractor_name} {first.extractor_version} via {first.provider or 'n/a'}"
        f" {first.model or ''}{tokens}".rstrip()
    )
    if not step.complete:
        unread = ", ".join(f"[{start}:{end})" for start, end in step.uncovered)
        lines.append(f"  INCOMPLETE - {unread} never extracted; this source is not fully read")
    lines.append("")
    return lines


def _kind_counts(items: Sequence[ExtractionItemRead]) -> str:
    counts: dict[str, int] = {}
    for item in items:
        counts[str(item.kind)] = counts.get(str(item.kind), 0) + 1
    if not counts:
        return "none"
    return ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))


def _resolution_section(result: ProcessResult) -> list[str]:
    if not result.resolutions:
        return ["RESOLUTION", "  nothing in this note names an entity.", ""]

    lines = ["RESOLUTION"]
    for record in result.resolutions:
        lines += _resolution_line(record)
    open_questions = sum(1 for record in result.resolutions if record.outcome == "declined")
    if open_questions:
        lines.append(
            f"  {open_questions} left unresolved. Unresolved is a valid state; "
            f"the commands above settle one when you decide it should be."
        )
    lines.append("")
    return lines


def _resolution_line(record: ResolutionRecord) -> list[str]:
    marker = MARKERS[record.outcome]
    name = record.name or "(unnamed)"
    head = f"  {marker} {name}  [{record.role.value}]"

    if record.outcome in {"resolved", "created", "already_resolved"}:
        verb = {
            "resolved": "->",
            "created": "-> created",
            "already_resolved": "== already",
        }[record.outcome]
        return [f"{head}  {verb} {record.entity_name}  {record.entity_id}"]

    if record.outcome == "unnamed":
        return [f"{head}  names nobody; left unattributed"]

    reason = record.reason.value if record.reason else "unresolved"
    lines = [f"{head}  {reason}"]
    lines += [f"      {command}" for command in record.follow_up]
    return lines


def _intelligence_section(result: ProcessResult) -> list[str]:
    if not result.analysis_requested:
        return ["INTELLIGENCE", "  skipped (--no-analyze).", ""]
    if not result.analyses:
        return [
            "INTELLIGENCE",
            "  no entity gained an attributed claim in this run, so there was",
            "  nothing new to relate. Only claims attributed to an entity can",
            "  stand in a relation.",
            "",
        ]

    lines = ["INTELLIGENCE"]
    for step in result.analyses:
        telemetry = step.response.telemetry
        lines.append(f"  {step.entity.canonical_name}  {step.entity.id}")
        lines.append(
            f"    {telemetry.candidate_pairs} candidates from "
            f"{telemetry.theoretical_pairs} possible pairs; "
            f"{telemetry.reused_pairs} reused, {telemetry.classified_pairs} classified "
            f"in {telemetry.external_call_count} call(s)"
        )
        if not step.intelligence.derived_relations:
            lines.append("    no relation worth surfacing.")
        for relation in step.intelligence.derived_relations:
            lines += _relation_lines(relation, indent="    ")
        lines.append("")
    return lines


def _relation_lines(relation: DerivedRelation, *, indent: str) -> list[str]:
    lines = [
        f"{indent}{relation.relation.relation_type.value.upper()}  ({relation.reading})",
        f"{indent}  A: {_quote(relation.left.text)}",
        f"{indent}  B: {_quote(relation.right.text)}",
    ]
    if relation.relation.explanation:
        lines += _wrapped(f"derived: {relation.relation.explanation}", indent=f"{indent}  ")
    lines.append(f"{indent}  relation {relation.relation.id}")
    return lines


def render_intelligence(intelligence: EntityIntelligence) -> str:
    lines = [
        f"{intelligence.entity.canonical_name}  ({intelligence.entity.entity_type})"
        f"  {intelligence.entity.id}",
        "",
        f"GROUNDED CLAIMS ({len(intelligence.grounded_claims)})",
    ]
    for claim in intelligence.grounded_claims:
        lines += _wrapped(f"- {claim.claim}", indent="  ")
        lines.append(f"    stance {claim.stance or 'unstated'}  evidence {claim.evidence_id}")
    lines += ["", f"DERIVED RELATIONS ({len(intelligence.derived_relations)})"]
    if not intelligence.derived_relations:
        lines.append("  none.")
    for relation in intelligence.derived_relations:
        lines += _relation_lines(relation, indent="  ")
        lines.append("")
    return "\n".join(lines)


def render_profile(profile: EntityProfile) -> str:
    entity = profile.entity
    lines = [f"{entity.canonical_name}  ({entity.entity_type})  {entity.id}"]
    if entity.disambiguator:
        lines.append(f"  {entity.disambiguator}")
    lines += [
        f"  {profile.evidence_count} evidence item(s)",
        "",
        f"MENTIONS ({len(profile.mentions)})",
    ]
    for mention in profile.mentions:
        described = f" - {mention.described_as}" if mention.described_as else ""
        lines.append(f"  {mention.name_as_mentioned}{described}")
        lines.append(f'    "{mention.grounding_quote}"')
        lines.append(f"    evidence {mention.evidence_id}")
    lines += ["", f"ATTRIBUTED CLAIMS ({len(profile.attributed_claims)})"]
    for claim in profile.attributed_claims:
        lines += _wrapped(f"- {claim.claim}", indent="  ")
        lines.append(f"    stance {claim.stance or 'unstated'}")
        lines.append(f'    "{claim.grounding_quote}"')
        lines.append(f"    item {claim.extraction_item_id}")
    # Rendered under its own heading, never merged into the claims above. The
    # operator must be able to see at a glance what this entity SAID versus what
    # was OBSERVED about them; one combined list would erase that.
    lines += ["", f"OBSERVED ABOUT ({len(profile.subject_observations)})"]
    for observed in profile.subject_observations:
        lines += _wrapped(f"- {observed.observation}", indent="  ")
        lines.append(f"    subject as worded: {observed.subject or 'unstated'}")
        lines.append(f'    "{observed.grounding_quote}"')
        lines.append(f"    item {observed.extraction_item_id}")
    return "\n".join(lines)


def render_evidence(evidence: EvidenceRead) -> str:
    return "\n".join(
        [
            f"{evidence.id}",
            f"  source     {evidence.source_type or '-'} / {evidence.source_label or '-'}",
            f"  captured   {evidence.captured_at.isoformat() if evidence.captured_at else '-'}",
            f"  ingested   {evidence.created_at.isoformat()}",
            f"  sha256     {evidence.content_hash}",
            f"  metadata   {json.dumps(evidence.metadata, sort_keys=True)}",
            "",
            evidence.content,
        ]
    )


def render_items(run: ExtractionRunRead) -> str:
    lines = [
        f"run {run.id}  {run.status}  {run.extractor_name} {run.extractor_version}",
        f"{len(run.items)} item(s)",
        "",
    ]
    for item in run.items:
        lines.append(f"{item.id}  {item.kind}")
        lines.append(f"  {json.dumps(item.payload, sort_keys=True, ensure_ascii=False)}")
        lines.append(f'  quote: "{item.grounding_quote}"')
        lines.append("")
    return "\n".join(lines)


def render_runs(
    runs: Sequence[ExtractionRunRead],
    *,
    complete: bool,
    uncovered: Sequence[tuple[int, int]] = (),
) -> str:
    """Every interval of one source, in order, as one reading of one document.

    A source read in one pass renders exactly as `render_items` always did. A
    source read in several gets a header naming each interval by its character
    range and a loud trailer if any region went unread, because a partial
    extraction that reads like a whole one is the failure this reporting exists
    to prevent.

    Intervals are identified by range rather than by an ordinal, which is not
    only durable but more useful here: `[11375:17326)` locates the text in the
    source, while "window 3 of 11" locates it only within a plan the reader
    cannot see and that subdivision may already have changed.
    """
    if not runs:
        return "no extraction runs\n"

    windowed = len(runs) > 1
    blocks: list[str] = []
    for run in runs:
        header = f"interval [chars {run.window_start}:{run.window_end})\n" if windowed else ""
        blocks.append(header + render_items(run))

    total = sum(len(run.items) for run in runs)
    if windowed:
        blocks.append(f"{total} item(s) from {len(runs)} interval(s) of one source")
    if not complete:
        unread = ", ".join(f"[{start}:{end})" for start, end in uncovered)
        blocks.append(f"INCOMPLETE - {unread} never extracted; this source is not fully read")
    return "\n".join(blocks)


def render_entity(entity: EntityRead) -> str:
    suffix = f"  - {entity.disambiguator}" if entity.disambiguator else ""
    return f"{entity.id}  {entity.entity_type}  {entity.canonical_name}{suffix}"


def render_entities(page: EntityPage) -> str:
    if not page.items:
        return "no entities on file. Nothing creates them as a side effect; use `pie entity new`."
    lines = [render_entity(entity) for entity in page.items]
    if page.next_cursor:
        lines.append(f"(more; next cursor {page.next_cursor})")
    return "\n".join(lines)


def render_resolution(outcome: ResolutionOutcome) -> str:
    if outcome.status == "resolved" and outcome.resolution is not None:
        return (
            f"resolved  {outcome.resolution.id}\n"
            f"  item   {outcome.resolution.extraction_item_id}\n"
            f"  entity {outcome.resolution.entity_id}\n"
            f"  method {outcome.resolution.method.value}"
        )
    lines = [f"unresolved  {outcome.reason.value if outcome.reason else 'no reason given'}"]
    lines += [f"  candidate {entity_id}" for entity_id in outcome.candidate_entity_ids]
    lines.append("  unresolved is a valid outcome, not a failure.")
    return "\n".join(lines)


def render_analysis(response: AnalysisResponse) -> str:
    telemetry = response.telemetry
    return "\n".join(
        [
            f"entity {response.entity_id}",
            f"  classifier   {response.classifier_name} {response.classifier_version}",
            f"  model calls  {telemetry.external_call_count}"
            f" ({'spent' if response.performed_calls else 'all reused'})",
            f"  pairs        {telemetry.theoretical_pairs} possible ->"
            f" {telemetry.candidate_pairs} candidates ->"
            f" {telemetry.classified_pairs} classified,"
            f" {telemetry.reused_pairs} reused",
            f"  new          {telemetry.new_relations} stored,"
            f" {telemetry.new_meaningful_relations} worth surfacing",
        ]
    )


def to_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=_encode)


def process_payload(result: ProcessResult) -> dict[str, Any]:
    """The same facts as `render_process`, without the formatting."""
    return {
        "evidence": {
            "id": result.evidence.evidence.id,
            "reused": result.evidence.reused,
            "content_hash": result.evidence.evidence.content_hash,
        },
        "extraction": None
        if result.extraction is None
        else {
            "run_ids": [run.id for run in result.extraction.runs],
            "intervals": [(run.window_start, run.window_end) for run in result.extraction.runs],
            "reused": result.extraction.reused,
            "complete": result.extraction.complete,
            "uncovered": list(result.extraction.uncovered),
            "item_count": len(result.extraction.items),
        },
        "resolutions": [
            {
                "extraction_item_id": record.extraction_item_id,
                "kind": str(record.kind),
                "role": record.role.value,
                "name": record.name,
                "outcome": record.outcome,
                "entity_id": record.entity_id,
                "reason": record.reason.value if record.reason else None,
                "follow_up": list(record.follow_up),
            }
            for record in result.resolutions
        ],
        "analyses": [
            {
                "entity_id": step.entity.id,
                "performed_calls": step.response.performed_calls,
                "telemetry": step.response.telemetry.model_dump(),
                "relation_count": len(step.intelligence.derived_relations),
            }
            for step in result.analyses
        ],
    }


def _quote(text: str) -> str:
    return textwrap.shorten(text, width=WIDTH - 8, placeholder="...")


def _wrapped(text: str, *, indent: str) -> list[str]:
    return textwrap.wrap(
        text, width=WIDTH, initial_indent=indent, subsequent_indent=indent + "  "
    ) or [indent + text]


def _encode(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return list(value)
    return str(value)
