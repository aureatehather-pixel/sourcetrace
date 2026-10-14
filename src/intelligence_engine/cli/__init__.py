"""`pie` — the command-line way to actually use this system.

The engine reached M4 with no interface but `curl`. Putting one note through the
full loop meant roughly eight requests with four ids copied by hand between them,
which is why "sustained real usage" kept not happening: the bottleneck was never
the engine, it was the eight-step ritual in front of it.

So there is one happy-path command, `pie process <file>`, that runs the normal
M1-M4 flow end to end, and a granular command per stage for inspection, manual
control and reruns. The granular commands are not a fallback — a rejected
relation, a corrected resolution and a re-extraction all need them.

What the happy path deliberately does not do is hide a judgement. Where the
deterministic matcher declines, `process` asks a human or reports the exact
command that would settle it. Nothing in this package creates an entity, merges
an identity, or asserts a relation that the services would not have asserted on
their own.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

from intelligence_engine.cli.args import parse_args
from intelligence_engine.cli.commands import (
    AnalyzeCommand,
    CliUsageError,
    Command,
    EntityListCommand,
    EntityNewCommand,
    ExtractCommand,
    IngestCommand,
    PacketCommand,
    ProcessCommand,
    RejectCommand,
    ResolveCommand,
    ShowCommand,
    ValidateCommand,
)
from intelligence_engine.cli.context import Services, load_settings, open_services, run
from intelligence_engine.cli.decisions import AlwaysSkip, Decider, Prompt
from intelligence_engine.core.errors import ApplicationError
from intelligence_engine.entities.schemas import (
    EntityCreate,
    EntityType,
    ResolutionCreate,
    ResolutionRole,
)
from intelligence_engine.intelligence.schemas import RelationReject

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def main(argv: list[str] | None = None) -> int:
    """Parse, dispatch, and translate failures into an exit status.

    Every error an operator can cause is a message and a status, never a
    traceback: a stack trace tells them about this program, and they asked about
    their evidence.
    """
    try:
        command = parse_args(sys.argv[1:] if argv is None else argv)
    except CliUsageError as error:
        _write(sys.stderr, f"pie: {error}\n")
        return EXIT_USAGE

    _configure_logging()

    try:
        return run(_dispatch(command))
    except CliUsageError as error:
        _write(sys.stderr, f"pie: {error}\n")
        return EXIT_USAGE
    except ApplicationError as error:
        _write(sys.stderr, f"pie: {error}\n")
        return EXIT_FAILED
    except KeyboardInterrupt:
        _write(sys.stderr, "\npie: interrupted. Work already committed is kept.\n")
        return EXIT_FAILED


async def _dispatch(command: Command) -> int:
    from intelligence_engine.cli import render
    from intelligence_engine.cli.pipeline import process
    from intelligence_engine.cli.source import read_source

    settings = load_settings()

    async with open_services(settings) as services:
        if isinstance(command, ProcessCommand):
            document = read_source(command.path, command.meta)
            result = await process(
                services,
                content=document.content,
                meta=document.meta,
                force_new=command.force_new,
                analyze=command.analyze,
                decider=_decider(command.interactive),
            )
            return _emit(
                command.as_json,
                text=render.render_process(result),
                payload=render.process_payload(result),
            )

        if isinstance(command, IngestCommand):
            from intelligence_engine.cli.pipeline import ingest

            document = read_source(command.path, command.meta)
            step = await ingest(
                services,
                content=document.content,
                meta=document.meta,
                force_new=command.force_new,
            )
            verb = "already on file" if step.reused else "ingested"
            return _emit(
                command.as_json,
                text=f"{verb}  {step.evidence.id}",
                payload={"id": step.evidence.id, "reused": step.reused},
            )

        if isinstance(command, ExtractCommand):
            response = await services.extraction.extract(command.evidence_id)
            return _emit(
                command.as_json,
                text=("reused " if response.reused else "extracted ")
                + render.render_runs(
                    response.runs, complete=response.complete, uncovered=response.uncovered
                ),
                payload=response.model_dump(),
            )

        if isinstance(command, EntityNewCommand):
            entity = await services.entities.create_entity(
                EntityCreate(
                    entity_type=EntityType(command.entity_type),
                    canonical_name=command.canonical_name,
                    disambiguator=command.disambiguator,
                )
            )
            return _emit(
                command.as_json,
                text=render.render_entity(entity),
                payload=entity.model_dump(),
            )

        if isinstance(command, EntityListCommand):
            page = await services.entities.list_entities(
                limit=command.limit or settings.default_page_size,
                cursor=None,
                entity_type=EntityType(command.entity_type) if command.entity_type else None,
            )
            return _emit(
                command.as_json,
                text=render.render_entities(page),
                payload=page.model_dump(),
            )

        if isinstance(command, ResolveCommand):
            outcome = await services.entities.resolve(
                command.extraction_item_id,
                ResolutionCreate(
                    role=ResolutionRole(command.role),
                    entity_id=command.entity_id,
                    note=command.note,
                ),
            )
            return _emit(
                command.as_json,
                text=render.render_resolution(outcome),
                payload=outcome.model_dump(),
            )

        if isinstance(command, AnalyzeCommand):
            analysis = await services.intelligence.analyze(command.entity_id)
            return _emit(
                command.as_json,
                text=render.render_analysis(analysis),
                payload=analysis.model_dump(),
            )

        if isinstance(command, RejectCommand):
            relation = await services.intelligence.reject_relation(
                command.relation_id, RelationReject(note=command.note)
            )
            return _emit(
                command.as_json,
                text=f"rejected  {relation.id}\n  it stays on record and will not be regenerated.",
                payload=relation.model_dump(),
            )

        if isinstance(command, PacketCommand):
            from intelligence_engine.packet.render import render_packet
            from intelligence_engine.packet.schemas import DEFAULT_PACKET_LIMIT

            packet = await services.packet.assemble(limit=command.limit or DEFAULT_PACKET_LIMIT)
            return _emit(
                command.as_json,
                text=render_packet(packet),
                payload=packet.model_dump(),
            )

        if isinstance(command, ValidateCommand):
            return await _validate(services, command)

        if isinstance(command, ShowCommand):
            return await _show(services, command)

        raise AssertionError(f"unhandled command: {command!r}")


async def _validate(services: Services, command: ValidateCommand) -> int:
    """Reject an answer whose citations do not bind, or show what they bound to.

    `InvalidCitation` is an `ApplicationError`, so a rejection reaches `main` and
    becomes a message plus exit status 1. That is deliberate: the gate has to be
    usable from a script, and a rejected answer must not exit 0.

    **The packet this validates against is the whole corpus unless told
    otherwise.** This call used to be `assemble()`, inheriting
    `DEFAULT_PACKET_LIMIT`, and `pie packet --limit 1000` had no counterpart
    here — so an answer written against 496 items was checked against 200. Two
    consequences, and the second is the dangerous one: a citation of an item past
    the cut was *falsely rejected* as an unresolvable handle, and uniqueness was
    computed over a prefix, so a fragment colliding only outside it was *falsely
    accepted*. A validator that accepts what it exists to reject is worse than no
    validator.

    Assembling the corpus whole is the direction that cannot under-report a
    collision: uniqueness over a superset is strictly stricter, and handles are
    positional over a total order (`list_items` sorts by evidence arrival, id,
    window start, position), so truncation is a stable prefix and `E44` names the
    same item in every packet large enough to contain it. Resolution therefore
    cannot drift with the limit; only the collision set grows.

    If the corpus outgrows `MAX_PACKET_LIMIT` there is no whole-corpus packet to
    assemble, and this refuses rather than quietly checking a prefix under the
    same verdict word. `--limit` remains available for the legitimate case —
    reproducing the packet an answer was actually written against — and the audit
    output names the scope either way, so a narrowed check is never mistaken for
    a complete one.
    """
    from intelligence_engine.packet.citations import validate_bound_answer
    from intelligence_engine.packet.errors import CorpusExceedsPacketLimit
    from intelligence_engine.packet.render import render_audit
    from intelligence_engine.packet.schemas import MAX_PACKET_LIMIT

    try:
        answer = command.path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise CliUsageError(f"no such file: {command.path}") from error
    except OSError as error:
        raise CliUsageError(f"could not read {command.path}: {error}") from error
    except UnicodeDecodeError as error:
        raise CliUsageError(f"{command.path} is not UTF-8 text") from error

    packet = await services.packet.assemble(limit=command.limit or MAX_PACKET_LIMIT)
    if command.limit is None and packet.truncated:
        raise CorpusExceedsPacketLimit(
            f"{packet.total_items_available} grounded items are stored and a packet holds "
            f"at most {MAX_PACKET_LIMIT}, so the whole corpus cannot be assembled. "
            "Unique binding is a packet-wide property; checking a prefix would compute it "
            "over a subset and could accept a fragment that collides outside it. "
            "Pass --limit to validate against the packet the answer was written against, "
            "accepting that the guarantee is then scoped to those items."
        )

    audit = validate_bound_answer(answer, packet)
    return _emit(
        command.as_json,
        text=render_audit(audit, packet),
        payload={
            "passed": True,
            "packet_items": len(packet.items),
            "total_items_available": packet.total_items_available,
            "packet_truncated": packet.truncated,
            "bound_citations": [
                {"item_id": citation.item_id, "quote": citation.quote} for citation in audit.bound
            ],
            "cited": list(audit.cited),
            "uncited": list(audit.uncited),
        },
    )


async def _show(services: Services, command: ShowCommand) -> int:
    from intelligence_engine.cli import render

    if command.target == "evidence":
        evidence = await services.evidence.get(command.target_id)
        return _emit(
            command.as_json, text=render.render_evidence(evidence), payload=evidence.model_dump()
        )

    if command.target == "items":
        runs = await services.extraction.list_runs(command.target_id)
        succeeded = [run for run in runs.items if run.status == "succeeded"]
        if not succeeded:
            raise CliUsageError(
                f"no successful extraction for evidence {command.target_id}; "
                f"run `pie extract {command.target_id}` first"
            )
        return _emit(
            command.as_json,
            text=render.render_items(succeeded[0]),
            payload=succeeded[0].model_dump(),
        )

    if command.target == "entity":
        profile = await services.entities.get_profile(command.target_id)
        return _emit(
            command.as_json, text=render.render_profile(profile), payload=profile.model_dump()
        )

    intelligence = await services.intelligence.get_intelligence(command.target_id)
    return _emit(
        command.as_json,
        text=render.render_intelligence(intelligence),
        payload=intelligence.model_dump(),
    )


def _configure_logging() -> None:
    """Warnings to stderr, plainly.

    Deliberately *not* `core.logging.configure_logging`: that installs the JSON
    formatter on stdout, which is right for a server whose stdout is a log stream
    and wrong here, where stdout is the answer to a question and `--json` must
    stay parseable. Anything below WARNING is noise to someone reading one result.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("pie: %(levelname)s %(message)s"))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(logging.WARNING)


def _decider(interactive: bool) -> Decider:
    """A terminal is required to ask a question; a pipe cannot answer one."""
    if interactive and sys.stdin.isatty():
        return Prompt()
    return AlwaysSkip()


def _emit(as_json: bool, *, text: str, payload: object) -> int:
    from intelligence_engine.cli import render

    _write(sys.stdout, (render.to_json(payload) if as_json else text) + "\n")
    return EXIT_OK


def _write(stream: TextIO, text: str) -> None:
    """Write without letting a legacy console code page kill a finished command."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = stream.encoding or "ascii"
        stream.write(text.encode(encoding, errors="replace").decode(encoding))
    stream.flush()


__all__ = ["main"]
