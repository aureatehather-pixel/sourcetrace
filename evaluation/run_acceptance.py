"""Real-data acceptance for adaptive bounded extraction.

Runs one unaltered source through the *production* workflow — the same
`EvidenceService` ingest then `ExtractionService.extract` the CLI uses — against
an isolated database. Nothing is pre-split, no constant is tuned, and the
provider is the real one built from settings.

Usage:

    uv run python -m evaluation.run_acceptance <path-to-source.md> <database-name> [captured-at]

`captured-at` is when the source was recorded, with a timezone offset. Omitted,
the evidence is stored with no capture time, which honestly says nobody
recorded one; it is never inferred.

## Why this file is in the repository

It used to live in a session-scoped scratch directory and write its report next
to itself, and that combination lost a result.

The accepted Event 2 run — 60,241 characters, 11 leaf intervals, all succeeded,
exact `[0,60241)` coverage, 348 items, 76,614 input and 43,324 output tokens —
was produced on 2026-08-30, and 54 seconds later its database was deliberately
dropped to clear a duplicate-evidence artefact this harness had caused. The
clean re-run that replaced it then stopped on `provider_unavailable` partway
through. For a while the only surviving record of a paid, unreproducible run was
a JSON file in a temp directory nobody had decided to keep, and the database
that remained looked like a failure.

So the report now goes to `ACCEPTANCE_DIR`, which is in the repository tree and
in `.gitignore`. Ignored, because a report carries every extracted item and its
verbatim grounding quote and is therefore source-derived private data. Not
disposable, because the run cost money and cannot be repeated for free. Those
are two different properties and the previous arrangement confused them.

This is a harness, not product code. It is deliberately one file with no
abstraction over sources, providers or report shapes: the thing worth making
durable was the *artefact*, not a framework for producing artefacts.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

#: The repository root, derived from this file rather than hard-coded. The
#: previous copy carried an absolute path to one machine's checkout, which is
#: the other half of why it could not live anywhere but the session that wrote
#: it.
REPO = Path(__file__).resolve().parent.parent

#: Where a completed report is written. In the tree, out of git — see the module
#: docstring. Created on demand so a clean checkout needs no setup step.
ACCEPTANCE_DIR = REPO / "evaluation" / "acceptance"

if len(sys.argv) not in (3, 4):  # pragma: no cover - argument handling for a manual script
    raise SystemExit(__doc__)

SOURCE = Path(sys.argv[1])
DB_NAME = sys.argv[2]
CAPTURED_AT = sys.argv[3] if len(sys.argv) == 4 else None
ADMIN_URL = "postgresql+psycopg://pie:pie_local_dev@localhost:5433/pie"
TARGET_URL = f"postgresql+psycopg://pie:pie_local_dev@localhost:5433/{DB_NAME}"

# Isolated state, set before anything reads configuration.
os.environ["DATABASE_URL"] = TARGET_URL
os.environ["ENVIRONMENT"] = "local"

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from intelligence_engine.core.config import Settings  # noqa: E402
from intelligence_engine.core.platform import run_async  # noqa: E402
from intelligence_engine.db.session import create_engine, create_session_factory  # noqa: E402
from intelligence_engine.entities.schemas import ResolutionRole  # noqa: E402
from intelligence_engine.evidence.schemas import EvidenceCreate  # noqa: E402
from intelligence_engine.evidence.service import EvidenceService  # noqa: E402
from intelligence_engine.extraction.factory import build_provider  # noqa: E402
from intelligence_engine.extraction.service import ExtractionService  # noqa: E402
from intelligence_engine.extraction.windowing import coverage, plan_windows  # noqa: E402
from intelligence_engine.intelligence.candidates import (  # noqa: E402
    RelatableItem,
    generate_candidates,
)
from intelligence_engine.packet.service import PacketService  # noqa: E402

SUBDIVISIONS: list[dict[str, object]] = []


class _Capture(logging.Handler):
    """Records subdivision events so the tree can be reported exactly."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == "extraction.subdivided":
            SUBDIVISIONS.append(
                {
                    "parent": (record.window_start, record.window_end),  # type: ignore[attr-defined]
                    "depth": record.depth,  # type: ignore[attr-defined]
                    "children": record.children,  # type: ignore[attr-defined]
                    "reason": record.error_code,  # type: ignore[attr-defined]
                }
            )


async def _create_database() -> bool:
    """Create the isolated database, or keep an existing one so a retry resumes.

    Resuming is the point: a run that failed partway leaves its successful
    intervals committed, and re-running must pay only for what is still missing.
    Dropping the database on every attempt would hide exactly that.

    This never drops. The one time a database holding a completed run was
    dropped by hand, the run survived only because its report happened to still
    be on disk.
    """
    engine = create_async_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            existing = (
                await connection.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": DB_NAME}
                )
            ).scalar()
            if existing:
                return True
            await connection.execute(text(f'CREATE DATABASE "{DB_NAME}"'))
            return False
    finally:
        await engine.dispose()


def _migrate() -> None:
    """Bring the isolated database to head before anything reads it.

    Sync, and called from sync code, because it blocks. Running it inside the
    async body would stall the loop psycopg is using — the linter says so and it
    is right; a blocking subprocess is simply not async work.
    """
    env = dict(os.environ, DATABASE_URL=TARGET_URL)
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(REPO),
        env=env,
        check=True,
        capture_output=True,
    )


async def _extract(resumed: bool, content: str) -> dict[str, object]:
    """The whole acceptance run. Returns the report; writing it is the caller's job.

    Separated so nothing that touches the filesystem happens inside the event
    loop, and — more usefully — so the report exists as a value before anything
    can fail while persisting it.
    """
    logging.getLogger("intelligence_engine.extraction.service").addHandler(_Capture())
    logging.getLogger("intelligence_engine.extraction.service").setLevel(logging.INFO)

    settings = Settings()
    report: dict[str, object] = {
        "resumed_existing_database": resumed,
        "database": DB_NAME,
        "source": SOURCE.name,
        "source_chars": len(content),
        "model": settings.extraction_model,
        "window_max_chars": settings.extraction_window_max_chars,
        "max_output_tokens": settings.extraction_max_output_tokens,
    }

    engine = create_engine(TARGET_URL)
    factory = create_session_factory(engine)
    try:
        # The production ingest path: recognise content already on file rather
        # than recording a second observation of it. The duplicate check lives
        # in the CLI, not in EvidenceService.create, so a harness that calls
        # create directly manufactures a second logical source and makes the
        # packet honestly report two. That is exactly what happened on the first
        # accepted Event 2 run, and cleaning it up is what cost the database.
        async with factory() as session:
            service_evidence = EvidenceService(session)
            existing = await service_evidence.find_by_content(content)
            if len(existing) == 1:
                evidence = existing[0]
                report["evidence_reused"] = True
            elif not existing:
                evidence = await service_evidence.create(
                    EvidenceCreate.model_validate(
                        {
                            "content": content,
                            "source_type": "conversation_export",
                            "source_label": SOURCE.stem,
                            "captured_at": CAPTURED_AT,
                        }
                    )
                )
                report["evidence_reused"] = False
            else:
                raise SystemExit(f"content is on file {len(existing)} times; refusing to guess")
        report["evidence_id"] = str(evidence.id)

        plan = plan_windows(content, max_chars=settings.extraction_window_max_chars)
        report["initial_intervals"] = [(w.start, w.end, w.end - w.start) for w in plan]

        provider = build_provider(settings)
        report["provider"] = provider.identity.provider
        report["requested_model"] = provider.identity.model
        async with factory() as session:
            service = ExtractionService(
                session, provider, window_max_chars=settings.extraction_window_max_chars
            )
            result = await service.extract(evidence.id)

        report["complete"] = result.complete
        report["uncovered"] = result.uncovered
        report["reused"] = result.reused
        report["source_length"] = result.source_length
        report["leaf_calls"] = len(result.runs)
        report["total_items"] = sum(len(run.items) for run in result.runs)

        kinds: dict[str, int] = {}
        for run in result.runs:
            for item in run.items:
                kinds[str(item.kind)] = kinds.get(str(item.kind), 0) + 1
        report["kind_distribution"] = kinds

        report["leaves"] = [
            {
                "interval": (run.window_start, run.window_end),
                "chars": run.window_end - run.window_start,
                "items": len(run.items),
                "input_tokens": run.input_tokens,
                "output_tokens": run.output_tokens,
            }
            for run in result.runs
        ]
        report["subdivisions"] = SUBDIVISIONS

        # Every attempt, including the failures that caused subdivision.
        async with factory() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT window_start, window_end, status, error_code, "
                        "input_tokens, output_tokens FROM extraction_runs "
                        "ORDER BY window_start, window_end, created_at"
                    )
                )
            ).all()
        report["all_attempts"] = [
            {
                "interval": (row[0], row[1]),
                "chars": row[1] - row[0],
                "status": row[2],
                "error_code": row[3],
                "input_tokens": row[4],
                "output_tokens": row[5],
            }
            for row in rows
        ]
        report["stored_attempts"] = len(rows)
        report["failed_attempts"] = [
            {"interval": (row[0], row[1]), "error_code": row[3]}
            for row in rows
            if row[2] == "failed"
        ]

        covered = coverage(
            ((run.window_start, run.window_end) for run in result.runs),
            length=len(content),
        )
        report["tiles_exactly"] = covered.is_complete
        report["covered_chars"] = covered.covered_chars
        report["gaps"] = covered.gaps
        report["overlaps"] = covered.overlaps

        async with factory() as session:
            packet = await PacketService(session).assemble(limit=1000)
        report["packet_source_count"] = len(packet.sources)
        report["packet_item_count"] = len(packet.items)
        report["packet_source_complete"] = [source.complete for source in packet.sources]
        report["packet_evidence_ids"] = [str(source.evidence_id) for source in packet.sources]

        # M4 same-source suppression, over every item this source produced.
        items = [
            RelatableItem(
                item_id=item.id,
                kind=item.kind,
                role=ResolutionRole.CLAIMED_BY,
                text=item.grounding_quote,
                stance="asserted",
                claimed_by="someone",
                subject=None,
                grounding_quote=item.grounding_quote,
                extraction_run_id=run.id,
                evidence_id=evidence.id,
                resolution_id=evidence.id,
                source_label=SOURCE.stem,
                captured_at=None,
                evidence_created_at=evidence.created_at,
                position=item.position,
            )
            for run in result.runs
            for item in run.items
        ]
        selection = generate_candidates(items, max_pairs=500)
        report["m4_relatable_items"] = len(items)
        report["m4_theoretical_pairs"] = selection.theoretical_pairs
        report["m4_candidate_pairs"] = len(selection.pairs)

        # The full item list, for provenance spot-checking. This is what makes
        # the report a record of the run rather than a summary of it: every
        # quote here can be checked against the source without paying again.
        report["items"] = [
            {
                "interval": (run.window_start, run.window_end),
                "kind": str(item.kind),
                "payload": item.payload,
                "quote": item.grounding_quote,
            }
            for run in result.runs
            for item in run.items
        ]
    finally:
        await engine.dispose()

    return report


def main() -> None:
    """Create, migrate, run, then write the report — in that order, deliberately.

    The write is the last thing and it is unconditional. Everything above it can
    cost money; nothing above it is repeatable for free. A report that existed
    only as long as the process, or only as long as a temp directory, is how a
    completed run came to look like a failed one.
    """
    resumed = run_async(_create_database())
    _migrate()
    content = SOURCE.read_text(encoding="utf-8")

    report = run_async(_extract(resumed, content))

    ACCEPTANCE_DIR.mkdir(parents=True, exist_ok=True)
    out = ACCEPTANCE_DIR / f"acceptance_{SOURCE.stem}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    summary = {k: v for k, v in report.items() if k not in {"items", "leaves", "all_attempts"}}
    sys.stdout.write(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")
    sys.stdout.write(f"\nwrote {out}\n")


main()
