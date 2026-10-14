"""SourceTrace, end to end, on invented evidence.

    uv run python -m demo                              # disposable PostgreSQL via Docker
    uv run python -m demo --database-url postgresql+psycopg://...   # an EMPTY database

Every person, place and event below is invented. No API key is used and nothing
is paid for: the two model calls the system would make are replaced by scripted
responses, and *everything else is real* - the PostgreSQL schema and its
constraints, strict schema validation, the verbatim-quote check, the
conservative entity matcher, supersession, relation reuse, and the citation
gate. The scripted model is deliberately wrong at two points so that the checks
that catch it are shown working rather than described.

What the demo shows, in order:

1. Ingestion. Three notes are stored with provenance and a content fingerprint.
2. Extraction. A response that paraphrases its quote is rejected whole; a
   response whose quotes are verbatim is accepted.
3. Entity resolution. An exact, unique name match is asserted; an ambiguous one
   is declined; a manual link is corrected by supersession, keeping history.
4. Relations. A cross-note relation is derived; a trivial one is rejected by a
   reviewer; re-running reuses every settled pair and makes no model calls.
5. Citations. The gate accepts a verbatim, uniquely bound fragment and rejects
   a paraphrase and an ambiguous fragment - and it also accepts a citation that
   does not support its claim, which is the limit of what it can check.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from intelligence_engine.core.platform import run_async
from intelligence_engine.db.session import create_engine, create_session_factory
from intelligence_engine.entities.schemas import (
    EntityCreate,
    EntityType,
    ResolutionCreate,
    ResolutionRole,
    ResolutionSupersede,
)
from intelligence_engine.entities.service import EntityService
from intelligence_engine.evidence.schemas import EvidenceCreate
from intelligence_engine.evidence.service import EvidenceService
from intelligence_engine.extraction.errors import UngroundedItem
from intelligence_engine.extraction.fake_provider import FakeExtractionProvider
from intelligence_engine.extraction.provider import ProviderIdentity, ProviderResult, StopReason
from intelligence_engine.extraction.schemas import ExtractionItemRead, ItemKind
from intelligence_engine.extraction.service import ExtractionService
from intelligence_engine.intelligence.schemas import RelationReject, RelationType
from intelligence_engine.intelligence.service import IntelligenceService
from intelligence_engine.packet.citations import packet_handles, validate_bound_answer
from intelligence_engine.packet.errors import InvalidCitation
from intelligence_engine.packet.render import render_packet
from intelligence_engine.packet.service import PacketService

SessionFactory = async_sessionmaker[AsyncSession]
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- the synthetic evidence --------------------------------------------------
#
# Byte-identical to `examples/notes/`, as `pie process` would store them - the
# trailing newline included, because a note's body is kept exactly.

WORKSHOP = (
    "Dana Reyes told me at the Harbour Hall maker workshop that the quickest way to "
    "learn embedded systems is to rebuild a device you already own instead of starting "
    "from a kit. She said her first rebuild, a kitchen thermostat, took her a whole term.\n"
)
FOLLOW_UP = (
    "Follow-up call with Dana Reyes. She said the first month was almost entirely "
    "reading datasheets, not soldering. She added that most people who copy her "
    "approach give up during the first month.\n"
)
REVIEW = (
    "Design review. Alex Lee from the payments team signed off on the firmware update "
    "plan for the shop's card readers.\n"
)

ADVICE_QUOTE = "the quickest way to learn embedded systems is to rebuild a device you already own"
THERMOSTAT_QUOTE = "her first rebuild, a kitchen thermostat, took her a whole term"
READING_QUOTE = "the first month was almost entirely reading datasheets, not soldering"
GIVE_UP_QUOTE = "most people who copy her approach give up during the first month"


def _mention(name: str, quote: str, described_as: str | None = None) -> dict[str, Any]:
    return {
        "kind": "entity_mention",
        "data": {"name": name, "entity_type": "person", "described_as": described_as},
        "grounding": {"quote": quote},
    }


def _claim(claim: str, stance: str, quote: str) -> dict[str, Any]:
    return {
        "kind": "reported_claim",
        "data": {"claim": claim, "claimed_by": "Dana Reyes", "stance": stance},
        "grounding": {"quote": quote},
    }


WORKSHOP_ITEMS = {
    "items": [
        _mention("Dana Reyes", "Dana Reyes told me at the Harbour Hall maker workshop"),
        _claim(
            "The quickest way to learn embedded systems is to rebuild a device you "
            "already own instead of starting from a kit.",
            "recommended",
            ADVICE_QUOTE,
        ),
        _claim(
            "Her first rebuild, a kitchen thermostat, took her a whole term.",
            "asserted",
            THERMOSTAT_QUOTE,
        ),
    ]
}


def _follow_up_items(reading_quote: str) -> dict[str, Any]:
    return {
        "items": [
            _mention("Dana Reyes", "Follow-up call with Dana Reyes"),
            _claim(
                "The first month was almost entirely reading datasheets, not soldering.",
                "asserted",
                reading_quote,
            ),
            _claim(
                "Most people who copy her approach give up during the first month.",
                "asserted",
                GIVE_UP_QUOTE,
            ),
        ]
    }


#: What a model that "tidied up" the wording would return. Plausible, and wrong:
#: the source never says "mostly".
PARAPHRASED_FOLLOW_UP = _follow_up_items("the first month was mostly reading datasheets")
FOLLOW_UP_ITEMS = _follow_up_items(READING_QUOTE)

REVIEW_ITEMS = {"items": [_mention("Alex Lee", "Alex Lee from the payments team", "payments team")]}


# --- the scripted relation classifier ---------------------------------------


class ScriptedClassifier:
    """Answers each pair it is shown from the statements in it, not by position.

    Pair ids are per-batch labels the service assigns, so a script keyed on call
    order would pass for the wrong reason. This reads every pair back out of the
    prompt and decides from the two statements, as a model would.
    """

    _PAIR = re.compile(
        r"^(pair_\d+)\n  A:\n.*?statement: (.*?)\n.*?  B:\n.*?statement: (.*?)\n",
        re.MULTILINE | re.DOTALL,
    )

    def __init__(self, decide: Callable[[str, str], dict[str, Any]]) -> None:
        self._decide = decide
        self._identity = ProviderIdentity(provider="scripted", model="demo-classifier-1")
        self.calls = 0

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    async def extract(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> ProviderResult:
        self.calls += 1
        results = [
            {"pair_id": pair_id, **self._decide(a, b)}
            for pair_id, a, b in self._PAIR.findall(user_prompt)
        ]
        return ProviderResult(
            raw_output=json.dumps({"results": results}),
            provider=self._identity.provider,
            model=self._identity.model,
            input_tokens=None,
            output_tokens=None,
            stop_reason=StopReason.COMPLETE,
        )


def _decide(a: str, b: str) -> dict[str, Any]:
    texts = (a, b)
    if any("rebuild a device" in t for t in texts) and any("datasheets" in t for t in texts):
        return {
            "relation_type": "qualifies",
            "direction": "b_qualifies_a" if "datasheets" in b else "a_qualifies_b",
            "explanation": (
                "The advice to learn by rebuilding is qualified by what that actually "
                "involved at first: a month spent reading datasheets rather than building."
            ),
        }
    if any("thermostat" in t for t in texts) and any("give up" in t for t in texts):
        # Technically defensible and worth nothing - the kind of output a reviewer
        # rejects. The demo rejects it below.
        return {
            "relation_type": "reinforces",
            "explanation": "Both statements describe the approach taking a long time.",
        }
    # The contract forbids an explanation here: "no relation" needs no story.
    return {"relation_type": "no_meaningful_relation"}


# --- the walkthrough ---------------------------------------------------------


@dataclass
class DemoResult:
    """What happened, so a test can hold the demo to its own narration."""

    paraphrase_rejected: bool = False
    verbatim_accepted: bool = False
    dana_claims_resolved: int = 0
    ambiguous_declined: bool = False
    correction_history_rows: int = 0
    relation_types: list[str] = field(default_factory=list)
    rerun_model_calls: int = -1
    gate_accepted_verbatim: bool = False
    gate_rejected_paraphrase: bool = False
    gate_rejected_ambiguous: bool = False
    gate_accepted_unsupported_claim: bool = False


def _say(text: str = "") -> None:
    print(text)  # noqa: T201 - this module is a narrated walkthrough


def _step(title: str) -> None:
    _say()
    _say(f"== {title} " + "=" * max(0, 72 - len(title)))


async def _ingest(
    factory: SessionFactory, content: str, label: str, source_type: str, captured: str
) -> uuid.UUID:
    async with factory() as session:
        evidence = await EvidenceService(session).create(
            EvidenceCreate(
                content=content,
                source_type=source_type,
                source_label=label,
                captured_at=dt.datetime.fromisoformat(captured),
            )
        )
    _say(f"stored  {label!r:34} sha256 {evidence.content_hash[:16]}...")
    return evidence.id


async def _extract(
    factory: SessionFactory, evidence_id: uuid.UUID, payload: dict[str, Any]
) -> list[ExtractionItemRead]:
    async with factory() as session:
        response = await ExtractionService(
            session, FakeExtractionProvider(payload=payload)
        ).extract(evidence_id)
    return [item for run in response.runs for item in run.items]


async def walkthrough(factory: SessionFactory) -> DemoResult:
    result = DemoResult()

    _step("1. Ingest raw evidence, with provenance")
    workshop = await _ingest(
        factory,
        WORKSHOP,
        "Harbour Hall maker workshop",
        "conversation_note",
        "2026-03-02T18:30:00+00:00",
    )
    follow_up = await _ingest(
        factory, FOLLOW_UP, "Follow-up call", "conversation_note", "2026-04-15T09:00:00+00:00"
    )
    review = await _ingest(
        factory, REVIEW, "Design review", "meeting_note", "2026-04-20T14:00:00+00:00"
    )
    _say("Raw evidence is immutable. Everything below is derived from it and stored apart.")

    _step("2. Schema-validated extraction, with verbatim quote checks")
    try:
        await _extract(factory, follow_up, PARAPHRASED_FOLLOW_UP)
    except UngroundedItem as error:
        result.paraphrase_rejected = True
        _say(f"REJECTED  model output quoting a paraphrase: {error}")
        _say("          The whole run is recorded as failed. No partial data is kept.")

    workshop_items = await _extract(factory, workshop, WORKSHOP_ITEMS)
    follow_up_items = await _extract(factory, follow_up, FOLLOW_UP_ITEMS)
    review_items = await _extract(factory, review, REVIEW_ITEMS)
    result.verbatim_accepted = bool(workshop_items and follow_up_items and review_items)
    for item in (*workshop_items, *follow_up_items, *review_items):
        _say(f"accepted  {item.kind.value:15} quote: {item.grounding_quote!r}")
    _say("A verbatim quote proves the words are in the source. It does not prove the")
    _say("extracted statement is a faithful reading of them; that stays a human judgement.")

    _step("3. Conservative entity resolution, with manual correction")
    async with factory() as session:
        entities = EntityService(session)
        dana = await entities.create_entity(
            EntityCreate(entity_type=EntityType.PERSON, canonical_name="Dana Reyes")
        )
        payments_alex = await entities.create_entity(
            EntityCreate(
                entity_type=EntityType.PERSON,
                canonical_name="Alex Lee",
                disambiguator="payments team",
            )
        )
        studio_alex = await entities.create_entity(
            EntityCreate(
                entity_type=EntityType.PERSON,
                canonical_name="Alex Lee",
                disambiguator="design studio",
            )
        )

    for item in (*workshop_items, *follow_up_items):
        role = (
            ResolutionRole.CLAIMED_BY
            if item.kind is ItemKind.REPORTED_CLAIM
            else ResolutionRole.MENTION
        )
        async with factory() as session:
            outcome = await EntityService(session).resolve(item.id, ResolutionCreate(role=role))
        if outcome.status == "resolved" and role is ResolutionRole.CLAIMED_BY:
            result.dana_claims_resolved += 1
    _say(
        f"resolved  {result.dana_claims_resolved} claims to Dana Reyes "
        "(exact match, exactly one candidate)"
    )

    alex_mention = review_items[0]
    async with factory() as session:
        outcome = await EntityService(session).resolve(
            alex_mention.id, ResolutionCreate(role=ResolutionRole.MENTION)
        )
    result.ambiguous_declined = outcome.status == "unresolved"
    _say(
        f"declined  'Alex Lee': {outcome.reason} "
        f"({len(outcome.candidate_entity_ids)} candidates) - a guess could fuse two people"
    )

    async with factory() as session:
        wrong = await EntityService(session).resolve(
            alex_mention.id,
            ResolutionCreate(role=ResolutionRole.MENTION, entity_id=studio_alex.id),
        )
    assert wrong.resolution is not None  # noqa: S101 - a manual link always resolves
    async with factory() as session:
        await EntityService(session).supersede(
            wrong.resolution.id,
            ResolutionSupersede(entity_id=payments_alex.id, note="wrong Alex Lee"),
        )
        history = await EntityService(session).list_resolutions(alex_mention.id)
    result.correction_history_rows = len(history.items)
    for row in history.items:
        state = "active" if row.is_active else "superseded"
        _say(f"history   {row.method.value:7} -> entity {str(row.entity_id)[:8]}  [{state}]")

    _step("4. Bounded cross-evidence relations, with review and reuse")
    classifier = ScriptedClassifier(_decide)
    async with factory() as session:
        analysis = await IntelligenceService(session, classifier).analyze(dana.id)
    telemetry = analysis.telemetry
    _say(
        f"pairs     {telemetry.theoretical_pairs} theoretical, "
        f"{telemetry.candidate_pairs} cross-evidence candidates, "
        f"{telemetry.external_call_count} model call(s)"
    )
    trivial: uuid.UUID | None = None
    declined = 0
    for relation in analysis.relations:
        result.relation_types.append(relation.relation_type.value)
        if relation.relation_type is RelationType.NO_MEANINGFUL_RELATION:
            declined += 1
            continue
        _say(f"derived   {relation.relation_type.value:11} {relation.explanation}")
        if relation.relation_type is RelationType.REINFORCES:
            trivial = relation.id
    _say(f"declined  {declined} pairs as no_meaningful_relation - stored, so never asked again")
    if trivial is not None:
        async with factory() as session:
            await IntelligenceService(session, classifier).reject_relation(
                trivial, RelationReject(note="technically true and useless")
            )
        _say("rejected  the 'reinforces' relation: kept on record, no longer served")

    calls_before = classifier.calls
    async with factory() as session:
        rerun = await IntelligenceService(session, classifier).analyze(dana.id)
    result.rerun_model_calls = classifier.calls - calls_before
    _say(
        f"re-run    {rerun.telemetry.reused_pairs} settled pairs reused, "
        f"{result.rerun_model_calls} model calls - a rejected answer is never paid for twice"
    )

    _step("5. The evidence packet and the citation gate")
    async with factory() as session:
        packet = await PacketService(session).assemble()
    _say(render_packet(packet))
    handle_of = {item_id: handle for handle, item_id in packet_handles(packet).items()}
    by_quote = {item.grounding_quote: item.id for item in (*workshop_items, *follow_up_items)}
    advice = handle_of[by_quote[ADVICE_QUOTE]]
    reading = handle_of[by_quote[READING_QUOTE]]

    def gate(label: str, answer: str) -> bool:
        try:
            validate_bound_answer(answer, packet, require_any=True)
        except InvalidCitation as error:
            _say(f"REJECTED  {label}\n          {error}")
            return False
        _say(f"accepted  {label}")
        return True

    result.gate_accepted_verbatim = gate(
        "a verbatim, unique fragment",
        f'Dana recommends learning by rebuilding [{advice} "rebuild a device you already own"].',
    )
    result.gate_rejected_paraphrase = not gate(
        "a paraphrased fragment",
        f'Dana recommends learning by rebuilding [{advice} "rebuild something you own"].',
    )
    result.gate_rejected_ambiguous = not gate(
        "a fragment two items share",
        f'The hard part comes early [{reading} "the first month"].',
    )
    result.gate_accepted_unsupported_claim = gate(
        "a verbatim citation that does NOT support its claim",
        f"Dana says soldering is the skill that matters most "
        f'[{reading} "datasheets, not soldering"].',
    )
    _say("          The last answer passed: the fragment is verbatim and unique, so the gate")
    _say("          is satisfied, yet the source says the opposite of the claim. The gate")
    _say("          checks where a quote came from, not whether it supports what is said.")
    return result


async def _run(database_url: str) -> DemoResult:
    engine = create_engine(database_url)
    try:
        return await walkthrough(create_session_factory(engine))
    finally:
        await engine.dispose()


def run_migrations(database_url: str) -> None:
    """`alembic upgrade head`, exactly as an operator would run it."""
    from alembic import command
    from alembic.config import Config

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")


def run_against(database_url: str) -> DemoResult:
    """Run the walkthrough against an already-migrated, empty database."""
    return run_async(_run(database_url))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m demo", description=__doc__.split("\n")[0])
    parser.add_argument(
        "--database-url",
        help="an EMPTY PostgreSQL database (postgresql+psycopg://...). It is migrated to "
        "head and written to. Default: a disposable container, removed afterwards.",
    )
    args = parser.parse_args(argv)
    # The packet renderer prints non-ASCII; a Windows pipe defaults to the ANSI
    # code page, which cannot encode all of it.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.database_url:
        run_migrations(args.database_url)
        run_against(args.database_url)
        return 0

    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine", driver="psycopg") as container:
        url = container.get_connection_url()
        run_migrations(url)
        run_against(url)
    return 0
