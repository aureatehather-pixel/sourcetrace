"""Process-wide resources for one CLI invocation.

The CLI calls the service layer directly rather than going over HTTP. That is a
real choice with a real cost, recorded here rather than rediscovered later.

*Why direct:* the services already own their transaction boundaries and the
routes are thin translations of them, so HTTP would add a second process the
operator has to remember to start, a port to get wrong, and a serialisation
round-trip — all to reach code running on the same machine against the same
database. `pie process note.md` should work with PostgreSQL up and nothing else.

*What it costs:* two entry points now reach the same services, so orchestration
could drift between them. That is bounded by keeping all CLI orchestration in
`pipeline.py`, having it call the same service methods the routes call, and
giving it no persistence of its own.

The Windows event-loop constraint applies here exactly as it does to the server:
`run` goes through `core.platform`, never `asyncio.run`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from intelligence_engine.core.config import Settings, get_settings
from intelligence_engine.core.platform import run_async
from intelligence_engine.db.session import create_engine, create_session_factory
from intelligence_engine.entities.service import EntityService
from intelligence_engine.evidence.service import EvidenceService
from intelligence_engine.extraction.factory import build_provider
from intelligence_engine.extraction.service import ExtractionService
from intelligence_engine.intelligence.factory import build_relation_provider
from intelligence_engine.intelligence.service import IntelligenceService
from intelligence_engine.packet.service import PacketService


@dataclass(frozen=True)
class Services:
    """The services, sharing one session so one command is one story.

    `packet` is read-only and takes no provider: it is the one service here that
    works with no credentials configured and no network, which is exactly what
    `pie packet` is for.
    """

    evidence: EvidenceService
    extraction: ExtractionService
    entities: EntityService
    intelligence: IntelligenceService
    packet: PacketService


@asynccontextmanager
async def open_services(settings: Settings) -> AsyncIterator[Services]:
    """Build one engine, one session and the services, then dispose them.

    Providers are built once per invocation, as they are once per server process:
    the Anthropic client owns a connection pool that should not be rebuilt per
    call.

    The session is deliberately *not* wrapped in a rollback-on-exit scope. Each
    service commits its own work, and that is what a pipeline needs: if
    extraction fails, the evidence already committed must stay, because raw
    evidence is the one thing that cannot be regenerated.
    """
    engine = create_engine(settings.database_url_str)
    try:
        session = create_session_factory(engine)()
        try:
            yield Services(
                evidence=EvidenceService(session),
                extraction=ExtractionService(
                    session,
                    build_provider(settings),
                    window_max_chars=settings.extraction_window_max_chars,
                ),
                entities=EntityService(session),
                intelligence=IntelligenceService(
                    session,
                    build_relation_provider(settings),
                    max_candidate_pairs=settings.relation_max_candidate_pairs,
                    batch_size=settings.relation_batch_size,
                ),
                packet=PacketService(session),
            )
        finally:
            await session.close()
    finally:
        await engine.dispose()


def load_settings() -> Settings:
    return get_settings()


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run one CLI invocation on a loop psycopg can actually use."""
    return run_async(coro)
