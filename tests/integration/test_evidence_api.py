"""Exercise the application through its real HTTP boundary.

The app is built by the same factory production uses, runs its real lifespan,
and talks to real PostgreSQL. Only the network socket is elided.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from intelligence_engine.core.config import Settings
from intelligence_engine.evidence.fingerprint import content_fingerprint
from intelligence_engine.main import create_app

pytestmark = pytest.mark.integration

EVIDENCE_URL = "/api/v1/evidence"

ADVICE = "Engineer told me serious projects mattered more than LeetCode."


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    """A client over a fully started application, lifespan included."""
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as http:
            yield http


async def restart(settings: Settings) -> AsyncIterator[AsyncClient]:
    """Yield a client backed by a brand-new app, engine and connection pool."""
    app: FastAPI = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as http:
            yield http


async def test_health_reports_process_liveness(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_reports_database_reachability(client: AsyncClient) -> None:
    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


async def test_golden_path_submit_then_retrieve_after_restart(settings: Settings) -> None:
    """The M1 completion gate, end to end.

    POST -> 201 -> id, then a completely new application instance retrieves the
    original evidence unchanged.
    """
    body = {
        "content": ADVICE,
        "source_type": "conversation_note",
        "source_label": "Conversation at DevGuild",
        "source_url": None,
        "captured_at": "2026-08-22T15:30:00+08:00",
        "metadata": {"event": "DevGuild"},
    }

    async for http in restart(settings):
        created = await http.post(EVIDENCE_URL, json=body)
        assert created.status_code == 201
        evidence_id = created.json()["id"]

    # New app object, new engine, new pool - as if the process had been restarted.
    async for http in restart(settings):
        fetched = await http.get(f"{EVIDENCE_URL}/{evidence_id}")

    assert fetched.status_code == 200
    payload = fetched.json()
    assert payload["id"] == evidence_id
    assert payload["content"] == ADVICE
    assert payload["content_hash"] == content_fingerprint(ADVICE)
    assert payload["source_type"] == "conversation_note"
    assert payload["source_label"] == "Conversation at DevGuild"
    assert payload["metadata"] == {"event": "DevGuild"}
    assert dt.datetime.fromisoformat(payload["captured_at"]) == dt.datetime(
        2026, 8, 22, 15, 30, tzinfo=dt.timezone(dt.timedelta(hours=8))
    )


async def test_created_response_carries_the_full_persisted_representation(
    client: AsyncClient,
) -> None:
    response = await client.post(EVIDENCE_URL, json={"content": ADVICE})

    assert response.status_code == 201
    payload = response.json()
    assert set(payload) == {
        "id",
        "content",
        "content_hash",
        "source_type",
        "source_label",
        "source_url",
        "captured_at",
        "created_at",
        "metadata",
    }
    uuid.UUID(payload["id"])  # parses, so it is a real UUID
    assert payload["content_hash"] == content_fingerprint(ADVICE)


async def test_same_content_different_provenance_yields_two_retrievable_records(
    client: AsyncClient,
) -> None:
    """The critical epistemic regression test, at the HTTP boundary."""
    first = await client.post(
        EVIDENCE_URL,
        json={
            "content": ADVICE,
            "source_type": "conversation_note",
            "source_label": "Senior engineer at DevGuild",
        },
    )
    second = await client.post(
        EVIDENCE_URL,
        json={
            "content": ADVICE,
            "source_type": "conversation_note",
            "source_label": "Hiring manager at a fintech",
        },
    )

    assert first.status_code == second.status_code == 201
    first_id, second_id = first.json()["id"], second.json()["id"]
    assert first_id != second_id
    assert first.json()["content_hash"] == second.json()["content_hash"]

    a = await client.get(f"{EVIDENCE_URL}/{first_id}")
    b = await client.get(f"{EVIDENCE_URL}/{second_id}")
    assert a.json()["source_label"] == "Senior engineer at DevGuild"
    assert b.json()["source_label"] == "Hiring manager at a fintech"

    listed = await client.get(EVIDENCE_URL)
    assert {item["id"] for item in listed.json()["items"]} == {first_id, second_id}


async def test_unknown_id_returns_404_without_internals(client: AsyncClient) -> None:
    response = await client.get(f"{EVIDENCE_URL}/{uuid.uuid4()}")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "evidence_not_found"
    assert "request_id" in error


async def test_malformed_id_returns_422_not_500(client: AsyncClient) -> None:
    response = await client.get(f"{EVIDENCE_URL}/not-a-uuid")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"content": ""},
        {"content": "   "},
        {"content": ADVICE, "captured_at": "2026-08-22T15:30:00"},  # naive
        {"content": ADVICE, "source_url": "not-a-url"},
        {"content": ADVICE, "metadata": "not-an-object"},
        {"content": ADVICE, "unexpected_field": 1},
    ],
)
async def test_invalid_submissions_are_rejected_with_422(
    client: AsyncClient, body: dict[str, object]
) -> None:
    response = await client.post(EVIDENCE_URL, json=body)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["details"]


async def test_no_update_verbs_are_exposed(client: AsyncClient) -> None:
    """Invariant 4 - raw evidence has no HTTP mutation path."""
    created = await client.post(EVIDENCE_URL, json={"content": ADVICE})
    evidence_id = created.json()["id"]

    for method in ("PUT", "PATCH", "DELETE"):
        response = await client.request(
            method, f"{EVIDENCE_URL}/{evidence_id}", json={"content": "rewritten"}
        )
        assert response.status_code == 405, method

    unchanged = await client.get(f"{EVIDENCE_URL}/{evidence_id}")
    assert unchanged.json()["content"] == ADVICE


async def test_list_is_ordered_newest_first(client: AsyncClient) -> None:
    ids = []
    for index in range(5):
        response = await client.post(
            EVIDENCE_URL, json={"content": f"Observation {index}", "source_label": f"src-{index}"}
        )
        ids.append(response.json()["id"])

    listed = await client.get(EVIDENCE_URL)

    assert listed.status_code == 200
    returned = [item["id"] for item in listed.json()["items"]]
    assert returned == list(reversed(ids))
    assert listed.json()["next_cursor"] is None


async def test_cursor_pagination_walks_every_record_exactly_once(
    client: AsyncClient,
) -> None:
    for index in range(7):
        await client.post(EVIDENCE_URL, json={"content": f"Observation {index}"})

    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        params: dict[str, str | int] = {"limit": 3}
        if cursor:
            params["cursor"] = cursor
        page = (await client.get(EVIDENCE_URL, params=params)).json()
        seen.extend(item["id"] for item in page["items"])
        pages += 1
        cursor = page["next_cursor"]
        if cursor is None:
            break
        assert pages < 10, "pagination did not terminate"

    assert len(seen) == 7
    assert len(set(seen)) == 7  # no duplicates, no gaps
    assert pages == 3


async def test_list_limit_is_bounded(client: AsyncClient) -> None:
    over_max = await client.get(EVIDENCE_URL, params={"limit": 10_000})
    assert over_max.status_code == 422

    zero = await client.get(EVIDENCE_URL, params={"limit": 0})
    assert zero.status_code == 422


async def test_malformed_cursor_returns_400(client: AsyncClient) -> None:
    response = await client.get(EVIDENCE_URL, params={"cursor": "obviously-not-a-cursor"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_cursor"


async def test_request_id_is_echoed_and_honoured(client: AsyncClient) -> None:
    supplied = "req-test-12345"
    response = await client.get("/health", headers={"X-Request-ID": supplied})

    assert response.headers["X-Request-ID"] == supplied


async def test_generated_request_id_appears_when_none_supplied(client: AsyncClient) -> None:
    response = await client.get(f"{EVIDENCE_URL}/{uuid.uuid4()}")

    header_id = response.headers["X-Request-ID"]
    uuid.UUID(header_id)
    assert response.json()["error"]["request_id"] == header_id


async def test_error_responses_never_leak_internals(client: AsyncClient) -> None:
    response = await client.get(f"{EVIDENCE_URL}/{uuid.uuid4()}")

    body = response.text.lower()
    for leak in ("traceback", "sqlalchemy", "select ", "psycopg", "password", "postgresql://"):
        assert leak not in body
