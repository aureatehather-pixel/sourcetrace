"""Extraction through the real HTTP boundary.

The app is built by the production factory and runs its real lifespan; only the
provider on `app.state` is swapped for a deterministic fake, which is exactly
the seam the provider boundary exists to give us.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from intelligence_engine.core.config import Settings
from intelligence_engine.extraction import versioning as extraction_versioning
from intelligence_engine.extraction.errors import (
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
)
from intelligence_engine.extraction.fake_provider import (
    FakeExtractionProvider,
    ScriptedFailureProvider,
)
from intelligence_engine.extraction.provider import ExtractionProvider
from intelligence_engine.main import create_app
from tests.extraction_fixtures import (
    EVIDENCE_CONTENT,
    QUOTE_CLAIM,
    claim_item,
    payload_with_items,
    valid_payload,
)

pytestmark = pytest.mark.integration

EVIDENCE_URL = "/api/v1/evidence"

EVIDENCE_BODY = {
    "content": EVIDENCE_CONTENT,
    "source_type": "conversation_note",
    "source_label": "Conversation at DevGuild",
    "captured_at": "2026-08-22T15:30:00+08:00",
    "metadata": {"event": "DevGuild"},
}


async def _client(settings: Settings, provider: ExtractionProvider) -> AsyncIterator[AsyncClient]:
    app: FastAPI = create_app(settings)
    async with app.router.lifespan_context(app):
        app.state.extraction_provider = provider
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as http:
            yield http


async def _create_evidence(http: AsyncClient, **overrides: object) -> str:
    response = await http.post(EVIDENCE_URL, json={**EVIDENCE_BODY, **overrides})
    assert response.status_code == 201
    return str(response.json()["id"])


# --- golden path ----------------------------------------------------------


async def test_extract_returns_201_with_validated_items(settings: Settings) -> None:
    provider = FakeExtractionProvider(payload=valid_payload())

    async for http in _client(settings, provider):
        evidence_id = await _create_evidence(http)
        response = await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")

    assert response.status_code == 201
    body = response.json()
    assert body["reused"] is False

    run = body["runs"][0]
    assert run["status"] == "succeeded"
    assert run["evidence_id"] == evidence_id
    assert run["extractor_name"] == "evidence_structurer"
    assert run["extractor_version"] == extraction_versioning.EXTRACTOR_VERSION
    assert run["schema_version"] == 2
    assert run["provider"] == "fake"
    assert run["model"] == "fake-deterministic-1"
    assert run["error_code"] is None
    assert len(run["items"]) == 4


async def test_response_traces_back_to_the_exact_evidence_fingerprint(
    settings: Settings,
) -> None:
    """Every extraction must name the exact bytes it describes."""
    provider = FakeExtractionProvider(payload=valid_payload())

    async for http in _client(settings, provider):
        evidence_id = await _create_evidence(http)
        evidence = (await http.get(f"{EVIDENCE_URL}/{evidence_id}")).json()
        run = (await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")).json()["runs"][0]

    assert run["input_content_hash"] == evidence["content_hash"]


async def test_extraction_reports_token_usage(settings: Settings) -> None:
    provider = FakeExtractionProvider(payload=valid_payload())

    async for http in _client(settings, provider):
        evidence_id = await _create_evidence(http)
        run = (await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")).json()["runs"][0]

    assert run["input_tokens"] == 100
    assert run["output_tokens"] == 50


async def test_items_carry_kind_payload_grounding_and_confidence(
    settings: Settings,
) -> None:
    provider = FakeExtractionProvider(payload=valid_payload())

    async for http in _client(settings, provider):
        evidence_id = await _create_evidence(http)
        run = (await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")).json()["runs"][0]

    first = run["items"][0]
    assert first["kind"] == "reported_claim"
    assert first["grounding_quote"] == QUOTE_CLAIM
    assert first["position"] == 0
    # Epistemics survive the whole round trip: advice stays attributed.
    assert first["payload"]["claimed_by"] == "An engineer at DevGuild"


async def test_evidence_is_unchanged_after_extraction_over_http(
    settings: Settings,
) -> None:
    """The M2 invariant, asserted through the same API a user would use."""
    provider = FakeExtractionProvider(payload=valid_payload())

    async for http in _client(settings, provider):
        evidence_id = await _create_evidence(http)
        before = (await http.get(f"{EVIDENCE_URL}/{evidence_id}")).json()
        await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")
        after = (await http.get(f"{EVIDENCE_URL}/{evidence_id}")).json()

    assert after == before
    assert after["content"] == EVIDENCE_CONTENT


# --- reuse ----------------------------------------------------------------


async def test_second_extract_returns_200_and_reuses(settings: Settings) -> None:
    """200 vs 201 makes a cache hit observable without reading the body."""
    provider = FakeExtractionProvider(payload=valid_payload())

    async for http in _client(settings, provider):
        evidence_id = await _create_evidence(http)
        first = await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")
        second = await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")

    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json()["reused"] is False
    assert second.json()["reused"] is True
    assert second.json()["runs"][0]["id"] == first.json()["runs"][0]["id"]
    assert provider.call_count == 1


async def test_identical_content_from_two_sources_extracts_twice(
    settings: Settings,
) -> None:
    """M1's epistemic rule carried into M2: two observations, two extractions."""
    provider = FakeExtractionProvider(payload=valid_payload())

    async for http in _client(settings, provider):
        first_id = await _create_evidence(http, source_label="Engineer at DevGuild")
        second_id = await _create_evidence(http, source_label="Hiring manager, fintech")
        first = await http.post(f"{EVIDENCE_URL}/{first_id}/extract")
        second = await http.post(f"{EVIDENCE_URL}/{second_id}/extract")

    assert first.status_code == second.status_code == 201
    assert first.json()["runs"][0]["id"] != second.json()["runs"][0]["id"]
    assert provider.call_count == 2


# --- listing --------------------------------------------------------------


async def test_list_extractions_returns_history_newest_first(
    settings: Settings,
) -> None:
    """Failures are part of the history and must be visible."""
    async for http in _client(settings, ScriptedFailureProvider(ProviderTimeout("t"))):
        evidence_id = await _create_evidence(http)
        await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")

    # Same database, a provider that now succeeds.
    async for http in _client(settings, FakeExtractionProvider(payload=valid_payload())):
        await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")
        listed = await http.get(f"{EVIDENCE_URL}/{evidence_id}/extractions")

    assert listed.status_code == 200
    runs = listed.json()["items"]
    assert [run["status"] for run in runs] == ["succeeded", "failed"]
    assert runs[1]["error_code"] == "provider_timeout"


async def test_list_extractions_is_empty_before_any_attempt(settings: Settings) -> None:
    async for http in _client(settings, FakeExtractionProvider(payload=valid_payload())):
        evidence_id = await _create_evidence(http)
        listed = await http.get(f"{EVIDENCE_URL}/{evidence_id}/extractions")

    assert listed.status_code == 200
    assert listed.json()["items"] == []


# --- errors ---------------------------------------------------------------


async def test_extracting_unknown_evidence_returns_404(settings: Settings) -> None:
    provider = FakeExtractionProvider(payload=valid_payload())

    async for http in _client(settings, provider):
        response = await http.post(f"{EVIDENCE_URL}/{uuid.uuid4()}/extract")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "evidence_not_found"
    assert provider.call_count == 0


async def test_listing_extractions_for_unknown_evidence_returns_404(
    settings: Settings,
) -> None:
    async for http in _client(settings, FakeExtractionProvider(payload=valid_payload())):
        response = await http.get(f"{EVIDENCE_URL}/{uuid.uuid4()}/extractions")

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ProviderTimeout("timed out"), "provider_timeout"),
        (ProviderUnavailable("upstream down"), "provider_unavailable"),
    ],
)
async def test_provider_failure_returns_502_with_a_bounded_code(
    settings: Settings, error: ProviderError, expected_code: str
) -> None:
    async for http in _client(settings, ScriptedFailureProvider(error)):
        evidence_id = await _create_evidence(http)
        response = await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == expected_code
    assert "request_id" in response.json()["error"]


async def test_invalid_model_output_returns_502_not_500(settings: Settings) -> None:
    provider = FakeExtractionProvider(raw_output="Sure! Here are the things I found:")

    async for http in _client(settings, provider):
        evidence_id = await _create_evidence(http)
        response = await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_provider_output"


async def test_ungrounded_output_returns_502(settings: Settings) -> None:
    provider = FakeExtractionProvider(
        payload=payload_with_items(claim_item(quote="a quote that was never written"))
    )

    async for http in _client(settings, provider):
        evidence_id = await _create_evidence(http)
        response = await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ungrounded_item"


async def test_a_failed_extraction_leaves_no_items_visible(settings: Settings) -> None:
    provider = FakeExtractionProvider(raw_output='{"items": [{"kind": "nope"}]}')

    async for http in _client(settings, provider):
        evidence_id = await _create_evidence(http)
        await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")
        listed = await http.get(f"{EVIDENCE_URL}/{evidence_id}/extractions")

    runs = listed.json()["items"]
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["items"] == []


async def test_error_responses_never_leak_internals_or_evidence(
    settings: Settings,
) -> None:
    """A validation failure must not echo the payload it rejected."""
    provider = FakeExtractionProvider(
        raw_output='{"items": [{"kind": "reported_claim", "data": '
        '{"claim": "SENSITIVE-EVIDENCE-TEXT"}}]}'
    )

    async for http in _client(settings, provider):
        evidence_id = await _create_evidence(http)
        response = await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")

    body = response.text
    assert "SENSITIVE-EVIDENCE-TEXT" not in body
    for leak in ("Traceback", "sqlalchemy", "anthropic", "api_key", "psycopg"):
        assert leak.lower() not in body.lower()


async def test_no_update_or_delete_verbs_on_extractions(settings: Settings) -> None:
    """Extraction runs are history, and history is not edited."""
    provider = FakeExtractionProvider(payload=valid_payload())

    async for http in _client(settings, provider):
        evidence_id = await _create_evidence(http)
        await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")
        codes = {
            method: (
                await http.request(method, f"{EVIDENCE_URL}/{evidence_id}/extractions", json={})
            ).status_code
            for method in ("PUT", "PATCH", "DELETE")
        }

    assert set(codes.values()) == {405}
