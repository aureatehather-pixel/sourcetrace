"""Cross-evidence intelligence through the real HTTP boundary.

The app is built by the production factory and runs its real lifespan; only the
two providers on `app.state` are swapped for deterministic fakes.

The centrepiece is `test_the_whole_m4_loop_over_http`, which walks the product
loop end to end: two evidence items in, extraction on each, resolution to one
entity, relations derived across the two notes, and an intelligence view that
keeps the derived assertion structurally separate from the grounded claims while
tracing it back to both.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from intelligence_engine.core.config import Settings
from intelligence_engine.extraction.fake_provider import FakeExtractionProvider
from intelligence_engine.extraction.provider import ExtractionProvider
from intelligence_engine.main import create_app
from tests.entity_fixtures import (
    EVIDENCE_A_CONTENT,
    EVIDENCE_B_CONTENT,
    evidence_a_payload,
    evidence_b_payload,
)
from tests.integration.test_entity_api import _RoutingProvider
from tests.intelligence_fixtures import classifier_response, no_relation, qualifies

pytestmark = pytest.mark.integration

EVIDENCE_URL = "/api/v1/evidence"
ENTITIES_URL = "/api/v1/entities"
RELATIONS_URL = "/api/v1/intelligence-relations"


class _CountingProvider(FakeExtractionProvider):
    """A fake that also answers "did this request cost anything?"."""


async def _client(
    settings: Settings, extraction: ExtractionProvider, relations: ExtractionProvider
) -> AsyncIterator[AsyncClient]:
    app: FastAPI = create_app(settings)
    async with app.router.lifespan_context(app):
        app.state.extraction_provider = extraction
        app.state.relation_provider = relations
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as http:
            yield http


async def _ingest(http: AsyncClient, *, content: str, label: str) -> list[dict[str, Any]]:
    created = await http.post(
        EVIDENCE_URL,
        json={"content": content, "source_type": "conversation_note", "source_label": label},
    )
    assert created.status_code == 201
    extracted = await http.post(f"{EVIDENCE_URL}/{created.json()['id']}/extract")
    assert extracted.status_code == 201, extracted.text
    items: list[dict[str, Any]] = extracted.json()["runs"][0]["items"]
    return items


async def _connected(http: AsyncClient) -> str:
    """Two notes, one entity, every attributed claim resolved."""
    items_a = await _ingest(http, content=EVIDENCE_A_CONTENT, label="Kestrel Python meetup")
    items_b = await _ingest(http, content=EVIDENCE_B_CONTENT, label="Coffee, Brookside")

    created = await http.post(
        ENTITIES_URL, json={"entity_type": "person", "canonical_name": "Zilan"}
    )
    assert created.status_code == 201
    entity_id = str(created.json()["id"])

    for item in (*items_a, *items_b):
        if item["kind"] != "reported_claim":
            continue
        resolved = await http.post(
            f"/api/v1/extraction-items/{item['id']}/resolution", json={"role": "claimed_by"}
        )
        assert resolved.status_code == 201, resolved.text
    return entity_id


def _relations(count: int) -> str:
    return classifier_response(*[qualifies(f"pair_{n:02d}") for n in range(1, count + 1)])


def _declines(count: int) -> str:
    return classifier_response(*[no_relation(f"pair_{n:02d}") for n in range(1, count + 1)])


# --- the product loop -----------------------------------------------------


async def test_the_whole_m4_loop_over_http(settings: Settings) -> None:
    extraction = _RoutingProvider(
        {EVIDENCE_A_CONTENT: evidence_a_payload(), EVIDENCE_B_CONTENT: evidence_b_payload()}
    )
    classifier = _CountingProvider(raw_output=_relations(2))

    async for http in _client(settings, extraction, classifier):
        entity_id = await _connected(http)

        analysed = await http.post(f"{ENTITIES_URL}/{entity_id}/analyze-relations")
        assert analysed.status_code == 201, analysed.text
        body = analysed.json()

        assert body["performed_calls"] is True
        assert body["classifier_name"] == "relation_classifier"
        assert body["telemetry"]["theoretical_pairs"] == 3
        assert body["telemetry"]["candidate_pairs"] == 2
        assert body["telemetry"]["classified_pairs"] == 2
        assert body["telemetry"]["external_call_count"] == 1
        assert body["telemetry"]["input_tokens"] == 100

        view = await http.get(f"{ENTITIES_URL}/{entity_id}/intelligence")
        assert view.status_code == 200
        payload = view.json()

        assert len(payload["grounded_claims"]) == 3
        assert len(payload["derived_relations"]) == 2

        for derived in payload["derived_relations"]:
            relation = derived["relation"]
            assert relation["relation_type"] == "qualifies"
            assert relation["direction"] in {"left_to_right", "right_to_left"}
            assert relation["explanation"]
            assert relation["status"] == "current"
            # Full provenance on both sides, and the two sides are different notes.
            for side in (derived["left"], derived["right"]):
                assert uuid.UUID(side["evidence_id"])
                assert uuid.UUID(side["extraction_run_id"])
                assert uuid.UUID(side["extraction_item_id"])
                assert uuid.UUID(side["resolution_id"])
                assert side["grounding_quote"]
            assert derived["left"]["evidence_id"] != derived["right"]["evidence_id"]


async def test_a_second_analysis_returns_200_and_costs_nothing(settings: Settings) -> None:
    """The cost contract at the HTTP boundary: 201 spent money, 200 did not."""
    extraction = _RoutingProvider(
        {EVIDENCE_A_CONTENT: evidence_a_payload(), EVIDENCE_B_CONTENT: evidence_b_payload()}
    )
    classifier = _CountingProvider(raw_output=_relations(2))

    async for http in _client(settings, extraction, classifier):
        entity_id = await _connected(http)

        first = await http.post(f"{ENTITIES_URL}/{entity_id}/analyze-relations")
        second = await http.post(f"{ENTITIES_URL}/{entity_id}/analyze-relations")

        assert first.status_code == 201
        assert second.status_code == 200
        assert second.json()["performed_calls"] is False
        assert second.json()["telemetry"]["reused_pairs"] == 2
        assert second.json()["telemetry"]["external_call_count"] == 0
        assert classifier.call_count == 1


async def test_the_explanation_is_never_presented_as_a_quote(settings: Settings) -> None:
    """Derived reasoning and grounded text are different fields, always."""
    extraction = _RoutingProvider(
        {EVIDENCE_A_CONTENT: evidence_a_payload(), EVIDENCE_B_CONTENT: evidence_b_payload()}
    )
    async for http in _client(settings, extraction, _CountingProvider(raw_output=_relations(2))):
        entity_id = await _connected(http)
        await http.post(f"{ENTITIES_URL}/{entity_id}/analyze-relations")
        payload = (await http.get(f"{ENTITIES_URL}/{entity_id}/intelligence")).json()

    derived = payload["derived_relations"][0]
    assert "grounding_quote" not in derived["relation"]
    assert "explanation" not in derived["left"]
    assert "explanation" not in derived["right"]


async def test_declined_pairs_produce_an_empty_intelligence_view(settings: Settings) -> None:
    extraction = _RoutingProvider(
        {EVIDENCE_A_CONTENT: evidence_a_payload(), EVIDENCE_B_CONTENT: evidence_b_payload()}
    )
    async for http in _client(settings, extraction, _CountingProvider(raw_output=_declines(2))):
        entity_id = await _connected(http)
        analysed = await http.post(f"{ENTITIES_URL}/{entity_id}/analyze-relations")
        view = await http.get(f"{ENTITIES_URL}/{entity_id}/intelligence")

    assert analysed.status_code == 201
    assert analysed.json()["telemetry"]["new_meaningful_relations"] == 0
    assert view.json()["derived_relations"] == []
    # The grounded material is still there; there is simply nothing derived.
    assert len(view.json()["grounded_claims"]) == 3


# --- review ---------------------------------------------------------------


async def test_rejecting_a_relation_withdraws_it_from_the_view(settings: Settings) -> None:
    extraction = _RoutingProvider(
        {EVIDENCE_A_CONTENT: evidence_a_payload(), EVIDENCE_B_CONTENT: evidence_b_payload()}
    )
    async for http in _client(settings, extraction, _CountingProvider(raw_output=_relations(2))):
        entity_id = await _connected(http)
        analysed = await http.post(f"{ENTITIES_URL}/{entity_id}/analyze-relations")
        target = analysed.json()["relations"][0]["id"]

        rejected = await http.post(
            f"{RELATIONS_URL}/{target}/reject",
            json={"note": "Different projects; not a qualification."},
        )
        assert rejected.status_code == 200
        assert rejected.json()["status"] == "rejected"

        view = await http.get(f"{ENTITIES_URL}/{entity_id}/intelligence")
        assert target not in {item["relation"]["id"] for item in view.json()["derived_relations"]}

        again = await http.post(f"{RELATIONS_URL}/{target}/reject", json={"note": "still wrong"})
        assert again.status_code == 409
        assert again.json()["error"]["code"] == "relation_already_rejected"


async def test_rejecting_an_unknown_relation_is_404(settings: Settings) -> None:
    async for http in _client(settings, FakeExtractionProvider(), FakeExtractionProvider()):
        response = await http.post(f"{RELATIONS_URL}/{uuid.uuid4()}/reject", json={})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "relation_not_found"


# --- errors ---------------------------------------------------------------


async def test_analysing_an_unknown_entity_is_404(settings: Settings) -> None:
    async for http in _client(settings, FakeExtractionProvider(), FakeExtractionProvider()):
        response = await http.post(f"{ENTITIES_URL}/{uuid.uuid4()}/analyze-relations")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "entity_not_found"


async def test_the_intelligence_view_of_an_unknown_entity_is_404(settings: Settings) -> None:
    async for http in _client(settings, FakeExtractionProvider(), FakeExtractionProvider()):
        response = await http.get(f"{ENTITIES_URL}/{uuid.uuid4()}/intelligence")

    assert response.status_code == 404


async def test_unusable_classifier_output_is_a_502_with_a_bounded_code(
    settings: Settings,
) -> None:
    """Internals, provider text and claim content never reach the client."""
    extraction = _RoutingProvider(
        {EVIDENCE_A_CONTENT: evidence_a_payload(), EVIDENCE_B_CONTENT: evidence_b_payload()}
    )
    async for http in _client(
        settings, extraction, FakeExtractionProvider(raw_output="here are some links")
    ):
        entity_id = await _connected(http)
        response = await http.post(f"{ENTITIES_URL}/{entity_id}/analyze-relations")

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "invalid_relation_output"
    assert "here are some links" not in response.text
    assert "Traceback" not in response.text


async def test_there_is_no_delete_or_update_on_a_relation(settings: Settings) -> None:
    """Derived assertions are history. History is withdrawn, never edited away.

    Asserted against the published API surface rather than by probing verbs: a
    route that does not exist answers 404, which proves nothing about intent. An
    empty set of mutating verbs across every intelligence path does.
    """
    async for http in _client(settings, FakeExtractionProvider(), FakeExtractionProvider()):
        schema = (await http.get("/openapi.json")).json()

    intelligence_paths = {
        path: set(operations)
        for path, operations in schema["paths"].items()
        if "intelligence" in path or "analyze-relations" in path
    }
    assert intelligence_paths, "the M4 routes should be published"
    for path, verbs in intelligence_paths.items():
        assert verbs & {"delete", "put", "patch"} == set(), path


async def test_a_rejection_note_cannot_be_empty(settings: Settings) -> None:
    extraction = _RoutingProvider(
        {EVIDENCE_A_CONTENT: evidence_a_payload(), EVIDENCE_B_CONTENT: evidence_b_payload()}
    )
    async for http in _client(settings, extraction, _CountingProvider(raw_output=_relations(2))):
        entity_id = await _connected(http)
        analysed = await http.post(f"{ENTITIES_URL}/{entity_id}/analyze-relations")
        target = analysed.json()["relations"][0]["id"]

        response = await http.post(f"{RELATIONS_URL}/{target}/reject", json={"note": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


async def test_unknown_fields_on_a_rejection_are_refused(settings: Settings) -> None:
    """Default to deny at the trust boundary, as M2 and M3 do."""
    async for http in _client(settings, FakeExtractionProvider(), FakeExtractionProvider()):
        response = await http.post(
            f"{RELATIONS_URL}/{uuid.uuid4()}/reject",
            json={"note": "ok", "relation_type": "contradicts"},
        )

    assert response.status_code == 422
