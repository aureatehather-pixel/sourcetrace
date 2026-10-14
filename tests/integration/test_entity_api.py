"""Entity connection through the real HTTP boundary.

The app is built by the production factory and runs its real lifespan; only the
extraction provider on `app.state` is swapped for a deterministic fake.

The centrepiece is `test_two_evidence_items_become_one_profile`, which walks the
whole M3 product loop over HTTP: two evidence items in, extraction on each,
resolution to one entity, and a profile that carries grounded material from both
with provenance intact.
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
from intelligence_engine.extraction.provider import ExtractionProvider, ProviderIdentity
from intelligence_engine.main import create_app
from tests.entity_fixtures import (
    A_QUOTE_MENTION,
    ALEX_ONE_QUOTE,
    B_QUOTE_MENTION,
    EVIDENCE_A_CONTENT,
    EVIDENCE_ALEX_ONE_CONTENT,
    EVIDENCE_B_CONTENT,
    EVIDENCE_UNNAMED_CONTENT,
    alex_payload,
    evidence_a_payload,
    evidence_b_payload,
    unnamed_payload,
)

pytestmark = pytest.mark.integration

EVIDENCE_URL = "/api/v1/evidence"
ENTITIES_URL = "/api/v1/entities"


class _RoutingProvider:
    """Returns a different payload per evidence content.

    One app instance, two evidence items, two extraction results - which is what
    the real system does and what the two-evidence profile test needs.
    """

    def __init__(self, by_content: dict[str, dict[str, Any]]) -> None:
        self._by_content = by_content

    @property
    def identity(self) -> ProviderIdentity:
        """The same identity every delegate reports, because it delegates to them."""
        return FakeExtractionProvider().identity

    async def extract(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> Any:
        for content, payload in self._by_content.items():
            if content in user_prompt:
                return await FakeExtractionProvider(payload=payload).extract(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_schema=json_schema,
                )
        raise AssertionError("No fixture payload matches this evidence")


async def _client(settings: Settings, provider: ExtractionProvider) -> AsyncIterator[AsyncClient]:
    app: FastAPI = create_app(settings)
    async with app.router.lifespan_context(app):
        app.state.extraction_provider = provider
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as http:
            yield http


async def _ingest(http: AsyncClient, *, content: str, label: str) -> list[dict[str, Any]]:
    created = await http.post(
        EVIDENCE_URL,
        json={"content": content, "source_type": "conversation_note", "source_label": label},
    )
    assert created.status_code == 201
    evidence_id = created.json()["id"]

    extracted = await http.post(f"{EVIDENCE_URL}/{evidence_id}/extract")
    assert extracted.status_code == 201, extracted.text
    items: list[dict[str, Any]] = extracted.json()["runs"][0]["items"]
    return items


async def _create_entity(
    http: AsyncClient,
    *,
    name: str,
    entity_type: str = "person",
    disambiguator: str | None = None,
) -> str:
    response = await http.post(
        ENTITIES_URL,
        json={
            "entity_type": entity_type,
            "canonical_name": name,
            "disambiguator": disambiguator,
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _resolve(
    http: AsyncClient,
    item_id: str,
    *,
    role: str = "mention",
    entity_id: str | None = None,
) -> Any:
    body: dict[str, Any] = {"role": role}
    if entity_id is not None:
        body["entity_id"] = entity_id
    return await http.post(f"/api/v1/extraction-items/{item_id}/resolution", json=body)


def _of_kind(items: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [item for item in items if item["kind"] == kind]


# --- entities -------------------------------------------------------------


async def test_creating_and_reading_an_entity(settings: Settings) -> None:
    async for http in _client(settings, FakeExtractionProvider(payload=evidence_a_payload())):
        entity_id = await _create_entity(http, name="Zilan", disambiguator="Kestrel Python meetup")
        response = await http.get(f"{ENTITIES_URL}/{entity_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["entity_type"] == "person"
    assert body["canonical_name"] == "Zilan"
    assert body["normalized_name"] == "zilan"
    assert body["disambiguator"] == "Kestrel Python meetup"


async def test_two_same_named_entities_are_both_created_and_both_listed(
    settings: Settings,
) -> None:
    async for http in _client(settings, FakeExtractionProvider(payload=evidence_a_payload())):
        first = await _create_entity(http, name="Alex Lee", disambiguator="design")
        second = await _create_entity(http, name="Alex Lee", disambiguator="payments")
        listing = await http.get(ENTITIES_URL, params={"entity_type": "person"})

    assert first != second
    ids = {item["id"] for item in listing.json()["items"]}
    assert ids == {first, second}


async def test_unknown_entity_returns_404_with_a_stable_code(settings: Settings) -> None:
    async for http in _client(settings, FakeExtractionProvider(payload=evidence_a_payload())):
        response = await http.get(f"{ENTITIES_URL}/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "entity_not_found"


async def test_there_is_no_delete_endpoint_for_entities(settings: Settings) -> None:
    """History is not deleted, so the verb does not exist."""
    async for http in _client(settings, FakeExtractionProvider(payload=evidence_a_payload())):
        entity_id = await _create_entity(http, name="Zilan")
        deleted = await http.delete(f"{ENTITIES_URL}/{entity_id}")
        updated = await http.put(f"{ENTITIES_URL}/{entity_id}", json={})

    assert deleted.status_code == 405
    assert updated.status_code == 405


# --- resolution over HTTP -------------------------------------------------


async def test_resolving_returns_201_and_declining_returns_200(settings: Settings) -> None:
    """The status code and the body both carry the outcome."""
    async for http in _client(settings, FakeExtractionProvider(payload=evidence_a_payload())):
        items = await _ingest(http, content=EVIDENCE_A_CONTENT, label="meetup")
        mention = _of_kind(items, "entity_mention")[0]

        declined = await _resolve(http, mention["id"])
        entity_id = await _create_entity(http, name="Zilan")
        resolved = await _resolve(http, mention["id"])

    assert declined.status_code == 200
    assert declined.json()["status"] == "unresolved"
    assert declined.json()["reason"] == "no_candidate"

    assert resolved.status_code == 201
    body = resolved.json()
    assert body["status"] == "resolved"
    assert body["resolution"]["entity_id"] == entity_id
    assert body["resolution"]["method"] == "exact_match"
    assert body["resolution"]["is_active"] is True


async def test_ambiguity_is_reported_with_both_candidates(settings: Settings) -> None:
    """The API hands the operator what it saw instead of picking for them."""
    async for http in _client(
        settings, FakeExtractionProvider(payload=alex_payload(ALEX_ONE_QUOTE, "Design team"))
    ):
        items = await _ingest(http, content=EVIDENCE_ALEX_ONE_CONTENT, label="design review")
        first = await _create_entity(http, name="Alex Lee", disambiguator="design")
        second = await _create_entity(http, name="Alex Lee", disambiguator="payments")

        response = await _resolve(http, _of_kind(items, "entity_mention")[0]["id"])

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unresolved"
    assert body["reason"] == "ambiguous_candidates"
    assert body["resolution"] is None
    assert set(body["candidate_entity_ids"]) == {first, second}


async def test_an_unnamed_attribution_stays_unresolved_over_http(settings: Settings) -> None:
    async for http in _client(settings, FakeExtractionProvider(payload=unnamed_payload())):
        items = await _ingest(http, content=EVIDENCE_UNNAMED_CONTENT, label="DevGuild")
        await _create_entity(http, name="DevGuild", entity_type="organization")

        response = await _resolve(
            http, _of_kind(items, "reported_claim")[0]["id"], role="claimed_by"
        )

    assert response.status_code == 200
    assert response.json()["status"] == "unresolved"


async def test_a_role_that_does_not_fit_the_item_kind_is_422(settings: Settings) -> None:
    async for http in _client(settings, FakeExtractionProvider(payload=evidence_a_payload())):
        items = await _ingest(http, content=EVIDENCE_A_CONTENT, label="meetup")
        entity_id = await _create_entity(http, name="Zilan")

        response = await _resolve(
            http,
            _of_kind(items, "entity_mention")[0]["id"],
            role="claimed_by",
            entity_id=entity_id,
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "incompatible_resolution_role"


async def test_resolving_twice_is_409_not_a_silent_overwrite(settings: Settings) -> None:
    async for http in _client(settings, FakeExtractionProvider(payload=evidence_a_payload())):
        items = await _ingest(http, content=EVIDENCE_A_CONTENT, label="meetup")
        mention = _of_kind(items, "entity_mention")[0]
        first = await _create_entity(http, name="Zilan")
        second = await _create_entity(http, name="Zilan Orr")

        await _resolve(http, mention["id"], entity_id=first)
        response = await _resolve(http, mention["id"], entity_id=second)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "resolution_already_exists"


async def test_resolving_an_unknown_item_is_404(settings: Settings) -> None:
    async for http in _client(settings, FakeExtractionProvider(payload=evidence_a_payload())):
        response = await _resolve(http, str(uuid.uuid4()))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "extraction_item_not_found"


# --- human correction over HTTP -------------------------------------------


async def test_a_human_can_correct_a_wrong_resolution(settings: Settings) -> None:
    """ "These are not the same person" — expressible entirely through the API."""
    async for http in _client(
        settings, FakeExtractionProvider(payload=alex_payload(ALEX_ONE_QUOTE, "Design team"))
    ):
        items = await _ingest(http, content=EVIDENCE_ALEX_ONE_CONTENT, label="design review")
        mention = _of_kind(items, "entity_mention")[0]
        wrong = await _create_entity(http, name="Alex Lee", disambiguator="payments")
        right = await _create_entity(http, name="Alex Lee", disambiguator="design")

        first = await _resolve(http, mention["id"], entity_id=wrong)
        resolution_id = first.json()["resolution"]["id"]

        corrected = await http.post(
            f"/api/v1/entity-resolutions/{resolution_id}/supersede",
            json={"entity_id": right, "note": "wrong Alex Lee - this is the designer"},
        )
        history = await http.get(f"/api/v1/extraction-items/{mention['id']}/resolutions")
        wrong_profile = await http.get(f"{ENTITIES_URL}/{wrong}/profile")
        right_profile = await http.get(f"{ENTITIES_URL}/{right}/profile")

    assert corrected.status_code == 200
    body = corrected.json()
    assert body["status"] == "resolved"
    assert body["resolution"]["entity_id"] == right
    assert body["resolution"]["method"] == "manual"
    assert body["superseded"]["id"] == resolution_id
    assert body["superseded"]["is_active"] is False
    assert body["superseded"]["entity_id"] == wrong

    # Both decisions remain on the record.
    rows = history.json()["items"]
    assert len(rows) == 2
    assert [row["is_active"] for row in rows] == [True, False]

    # The profile follows the correction.
    assert wrong_profile.json()["mentions"] == []
    assert len(right_profile.json()["mentions"]) == 1


async def test_a_human_can_detach_without_naming_a_replacement(settings: Settings) -> None:
    async for http in _client(settings, FakeExtractionProvider(payload=evidence_a_payload())):
        items = await _ingest(http, content=EVIDENCE_A_CONTENT, label="meetup")
        mention = _of_kind(items, "entity_mention")[0]
        entity_id = await _create_entity(http, name="Zilan")
        first = await _resolve(http, mention["id"], entity_id=entity_id)
        resolution_id = first.json()["resolution"]["id"]

        detached = await http.post(
            f"/api/v1/entity-resolutions/{resolution_id}/supersede",
            json={"note": "a different Zilan"},
        )
        again = await http.post(f"/api/v1/entity-resolutions/{resolution_id}/supersede", json={})

    assert detached.status_code == 200
    assert detached.json()["status"] == "unresolved"
    assert detached.json()["reason"] == "detached_by_correction"

    assert again.status_code == 409
    assert again.json()["error"]["code"] == "resolution_already_superseded"


# --- THE CENTRAL M3 TEST --------------------------------------------------


async def test_two_evidence_items_become_one_profile(settings: Settings) -> None:
    """Evidence A + Evidence B -> one entity -> one grounded profile.

    Two independent notes, weeks apart, extracted separately. After resolution
    the profile must carry material from both, and every line must still name
    the evidence it came from.
    """
    provider = _RoutingProvider(
        {
            EVIDENCE_A_CONTENT: evidence_a_payload(),
            EVIDENCE_B_CONTENT: evidence_b_payload(),
        }
    )

    async for http in _client(settings, provider):
        items_a = await _ingest(http, content=EVIDENCE_A_CONTENT, label="Kestrel Python meetup")
        items_b = await _ingest(http, content=EVIDENCE_B_CONTENT, label="Follow-up coffee")

        zilan = await _create_entity(
            http, name="Zilan", disambiguator="Kestrel Python meetup, query planner rebuild"
        )

        # Evidence A: the mention resolves off the entity just created.
        first_mention = await _resolve(http, _of_kind(items_a, "entity_mention")[0]["id"])
        for claim in _of_kind(items_a, "reported_claim"):
            assert (await _resolve(http, claim["id"], role="claimed_by")).status_code == 201

        # Evidence B: same person, separate evidence, resolved the same way.
        second_mention = await _resolve(
            http,
            next(
                item
                for item in _of_kind(items_b, "entity_mention")
                if item["payload"]["name"] == "Zilan"
            )["id"],
        )
        for claim in _of_kind(items_b, "reported_claim"):
            assert (await _resolve(http, claim["id"], role="claimed_by")).status_code == 201

        profile = (await http.get(f"{ENTITIES_URL}/{zilan}/profile")).json()
        # Follow one profile line back to its evidence, the way a reader would.
        traced = [
            (await http.get(f"{EVIDENCE_URL}/{entry['evidence_id']}")).json()
            for entry in profile["mentions"]
        ]

    assert first_mention.json()["resolution"]["method"] == "exact_match"
    assert second_mention.json()["resolution"]["method"] == "exact_match"

    # Both evidence items feed one identity.
    assert profile["evidence_count"] == 2
    assert {source["source_label"] for source in profile["sources"]} == {
        "Kestrel Python meetup",
        "Follow-up coffee",
    }

    # Two mentions, from two different evidence items.
    quotes = {entry["grounding_quote"] for entry in profile["mentions"]}
    assert quotes == {A_QUOTE_MENTION, B_QUOTE_MENTION}
    assert len({entry["evidence_id"] for entry in profile["mentions"]}) == 2

    # Three attributed claims, including the later one that qualifies the
    # earlier one. Both survive separately, each with its own source.
    assert len(profile["attributed_claims"]) == 3
    assert {entry["stance"] for entry in profile["attributed_claims"]} == {
        "recommended",
        "asserted",
    }

    # Every line names its evidence, run and item, and quotes real source text.
    for entry in [*profile["mentions"], *profile["attributed_claims"]]:
        assert entry["evidence_id"]
        assert entry["extraction_run_id"]
        assert entry["extraction_item_id"]
        assert entry["resolution_id"]
        assert entry["grounding_quote"]

    # And the evidence is retrievable, unchanged, and actually contains the
    # quote the profile attributed to it.
    contents = {item["id"]: item["content"] for item in traced}
    assert set(contents.values()) == {EVIDENCE_A_CONTENT, EVIDENCE_B_CONTENT}
    for entry in profile["mentions"]:
        assert entry["grounding_quote"] in contents[entry["evidence_id"]]


async def test_resolving_a_person_does_not_absorb_a_co_mentioned_organisation(
    settings: Settings,
) -> None:
    """Co-occurrence is not attribution, proven through the API."""
    async for http in _client(settings, FakeExtractionProvider(payload=evidence_b_payload())):
        items = await _ingest(http, content=EVIDENCE_B_CONTENT, label="Follow-up coffee")
        zilan = await _create_entity(http, name="Zilan")
        glyde = await _create_entity(http, name="Glyde", entity_type="organization")

        for mention in _of_kind(items, "entity_mention"):
            assert (await _resolve(http, mention["id"])).status_code == 201

        person_profile = (await http.get(f"{ENTITIES_URL}/{zilan}/profile")).json()
        org_profile = (await http.get(f"{ENTITIES_URL}/{glyde}/profile")).json()

    assert [entry["name_as_mentioned"] for entry in person_profile["mentions"]] == ["Zilan"]
    assert [entry["name_as_mentioned"] for entry in org_profile["mentions"]] == ["Glyde"]
    assert org_profile["attributed_claims"] == []


async def test_the_profile_of_a_brand_new_entity_is_empty_not_invented(
    settings: Settings,
) -> None:
    async for http in _client(settings, FakeExtractionProvider(payload=evidence_a_payload())):
        entity_id = await _create_entity(http, name="Nobody Yet")
        profile = (await http.get(f"{ENTITIES_URL}/{entity_id}/profile")).json()

    assert profile["evidence_count"] == 0
    assert profile["sources"] == []
    assert profile["mentions"] == []
    assert profile["attributed_claims"] == []
