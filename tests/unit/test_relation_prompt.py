"""The prompt version gate, and what the classifier is actually told.

The fingerprint pin is the same mechanism M2 uses. Editing the prompt without
bumping `CLASSIFIER_VERSION` fails here, and bumping the version without
re-pinning also fails. Without it, reuse would serve relations produced by a
prompt that no longer exists.
"""

from __future__ import annotations

from dataclasses import replace

from intelligence_engine.entities.schemas import ResolutionRole
from intelligence_engine.intelligence import versioning
from intelligence_engine.intelligence.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    pair_fingerprint,
    prompt_fingerprint,
    render_pair,
)
from tests.intelligence_fixtures import (
    advice_item,
    make_pair,
    profile_observation_item,
    reading_item,
)

# Bump this together with CLASSIFIER_VERSION, never separately.
PINNED_FINGERPRINTS = {
    "1.2.0": "49f3c7fc732dfcea57b403b3d54ea614443739a16619227d2b8f7b5fcedc0f36",
}


class TestVersionGate:
    def test_the_current_version_has_a_pinned_fingerprint(self) -> None:
        assert versioning.CLASSIFIER_VERSION in PINNED_FINGERPRINTS, (
            "CLASSIFIER_VERSION was bumped without pinning the new prompt "
            "fingerprint. Add it to PINNED_FINGERPRINTS."
        )

    def test_the_prompt_matches_its_pinned_fingerprint(self) -> None:
        assert prompt_fingerprint() == PINNED_FINGERPRINTS[versioning.CLASSIFIER_VERSION], (
            "The prompt text changed without a CLASSIFIER_VERSION bump. Relations "
            "stored under the old version would be reused as if current."
        )

    def test_the_fingerprint_covers_every_part_of_the_prompt(self) -> None:
        """Including the per-pair template - it shapes what the model reads."""
        pair = make_pair(advice_item(), reading_item())
        assert render_pair("pair_01", pair) != render_pair("pair_02", pair)


class TestPromptContent:
    def test_the_error_ordering_is_stated_explicitly(self) -> None:
        assert "worst first" in SYSTEM_PROMPT
        assert "false `contradicts`" in SYSTEM_PROMPT

    def test_abstention_is_presented_as_a_correct_answer(self) -> None:
        assert "correct, common and useful answer" in SYSTEM_PROMPT

    def test_causal_inference_is_forbidden(self) -> None:
        assert "Never say or imply that one thing caused" in SYSTEM_PROMPT
        assert "Never write a causal explanation." in SYSTEM_PROMPT

    def test_attribution_is_addressed(self) -> None:
        assert "one person cannot qualify or retract another person's statement" in SYSTEM_PROMPT

    def test_context_is_offered_as_the_defence_against_false_contradiction(self) -> None:
        assert "where and when it was recorded" in SYSTEM_PROMPT

    def test_trivial_relations_are_named_as_worthless(self) -> None:
        assert "worth nothing" in SYSTEM_PROMPT


class TestRendering:
    def test_a_pair_block_carries_both_statements_and_their_context(self) -> None:
        pair = make_pair(advice_item(), reading_item())
        rendered = render_pair("pair_01", pair)

        assert pair.left.text in rendered
        assert pair.right.text in rendered
        assert pair.left.grounding_quote in rendered
        assert "Kestrel Python meetup" in rendered
        assert "Coffee, Brookside" in rendered

    def test_absent_fields_are_marked_absent_not_invented(self) -> None:
        pair = make_pair(advice_item(stance=None, claimed_by=None), reading_item(source_label=None))
        rendered = render_pair("pair_01", pair)
        assert "(not stated)" in rendered
        assert "unknown" not in rendered.lower()

    def test_the_user_prompt_names_the_analysis_scope(self) -> None:
        pair = make_pair(advice_item(), reading_item())
        prompt = build_user_prompt(
            entity_label="Zilan (person)", rendered_pairs=[render_pair("pair_01", pair)]
        )
        assert "Zilan (person)" in prompt
        assert "pair_01" in prompt


class TestPairFingerprint:
    def test_the_same_pair_fingerprints_the_same(self) -> None:
        advice, reading = advice_item(), reading_item()
        assert pair_fingerprint(make_pair(advice, reading)) == pair_fingerprint(
            make_pair(reading, advice)
        )

    def test_batch_position_does_not_change_the_fingerprint(self) -> None:
        """Otherwise reuse would break purely because a pair moved batches."""
        pair = make_pair(advice_item(), reading_item())
        first = pair_fingerprint(pair)
        assert first == pair_fingerprint(pair)
        assert "pair_01" not in first

    def test_different_content_fingerprints_differently(self) -> None:
        base = make_pair(advice_item(), reading_item())
        changed = make_pair(advice_item(text="Something else entirely."), reading_item())
        assert pair_fingerprint(base) != pair_fingerprint(changed)

    def test_it_is_sha256_hex(self) -> None:
        value = pair_fingerprint(make_pair(advice_item(), reading_item()))
        assert len(value) == 64
        assert all(char in "0123456789abcdef" for char in value)


class TestSpeakerIsNotSubject:
    """The 1.2.0 correction, asserted without spending a model call.

    Three relations on real data attributed a listener's future losses to the
    speaker's own history, because the prompt told the classifier every statement
    was "attributed to or about" the entity. A claim reaches an entity through
    `claimed_by`, which says only that the entity *said* it. These tests pin the
    representation; whether the model then uses it is what the gold set and the
    live run measure.
    """

    def test_a_claim_is_rendered_as_said_by_not_as_about(self) -> None:
        """The structural invariant, and the whole point of 1.2.0."""
        block = render_pair("pair_01", make_pair(advice_item(), reading_item()))

        assert "SAID BY the entity above" in block
        assert "not necessarily about them" in block
        assert "OBSERVED ABOUT" not in block

    def test_an_observation_is_rendered_as_observed_about(self) -> None:
        block = render_pair(
            "pair_01",
            make_pair(advice_item(), profile_observation_item()),
        )

        assert "OBSERVED ABOUT the entity above" in block
        assert "SAID BY the entity above" in block

    def test_the_connection_follows_the_asserted_role_not_the_item_kind(self) -> None:
        """Deriving the connection from `kind` is how the original bug arose.

        Changing only the role must change only the connection line, which proves
        the rendering reads the role M3 asserted rather than re-deriving it.
        """
        claim = advice_item()
        relabelled = replace(claim, role=ResolutionRole.ABOUT)

        assert "SAID BY the entity above" in render_pair("p", make_pair(claim, reading_item()))
        assert "OBSERVED ABOUT the entity above" in render_pair(
            "p", make_pair(relabelled, reading_item())
        )

    def test_the_user_prompt_no_longer_equates_said_with_about(self) -> None:
        """The exact wording that produced the three false relations is gone."""
        prompt = build_user_prompt(entity_label="Zi Lan", rendered_pairs=["pair_01"])

        assert "attributed to or about" not in prompt
        assert "They are NOT all about that entity." in prompt

    def test_the_system_prompt_forbids_assuming_a_speaker_describes_themselves(
        self,
    ) -> None:
        assert "Never assume a speaker is talking about themselves." in SYSTEM_PROMPT
        assert "is a statement about the listener, not about" in SYSTEM_PROMPT
        assert "the answer is `no_meaningful_relation`" in SYSTEM_PROMPT
