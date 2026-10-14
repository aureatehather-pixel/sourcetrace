"""The prompt is part of the extractor's behaviour, so it is version-locked.

Reuse serves stored output whenever the version key matches. If the prompt
could change without the version changing, the system would confidently return
results produced by instructions that no longer exist. This test is the gate
that makes that impossible to do by accident.
"""

from __future__ import annotations

from intelligence_engine.extraction import versioning
from intelligence_engine.extraction.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    prompt_fingerprint,
)

# Bump EXTRACTOR_VERSION and update this fingerprint together, never separately.
# Only the current version is pinned - a fingerprint for a prompt no longer in
# the tree cannot be verified, and git history already records it.
PINNED_PROMPTS: dict[str, str] = {
    "3.0.1": "8a5fabec2d5fae67418232ef84ee4e3d53ab1934cff72af8dbe7ae75dda8aced",
}


def test_prompt_change_requires_an_extractor_version_bump() -> None:
    version = versioning.EXTRACTOR_VERSION
    assert version in PINNED_PROMPTS, (
        f"EXTRACTOR_VERSION {version} has no pinned prompt fingerprint. "
        "Add one to PINNED_PROMPTS when introducing a version."
    )
    assert prompt_fingerprint() == PINNED_PROMPTS[version], (
        "The extraction prompt changed but EXTRACTOR_VERSION did not. "
        "Bump the version and pin the new fingerprint, or revert the prompt — "
        "otherwise cached extractions will be served for a prompt that no "
        "longer exists."
    )


def test_prompt_forbids_upgrading_reported_claims_into_facts() -> None:
    """The single failure mode this prompt exists to prevent."""
    assert "Never upgrade a reported claim into a fact" in SYSTEM_PROMPT
    assert "`reported_claim`" in SYSTEM_PROMPT


def test_prompt_forbids_inferring_causation_and_biography() -> None:
    assert "Never infer causation" in SYSTEM_PROMPT
    assert "Never invent biography" in SYSTEM_PROMPT


def test_prompt_prefers_omission_over_invention() -> None:
    assert "Prefer omission over invention" in SYSTEM_PROMPT
    assert 'means "the evidence did not say"' in SYSTEM_PROMPT


def test_prompt_forbids_interpretation_and_praise() -> None:
    """No hero worship: the model transcribes, it does not admire."""
    assert "Do not editorialize, praise, or interpret" in SYSTEM_PROMPT


def test_prompt_demands_verbatim_grounding() -> None:
    assert "verbatim" in SYSTEM_PROMPT
    assert "exact substring" in SYSTEM_PROMPT


def test_prompt_forbids_scoring_confidence() -> None:
    """A model's self-rating is not calibrated, so it must not be produced."""
    assert "Do not score your confidence" in SYSTEM_PROMPT
    assert "There is no confidence field" in SYSTEM_PROMPT


def test_prompt_keeps_uncertainty_in_the_evidence_wording() -> None:
    """Uncertainty is preserved where it exists, not invented as a number."""
    assert "carry that hedge into the extracted text" in SYSTEM_PROMPT
    assert "`recommended`" in SYSTEM_PROMPT


def test_prompt_allows_an_empty_result() -> None:
    """Without this, a model under pressure invents items to look useful."""
    assert "An empty result is a correct result." in SYSTEM_PROMPT


def test_user_prompt_embeds_content_and_provenance_in_separate_blocks() -> None:
    prompt = build_user_prompt(content="RAW EVIDENCE TEXT", provenance="source_type: note")

    assert "<evidence>\nRAW EVIDENCE TEXT\n</evidence>" in prompt
    assert "source_type: note" in prompt
    # Provenance must not be extractable - it is context, not evidence.
    assert "never extract from this block" in prompt
