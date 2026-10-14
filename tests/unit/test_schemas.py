"""Trust-boundary validation. Prove the gate rejects, not only that it accepts."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from intelligence_engine.evidence.schemas import MAX_CONTENT_LENGTH, EvidenceCreate

MINIMAL = {"content": "An engineer's advice."}


def test_content_alone_is_sufficient() -> None:
    payload = EvidenceCreate.model_validate(MINIMAL)
    assert payload.content == "An engineer's advice."
    assert payload.source_type is None
    assert payload.metadata == {}


def test_content_is_stored_verbatim_not_cleaned() -> None:
    """Invariant 2 - evidence is never semantically rewritten before storage."""
    raw = "  spaced out\n\nadvice\twith tabs  "
    assert EvidenceCreate.model_validate({"content": raw}).content == raw


@pytest.mark.parametrize("content", ["", "   ", "\n\t  \n"])
def test_blank_content_is_rejected(content: str) -> None:
    with pytest.raises(ValidationError):
        EvidenceCreate.model_validate({"content": content})


def test_oversized_content_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EvidenceCreate.model_validate({"content": "x" * (MAX_CONTENT_LENGTH + 1)})


def test_naive_captured_at_is_rejected() -> None:
    """Invariant 7 - ambiguous local timestamps never enter the system."""
    with pytest.raises(ValidationError, match="timezone-aware"):
        EvidenceCreate.model_validate({**MINIMAL, "captured_at": "2026-08-22T15:30:00"})


def test_aware_captured_at_is_accepted_and_offset_preserved() -> None:
    payload = EvidenceCreate.model_validate({**MINIMAL, "captured_at": "2026-08-22T15:30:00+08:00"})
    assert payload.captured_at is not None
    assert payload.captured_at.utcoffset() == dt.timedelta(hours=8)


def test_unknown_fields_are_rejected() -> None:
    """Default to deny: a typo'd field must not silently vanish."""
    with pytest.raises(ValidationError):
        EvidenceCreate.model_validate({**MINIMAL, "sorce_label": "typo"})


@pytest.mark.parametrize("url", ["not-a-url", "javascript:alert(1)", "ftp:/broken"])
def test_malformed_source_url_is_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        EvidenceCreate.model_validate({**MINIMAL, "source_url": url})


def test_valid_source_url_is_accepted() -> None:
    payload = EvidenceCreate.model_validate({**MINIMAL, "source_url": "https://example.org/post/1"})
    assert str(payload.source_url) == "https://example.org/post/1"


def test_metadata_must_be_an_object() -> None:
    with pytest.raises(ValidationError):
        EvidenceCreate.model_validate({**MINIMAL, "metadata": ["not", "an", "object"]})
