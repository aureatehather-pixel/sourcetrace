"""Invariant 9 - fingerprint determinism."""

from __future__ import annotations

import hashlib

import pytest

from intelligence_engine.evidence.fingerprint import content_fingerprint

SAMPLE = "Engineer told me serious projects mattered more than LeetCode."


def test_is_sha256_of_utf8_bytes() -> None:
    assert content_fingerprint(SAMPLE) == hashlib.sha256(SAMPLE.encode("utf-8")).hexdigest()


def test_is_deterministic_across_calls() -> None:
    assert content_fingerprint(SAMPLE) == content_fingerprint(SAMPLE)


def test_output_shape_is_lowercase_hex_64() -> None:
    digest = content_fingerprint(SAMPLE)
    assert len(digest) == 64
    assert digest == digest.lower()
    assert set(digest) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Serious projects matter.", "serious projects matter."),  # case
        ("Serious projects matter.", "Serious projects matter. "),  # trailing space
        ("Serious projects matter.", " Serious projects matter."),  # leading space
        ("Line one\nLine two", "Line one\r\nLine two"),  # line endings
        ("a  b", "a b"),  # internal whitespace
    ],
)
def test_meaningful_differences_are_not_normalised_away(left: str, right: str) -> None:
    """No trimming, folding or collapsing: we fingerprint exactly what is stored."""
    assert content_fingerprint(left) != content_fingerprint(right)


def test_handles_non_ascii_evidence() -> None:
    """Malay, Chinese and emoji all round-trip through UTF-8 encoding."""
    for text in ("Nasihat daripada jurutera", "认真的项目更重要", "advice 🔑"):
        assert len(content_fingerprint(text)) == 64
