"""Deterministic content fingerprinting.

We hash the exact content that gets stored - no trimming, case folding or
whitespace collapsing. Normalisation would make two genuinely different pieces
of evidence collide, and the stored bytes are the thing we actually care about
being able to recognise again.

The fingerprint is not an identity or a uniqueness constraint. It exists for
future unchanged-content detection, caching, processing avoidance and
diagnostics.
"""

from __future__ import annotations

import hashlib


def content_fingerprint(content: str) -> str:
    """Return the lowercase hex SHA-256 of ``content`` encoded as UTF-8."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
