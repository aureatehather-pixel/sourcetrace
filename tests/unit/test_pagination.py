"""Cursor encode/decode round trip and rejection of malformed input."""

from __future__ import annotations

import datetime as dt
import uuid

import pytest

from intelligence_engine.core.errors import InvalidCursor
from intelligence_engine.core.pagination import decode_cursor, encode_cursor


def test_round_trip_preserves_the_sort_key() -> None:
    created_at = dt.datetime(2026, 8, 22, 15, 30, tzinfo=dt.UTC)
    item_id = uuid.uuid4()

    decoded = decode_cursor(encode_cursor(created_at, item_id))

    assert decoded.created_at == created_at
    assert decoded.id == item_id


def test_non_utc_offsets_normalise_to_the_same_instant() -> None:
    utc8 = dt.timezone(dt.timedelta(hours=8))
    created_at = dt.datetime(2026, 8, 22, 15, 30, tzinfo=utc8)
    item_id = uuid.uuid4()

    decoded = decode_cursor(encode_cursor(created_at, item_id))

    assert decoded.created_at == created_at  # same instant
    assert decoded.created_at.utcoffset() == dt.timedelta(0)


def test_naive_timestamps_are_refused() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        encode_cursor(dt.datetime(2026, 8, 22, 15, 30), uuid.uuid4())


@pytest.mark.parametrize(
    "cursor",
    [
        "not-base64!!",
        "",
        "YWJj",  # valid base64, no separator
        "MjAyNi0wOC0yMlQxNTozMDowMCswMDowMHxub3QtYS11dWlk",  # bad uuid
        "MjAyNi0wOC0yMlQxNTozMDowMHxkZWFkYmVlZg==",  # naive timestamp, bad uuid
    ],
)
def test_malformed_cursors_raise_invalid_cursor(cursor: str) -> None:
    with pytest.raises(InvalidCursor):
        decode_cursor(cursor)
