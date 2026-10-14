"""Keyset pagination over (created_at DESC, id DESC).

Lives in `core` because more than one aggregate now lists this way: evidence
and entities both order by (created_at DESC, id DESC) and both hand out the
same opaque cursor.

Keyset rather than OFFSET: the ordering key is unique and indexed, so pages stay
stable while new evidence is being ingested, and deep pages cost the same as
shallow ones.

The cursor is opaque to clients but intentionally not secret - it encodes only
the sort key of the last row on the previous page.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import uuid
from typing import NamedTuple

from intelligence_engine.core.errors import InvalidCursor

_SEPARATOR = "|"


class Cursor(NamedTuple):
    created_at: dt.datetime
    id: uuid.UUID


def encode_cursor(created_at: dt.datetime, item_id: uuid.UUID) -> str:
    if created_at.tzinfo is None:
        raise ValueError("cursor timestamps must be timezone-aware")
    raw = f"{created_at.astimezone(dt.UTC).isoformat()}{_SEPARATOR}{item_id}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> Cursor:
    """Decode a cursor, raising :class:`InvalidCursor` for anything malformed."""
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        timestamp_part, id_part = raw.split(_SEPARATOR, maxsplit=1)
        created_at = dt.datetime.fromisoformat(timestamp_part)
        item_id = uuid.UUID(id_part)
    except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
        raise InvalidCursor("Malformed pagination cursor") from exc

    if created_at.tzinfo is None:
        raise InvalidCursor("Malformed pagination cursor")
    return Cursor(created_at=created_at, id=item_id)
