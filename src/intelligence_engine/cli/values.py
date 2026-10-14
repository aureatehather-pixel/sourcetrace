"""Parsing for the few scalar values an operator types by hand.

These live apart from `args.py` because the same rules apply to two inputs that
arrive by different routes: a command-line flag, and a front-matter field in an
evidence file. One definition, so `--captured-at` and a file's `captured_at:`
cannot come to disagree about what a valid timestamp is.

The rules exist to protect provenance. A naive timestamp is refused rather than
assumed to be local time, because "2026-08-22T15:30:00" recorded as UTC when it
meant UTC+8 is a wrong observation time that nothing downstream can detect.
"""

from __future__ import annotations

import datetime as dt

from intelligence_engine.cli.commands import CliUsageError

#: Accepted in place of a timestamp, for evidence being written down as it happens.
NOW = "now"


def parse_timestamp(value: str, *, field: str) -> dt.datetime:
    """An ISO-8601 timestamp that carries a timezone, or the literal `now`.

    `field` names the source in the error message, so an operator is told whether
    the problem is in their flag or in the file they pointed at.
    """
    if value.strip().casefold() == NOW:
        return dt.datetime.now(dt.UTC)

    try:
        parsed = dt.datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise CliUsageError(f"{field}: not a valid timestamp: {value!r}") from exc

    if parsed.tzinfo is None:
        raise CliUsageError(
            f"{field}: must include a timezone offset, e.g. 2026-08-22T15:30:00+08:00 "
            f"(got {value!r}). A local time read as UTC is a wrong observation time, "
            f"so it is refused rather than guessed."
        )
    return parsed


def parse_meta_pair(value: str, *, field: str) -> tuple[str, str]:
    """`KEY=VALUE`, split on the first `=` so values may contain more."""
    key, separator, rest = value.partition("=")
    if not separator or not key.strip():
        raise CliUsageError(f"{field} expects KEY=VALUE, got {value!r}")
    return key.strip(), rest
