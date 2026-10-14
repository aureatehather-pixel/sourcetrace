"""Turning a file on disk into an evidence submission.

Evidence is normally written as a note, and a note's provenance is most reliably
recorded next to it rather than retyped as flags on every run. So a file may
carry optional front matter:

    ---
    source_type: conversation_note
    source_label: Coffee, Brookside
    captured_at: 2026-08-19T10:30:00+08:00
    event: DevGuild
    ---
    Zilan said the first six weeks were mostly reading...

Three deliberate choices:

- **Front matter is provenance, never evidence.** Only the body is stored as
  `content`, so the fingerprint covers what was actually observed. Re-labelling a
  note does not change its content hash, and re-running extraction on it is still
  free.
- **Recognised keys are provenance columns; everything else becomes metadata.**
  A bounded set of column names, and an open dictionary for the rest, matches
  what the evidence schema already offers.
- **Nothing is inferred.** A file with no front matter and no flags produces an
  evidence item with null provenance, which honestly says "nobody recorded
  where this came from". The file's mtime is *not* used as `captured_at`: a
  filesystem timestamp records when bytes were last written, not when the thing
  described was observed, and storing one as the other invents an observation.

The parser is intentionally not YAML. It accepts `key: value` lines and nothing
else — no nesting, no lists, no anchors — because a note's header needs none of
that, and a real YAML dependency would let a file express structures the
evidence schema cannot hold.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from intelligence_engine.cli.commands import CliUsageError, EvidenceMetadata
from intelligence_engine.cli.values import parse_meta_pair, parse_timestamp

FENCE = "---"

#: Front-matter keys that map onto evidence provenance columns. Anything else
#: is metadata, which is exactly what an open JSONB column is for.
PROVENANCE_KEYS = frozenset({"source_type", "source_label", "source_url", "captured_at"})


@dataclass(frozen=True)
class SourceDocument:
    """A file resolved into what will be submitted, and where each part came from."""

    content: str
    meta: EvidenceMetadata
    had_front_matter: bool


def read_source(path: Path, overrides: EvidenceMetadata) -> SourceDocument:
    """Read `path`, split front matter from body, and apply command-line overrides.

    A flag always beats the file: the file records what was true when the note
    was written, and the flag is the operator correcting it now.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CliUsageError(f"no such file: {path}") from exc
    except OSError as exc:
        raise CliUsageError(f"could not read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise CliUsageError(
            f"{path} is not UTF-8 text. Evidence is stored verbatim as text; "
            f"binary material needs a different ingestion path than this one."
        ) from exc

    body, file_meta = _split_front_matter(raw, path)

    if not body.strip():
        raise CliUsageError(
            f"{path} has no evidence content"
            + (" below its front matter" if file_meta is not None else "")
        )

    return SourceDocument(
        content=body,
        meta=_merge(file_meta or EvidenceMetadata(), overrides),
        had_front_matter=file_meta is not None,
    )


def _split_front_matter(raw: str, path: Path) -> tuple[str, EvidenceMetadata | None]:
    """Return `(body, metadata)`; metadata is None when there is no front matter.

    The body is **exactly** the bytes after the closing fence line — line endings
    kept, nothing trimmed from either end. That exactness is what makes the
    documented promise true: the same note with and without a front-matter header
    yields the same content, therefore the same fingerprint, so adding or
    correcting a label never orphans an existing extraction. Rejoining split lines
    silently drops the file's final newline and breaks exactly that.
    """
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].strip() != FENCE:
        return raw, None

    try:
        closing = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == FENCE)
    except StopIteration:
        raise CliUsageError(
            f"{path}: front matter opens with '---' but is never closed by a '---' line"
        ) from None

    return "".join(lines[closing + 1 :]), _parse_header(lines[1:closing], path)


def _parse_header(lines: list[str], path: Path) -> EvidenceMetadata:
    source_type: str | None = None
    source_label: str | None = None
    source_url: str | None = None
    captured_at = None
    metadata: dict[str, str] = {}

    for number, line in enumerate(lines, start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise CliUsageError(
                f"{path} line {number}: front matter expects 'key: value', got {line.strip()!r}"
            )
        key = key.strip()
        value = _unquote(value.strip())
        if not key:
            raise CliUsageError(f"{path} line {number}: front-matter key is empty")

        if key == "source_type":
            source_type = value or None
        elif key == "source_label":
            source_label = value or None
        elif key == "source_url":
            source_url = value or None
        elif key == "captured_at":
            captured_at = parse_timestamp(value, field=f"{path} line {number}: captured_at")
        else:
            metadata[key] = value

    return EvidenceMetadata(
        source_type=source_type,
        source_label=source_label,
        source_url=source_url,
        captured_at=captured_at,
        metadata=metadata,
    )


def _unquote(value: str) -> str:
    """Strip one layer of matching quotes, so a label may contain a colon."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _merge(from_file: EvidenceMetadata, overrides: EvidenceMetadata) -> EvidenceMetadata:
    """Command line wins field by field; metadata dictionaries merge key by key."""
    merged_metadata = {**from_file.metadata, **overrides.metadata}
    return replace(
        from_file,
        source_type=overrides.source_type or from_file.source_type,
        source_label=overrides.source_label or from_file.source_label,
        source_url=overrides.source_url or from_file.source_url,
        captured_at=overrides.captured_at or from_file.captured_at,
        metadata=merged_metadata,
    )


__all__ = ["FENCE", "PROVENANCE_KEYS", "SourceDocument", "parse_meta_pair", "read_source"]
