"""Front matter is provenance; the body is evidence. Both halves are tested.

The rejections matter more than the acceptances here. A front-matter parser that
quietly tolerates a malformed header stores wrong provenance, and wrong
provenance is undetectable later — the content still hashes correctly, so nothing
downstream can tell that `captured_at` was a guess.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from intelligence_engine.cli.commands import CliUsageError, EvidenceMetadata
from intelligence_engine.cli.source import read_source

NO_OVERRIDES = EvidenceMetadata()


def _write(tmp_path: Path, text: str, *, name: str = "note.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_plain_file_is_all_content_and_no_provenance(tmp_path: Path) -> None:
    """Nothing is inferred. Null provenance honestly says nobody recorded it."""
    path = _write(tmp_path, "Zilan said the first six weeks were mostly reading.\n")

    document = read_source(path, NO_OVERRIDES)

    assert document.content == "Zilan said the first six weeks were mostly reading.\n"
    assert document.had_front_matter is False
    assert document.meta == EvidenceMetadata()


def test_front_matter_is_stripped_from_the_stored_content(tmp_path: Path) -> None:
    """Re-labelling a note must not change its fingerprint."""
    path = _write(
        tmp_path,
        "---\n"
        "source_type: conversation_note\n"
        "source_label: Coffee, Brookside\n"
        "captured_at: 2026-08-19T10:30:00+08:00\n"
        "event: DevGuild\n"
        "---\n"
        "He walked back the query planner story slightly.\n",
    )

    document = read_source(path, NO_OVERRIDES)

    assert document.content == "He walked back the query planner story slightly.\n"
    assert "source_type" not in document.content
    assert document.had_front_matter is True
    assert document.meta.source_type == "conversation_note"
    assert document.meta.source_label == "Coffee, Brookside"
    assert document.meta.captured_at == dt.datetime(
        2026, 8, 19, 10, 30, tzinfo=dt.timezone(dt.timedelta(hours=8))
    )
    # Unrecognised keys are metadata, not silently dropped.
    assert document.meta.metadata == {"event": "DevGuild"}


def test_a_label_may_contain_a_colon(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        '---\nsource_label: "Talk: scaling Postgres"\n---\nNotes from the talk.\n',
    )

    assert read_source(path, NO_OVERRIDES).meta.source_label == "Talk: scaling Postgres"


def test_the_command_line_overrides_the_file_field_by_field(tmp_path: Path) -> None:
    """The file records what was true when written; the flag is the correction now."""
    path = _write(
        tmp_path,
        "---\nsource_type: conversation_note\nsource_label: Wrong\nevent: DevGuild\n---\nBody.\n",
    )

    document = read_source(
        path,
        EvidenceMetadata(source_label="Right", metadata={"event": "Kestrel Python meetup"}),
    )

    assert document.meta.source_label == "Right"
    assert document.meta.source_type == "conversation_note"  # untouched by the override
    assert document.meta.metadata == {"event": "Kestrel Python meetup"}


def test_unclosed_front_matter_is_refused(tmp_path: Path) -> None:
    """Otherwise the header silently becomes evidence."""
    path = _write(tmp_path, "---\nsource_type: note\nBody that never closed the fence.\n")

    with pytest.raises(CliUsageError, match="never closed"):
        read_source(path, NO_OVERRIDES)


def test_a_header_line_without_a_colon_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "---\nsource_type: note\nthis is not a key value pair\n---\nBody.\n")

    with pytest.raises(CliUsageError, match="expects 'key: value'"):
        read_source(path, NO_OVERRIDES)


def test_a_naive_captured_at_in_the_file_is_refused(tmp_path: Path) -> None:
    """A local time read as UTC is a wrong observation time, so it is never guessed."""
    path = _write(tmp_path, "---\ncaptured_at: 2026-08-19T10:30:00\n---\nBody.\n")

    with pytest.raises(CliUsageError, match="must include a timezone offset"):
        read_source(path, NO_OVERRIDES)


def test_an_unparseable_captured_at_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "---\ncaptured_at: last spring\n---\nBody.\n")

    with pytest.raises(CliUsageError, match="not a valid timestamp"):
        read_source(path, NO_OVERRIDES)


def test_front_matter_with_no_body_is_refused(tmp_path: Path) -> None:
    """A file of pure provenance is not evidence."""
    path = _write(tmp_path, "---\nsource_type: conversation_note\n---\n\n   \n")

    with pytest.raises(CliUsageError, match="below its front matter"):
        read_source(path, NO_OVERRIDES)


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CliUsageError, match="no evidence content"):
        read_source(_write(tmp_path, "   \n\n"), NO_OVERRIDES)


def test_a_missing_file_names_itself(tmp_path: Path) -> None:
    with pytest.raises(CliUsageError, match="no such file"):
        read_source(tmp_path / "absent.md", NO_OVERRIDES)


def test_binary_input_is_refused_rather_than_mangled(tmp_path: Path) -> None:
    """Evidence is stored verbatim as text; lossy decoding would corrupt it."""
    path = tmp_path / "photo.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\xfd")

    with pytest.raises(CliUsageError, match="not UTF-8 text"):
        read_source(path, NO_OVERRIDES)


def test_captured_at_now_is_timezone_aware(tmp_path: Path) -> None:
    path = _write(tmp_path, "---\ncaptured_at: now\n---\nWriting this down as it happens.\n")

    captured_at = read_source(path, NO_OVERRIDES).meta.captured_at

    assert captured_at is not None
    assert captured_at.tzinfo is not None


def test_comments_and_blank_lines_in_the_header_are_ignored(tmp_path: Path) -> None:
    path = _write(tmp_path, "---\n# a note to self\n\nsource_type: debrief\n---\nBody.\n")

    assert read_source(path, NO_OVERRIDES).meta.source_type == "debrief"


def test_labelling_a_note_does_not_change_its_content(tmp_path: Path) -> None:
    """The promise that makes re-labelling free: same body, same fingerprint.

    If the split trimmed so much as the trailing newline, adding a header to an
    existing note would produce different content, a different hash, and an
    orphaned extraction run.
    """
    body = "He walked back the query planner story slightly.\n"
    plain = _write(tmp_path, body, name="plain.md")
    labelled = _write(tmp_path, f"---\nsource_label: Coffee\n---\n{body}", name="labelled.md")

    assert read_source(plain, NO_OVERRIDES).content == read_source(labelled, NO_OVERRIDES).content
