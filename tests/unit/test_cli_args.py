from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from intelligence_engine.cli.args import parse_args
from intelligence_engine.cli.commands import (
    SHOW_TARGETS,
    AnalyzeCommand,
    CliUsageError,
    EntityListCommand,
    EntityNewCommand,
    EvidenceMetadata,
    ExtractCommand,
    IngestCommand,
    PacketCommand,
    ProcessCommand,
    RejectCommand,
    ResolveCommand,
    ShowCommand,
    ValidateCommand,
)

# ============================================================================
# Happy path tests - one per subcommand
# ============================================================================


def test_happy_path_process() -> None:
    cmd = parse_args(["process", "data.txt"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.path == Path("data.txt")
    assert cmd.analyze is True
    assert cmd.interactive is True
    assert cmd.force_new is False
    assert cmd.as_json is False
    assert cmd.meta == EvidenceMetadata()


def test_happy_path_ingest() -> None:
    cmd = parse_args(["ingest", "data.txt", "--force-new", "--json"])
    assert isinstance(cmd, IngestCommand)
    assert cmd.path == Path("data.txt")
    assert cmd.force_new is True
    assert cmd.as_json is True


def test_happy_path_extract() -> None:
    cmd = parse_args(["extract", "123e4567-e89b-12d3-a456-426614400000", "--json"])
    assert isinstance(cmd, ExtractCommand)
    assert cmd.evidence_id == uuid.UUID("123e4567-e89b-12d3-a456-426614400000")
    assert cmd.as_json is True


def test_happy_path_entity_new() -> None:
    cmd = parse_args(["entity", "new", "--type", "person", "--name", "John Doe", "--json"])
    assert isinstance(cmd, EntityNewCommand)
    assert cmd.entity_type == "person"
    assert cmd.canonical_name == "John Doe"
    assert cmd.disambiguator is None
    assert cmd.as_json is True


def test_happy_path_entity_new_with_disambiguator() -> None:
    cmd = parse_args(
        ["entity", "new", "--type", "organization", "--name", "ACME", "--disambiguator", "Inc"]
    )
    assert isinstance(cmd, EntityNewCommand)
    assert cmd.entity_type == "organization"
    assert cmd.canonical_name == "ACME"
    assert cmd.disambiguator == "Inc"


def test_happy_path_entity_list() -> None:
    cmd = parse_args(["entity", "list", "--type", "person", "--limit", "10", "--json"])
    assert isinstance(cmd, EntityListCommand)
    assert cmd.entity_type == "person"
    assert cmd.limit == 10
    assert cmd.as_json is True


def test_happy_path_resolve() -> None:
    cmd = parse_args(
        [
            "resolve",
            "123e4567-e89b-12d3-a456-426614400001",
            "--role",
            "mention",
            "--entity",
            "123e4567-e89b-12d3-a456-426614400002",
            "--note",
            "test note",
            "--json",
        ]
    )
    assert isinstance(cmd, ResolveCommand)
    assert cmd.extraction_item_id == uuid.UUID("123e4567-e89b-12d3-a456-426614400001")
    assert cmd.role == "mention"
    assert cmd.entity_id == uuid.UUID("123e4567-e89b-12d3-a456-426614400002")
    assert cmd.note == "test note"
    assert cmd.as_json is True


def test_happy_path_resolve_without_entity() -> None:
    cmd = parse_args(["resolve", "123e4567-e89b-12d3-a456-426614400001", "--role", "claimed_by"])
    assert isinstance(cmd, ResolveCommand)
    assert cmd.entity_id is None
    assert cmd.note is None


def test_happy_path_analyze() -> None:
    cmd = parse_args(["analyze", "123e4567-e89b-12d3-a456-426614400000", "--json"])
    assert isinstance(cmd, AnalyzeCommand)
    assert cmd.entity_id == uuid.UUID("123e4567-e89b-12d3-a456-426614400000")
    assert cmd.as_json is True


def test_happy_path_show() -> None:
    cmd = parse_args(["show", "evidence", "123e4567-e89b-12d3-a456-426614400000", "--json"])
    assert isinstance(cmd, ShowCommand)
    assert cmd.target == "evidence"
    assert cmd.target_id == uuid.UUID("123e4567-e89b-12d3-a456-426614400000")
    assert cmd.as_json is True


def test_happy_path_show_all_targets() -> None:
    for target in SHOW_TARGETS:
        cmd = parse_args(["show", target, "123e4567-e89b-12d3-a456-426614400000", "--json"])
        assert isinstance(cmd, ShowCommand)
        assert cmd.target == target


def test_happy_path_reject() -> None:
    cmd = parse_args(
        ["reject", "123e4567-e89b-12d3-a456-426614400000", "--note", "invalid", "--json"]
    )
    assert isinstance(cmd, RejectCommand)
    assert cmd.relation_id == uuid.UUID("123e4567-e89b-12d3-a456-426614400000")
    assert cmd.note == "invalid"
    assert cmd.as_json is True


# ============================================================================
# process defaults
# ============================================================================


def test_process_defaults() -> None:
    cmd = parse_args(["process", "data.txt"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.analyze is True
    assert cmd.interactive is True
    assert cmd.force_new is False
    assert cmd.as_json is False
    assert cmd.meta == EvidenceMetadata()


def test_process_no_analyze() -> None:
    cmd = parse_args(["process", "data.txt", "--no-analyze"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.analyze is False
    assert cmd.interactive is True


def test_process_non_interactive() -> None:
    cmd = parse_args(["process", "data.txt", "--non-interactive"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.analyze is True
    assert cmd.interactive is False


def test_process_both_flags() -> None:
    cmd = parse_args(["process", "data.txt", "--no-analyze", "--non-interactive"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.analyze is False
    assert cmd.interactive is False


# ============================================================================
# Validation rule tests
# ============================================================================


def test_empty_argv_raises() -> None:
    with pytest.raises(CliUsageError):
        parse_args([])


def test_unknown_command_raises() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["unknowncommand", "arg"])


def test_missing_subcommand_raises() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["process"])


def test_invalid_uuid_extraction() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["extract", "not-a-valid-uuid"])
    assert "not a valid id:" in str(exc.value)


def test_invalid_uuid_analyze() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["analyze", "invalid-uuid"])
    assert "not a valid id:" in str(exc.value)


def test_invalid_uuid_reject() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["reject", "bad-uuid"])
    assert "not a valid id:" in str(exc.value)


def test_invalid_uuid_resolve_extraction_item() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["resolve", "bad-id", "--role", "mention"])
    assert "not a valid id:" in str(exc.value)


def test_invalid_uuid_show_target_id() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["show", "evidence", "bad-uuid"])
    assert "not a valid id:" in str(exc.value)


def test_invalid_uuid_resolve_entity() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(
            [
                "resolve",
                "123e4567-e89b-12d3-a456-426614400000",
                "--role",
                "mention",
                "--entity",
                "bad",
            ]
        )
    assert "not a valid id:" in str(exc.value)


def test_captured_at_unparseable() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["process", "data.txt", "--captured-at", "not-a-timestamp"])
    assert "not a valid timestamp:" in str(exc.value)


def test_captured_at_naive_datetime() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["process", "data.txt", "--captured-at", "2026-08-22T15:30:00"])
    assert "must include a timezone offset" in str(exc.value)


def test_captured_at_now() -> None:
    cmd = parse_args(["process", "data.txt", "--captured-at", "now"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.meta.captured_at is not None
    assert cmd.meta.captured_at.tzinfo is not None


def test_captured_at_with_tz() -> None:
    cmd = parse_args(["process", "data.txt", "--captured-at", "2026-08-22T15:30:00+08:00"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.meta.captured_at is not None
    assert cmd.meta.captured_at.tzinfo is not None


def test_meta_first_equals_only() -> None:
    cmd = parse_args(["process", "data.txt", "--meta", "note=a=b"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.meta.metadata == {"note": "a=b"}


def test_meta_accumulation() -> None:
    cmd = parse_args(["process", "data.txt", "--meta", "source=auto", "--meta", "author=me"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.meta.metadata == {"source": "auto", "author": "me"}


def test_meta_last_wins() -> None:
    cmd = parse_args(["process", "data.txt", "--meta", "key=first", "--meta", "key=second"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.meta.metadata == {"key": "second"}


def test_meta_no_equals() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["process", "data.txt", "--meta", "noequals"])
    assert "--meta expects KEY=VALUE" in str(exc.value)


def test_meta_empty_key() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["process", "data.txt", "--meta", "=value"])
    assert "--meta expects KEY=VALUE" in str(exc.value)


def test_meta_empty_value_no_equals() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["process", "data.txt", "--meta", "key"])
    assert "--meta expects KEY=VALUE" in str(exc.value)


def test_meta_key_empty_vs_value_empty() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["process", "data.txt", "--meta", "=value"])
    assert "--meta expects KEY=VALUE" in str(exc.value)


def test_meta_value_with_equals() -> None:
    cmd = parse_args(["process", "data.txt", "--meta", "key=value=with=equals"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.meta.metadata == {"key": "value=with=equals"}


def test_type_invalid_choice() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["entity", "new", "--type", "invalid_type", "--name", "Test"])


def test_role_required() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["resolve", "123e4567-e89b-12d3-a456-426614400000"])
    assert "required" in str(exc.value)


def test_show_target_invalid() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["show", "invalid_target", "123e4567-e89b-12d3-a456-426614400000"])


def test_limit_zero() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["entity", "list", "--limit", "0"])
    assert "--limit must be between 1 and 500" in str(exc.value)


def test_limit_negative() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["entity", "list", "--limit", "-5"])
    assert "--limit must be between 1 and 500" in str(exc.value)


def test_limit_above_max() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["entity", "list", "--limit", "501"])
    assert "--limit must be between 1 and 500" in str(exc.value)


def test_limit_valid() -> None:
    cmd = parse_args(["entity", "list", "--limit", "500"])
    assert isinstance(cmd, EntityListCommand)
    assert cmd.limit == 500


def test_limit_valid_one() -> None:
    cmd = parse_args(["entity", "list", "--limit", "1"])
    assert isinstance(cmd, EntityListCommand)
    assert cmd.limit == 1


def test_limit_none_when_absent() -> None:
    cmd = parse_args(["entity", "list"])
    assert isinstance(cmd, EntityListCommand)
    assert cmd.limit is None


def test_name_empty() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["entity", "new", "--type", "person", "--name", ""])
    assert "--name must not be empty" in str(exc.value)


def test_name_whitespace_only() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["entity", "new", "--type", "person", "--name", "   "])
    assert "--name must not be empty" in str(exc.value)


def test_name_stripped() -> None:
    cmd = parse_args(["entity", "new", "--type", "person", "--name", "  John Doe  "])
    assert isinstance(cmd, EntityNewCommand)
    assert cmd.canonical_name == "John Doe"


def test_disambiguator_empty_becomes_none() -> None:
    cmd = parse_args(["entity", "new", "--type", "person", "--name", "John", "--disambiguator", ""])
    assert isinstance(cmd, EntityNewCommand)
    assert cmd.disambiguator is None


def test_disambiguator_whitespace_becomes_none() -> None:
    cmd = parse_args(
        ["entity", "new", "--type", "person", "--name", "John", "--disambiguator", "   "]
    )
    assert isinstance(cmd, EntityNewCommand)
    assert cmd.disambiguator is None


def test_disambiguator_stripped() -> None:
    cmd = parse_args(
        ["entity", "new", "--type", "person", "--name", "John", "--disambiguator", "  Inc  "]
    )
    assert isinstance(cmd, EntityNewCommand)
    assert cmd.disambiguator == "Inc"


def test_note_empty_becomes_none() -> None:
    cmd = parse_args(
        ["resolve", "123e4567-e89b-12d3-a456-426614400000", "--role", "mention", "--note", ""]
    )
    assert isinstance(cmd, ResolveCommand)
    assert cmd.note is None


def test_note_whitespace_becomes_none() -> None:
    cmd = parse_args(
        ["resolve", "123e4567-e89b-12d3-a456-426614400000", "--role", "mention", "--note", "   "]
    )
    assert isinstance(cmd, ResolveCommand)
    assert cmd.note is None


def test_note_stripped() -> None:
    cmd = parse_args(
        [
            "resolve",
            "123e4567-e89b-12d3-a456-426614400000",
            "--role",
            "mention",
            "--note",
            "  test  ",
        ]
    )
    assert isinstance(cmd, ResolveCommand)
    assert cmd.note == "test"


def test_reject_note_validation() -> None:
    cmd = parse_args(["reject", "123e4567-e89b-12d3-a456-426614400000", "--note", "  "])
    assert isinstance(cmd, RejectCommand)
    assert cmd.note is None


# ============================================================================
# Source flags passed through verbatim
# ============================================================================


def test_source_type_passthrough() -> None:
    cmd = parse_args(["process", "data.txt", "--source-type", "web"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.meta.source_type == "web"


def test_source_label_passthrough() -> None:
    cmd = parse_args(["process", "data.txt", "--source-label", "Test Label"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.meta.source_label == "Test Label"


def test_source_url_passthrough() -> None:
    cmd = parse_args(["process", "data.txt", "--source-url", "https://example.com"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.meta.source_url == "https://example.com"


def test_source_flags_none_when_absent() -> None:
    cmd = parse_args(["process", "data.txt"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.meta.source_type is None
    assert cmd.meta.source_label is None
    assert cmd.meta.source_url is None


def test_source_flags_ingest() -> None:
    cmd = parse_args(
        [
            "ingest",
            "data.txt",
            "--source-type",
            "import",
            "--source-label",
            "Archive",
            "--source-url",
            "https://archive.example.com",
        ]
    )
    assert isinstance(cmd, IngestCommand)
    assert cmd.meta.source_type == "import"
    assert cmd.meta.source_label == "Archive"
    assert cmd.meta.source_url == "https://archive.example.com"


# ============================================================================
# json flag defaults
# ============================================================================


def test_json_defaults_false() -> None:
    cmd = parse_args(["process", "data.txt"])
    assert isinstance(cmd, ProcessCommand)
    assert cmd.as_json is False

    cmd = parse_args(["ingest", "data.txt"])
    assert isinstance(cmd, IngestCommand)
    assert cmd.as_json is False

    cmd = parse_args(["extract", "123e4567-e89b-12d3-a456-426614400000"])
    assert isinstance(cmd, ExtractCommand)
    assert cmd.as_json is False

    cmd = parse_args(["analyze", "123e4567-e89b-12d3-a456-426614400000"])
    assert isinstance(cmd, AnalyzeCommand)
    assert cmd.as_json is False

    cmd = parse_args(["reject", "123e4567-e89b-12d3-a456-426614400000"])
    assert isinstance(cmd, RejectCommand)
    assert cmd.as_json is False


# ============================================================================
# Command not found - requires unknown subcommand or missing required args
# ============================================================================


def test_process_missing_path() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["process"])


def test_ingest_missing_path() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["ingest"])


def test_extract_missing_id() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["extract"])


def test_entity_new_missing_type() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["entity", "new", "--name", "Test"])


def test_entity_new_missing_name() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["entity", "new", "--type", "person"])


def test_resolve_missing_extraction_item_id() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["resolve", "--role", "mention"])


def test_analyze_missing_entity_id() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["analyze"])


def test_show_missing_target() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["show"])


def test_reject_missing_relation_id() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["reject"])


# ============================================================================
# pie packet - the read-only, provider-independent verb
# ============================================================================


def test_happy_path_packet() -> None:
    cmd = parse_args(["packet"])
    assert isinstance(cmd, PacketCommand)
    assert cmd.limit is None
    assert cmd.as_json is False


def test_packet_json_and_limit() -> None:
    cmd = parse_args(["packet", "--limit", "25", "--json"])
    assert isinstance(cmd, PacketCommand)
    assert cmd.limit == 25
    assert cmd.as_json is True


def test_packet_limit_zero_is_rejected() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["packet", "--limit", "0"])
    assert "--limit must be between 1 and 1000" in str(exc.value)


def test_packet_limit_above_the_ceiling_is_rejected() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["packet", "--limit", "1001"])


def test_packet_takes_no_topic_or_entity_argument() -> None:
    """Deliberate. A lexical topic gate is the filter `candidates.py` rejected.

    Pinned so the option cannot be added without a decision being made about it:
    the corpus's most valuable pair shares zero content words, so a topic filter
    would silently drop the evidence most worth reading.
    """
    with pytest.raises(CliUsageError):
        parse_args(["packet", "hackathons"])


# ============================================================================
# pie validate - the citation gate
# ============================================================================


def test_happy_path_validate() -> None:
    cmd = parse_args(["validate", "answer.md"])
    assert isinstance(cmd, ValidateCommand)
    assert cmd.path == Path("answer.md")
    assert cmd.as_json is False


def test_validate_json() -> None:
    cmd = parse_args(["validate", "answer.md", "--json"])
    assert isinstance(cmd, ValidateCommand)
    assert cmd.as_json is True


def test_validate_requires_a_file() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["validate"])


def test_validate_limit_defaults_to_the_whole_corpus() -> None:
    """`None` means every stored item, not `DEFAULT_PACKET_LIMIT`.

    This command used to have no limit at all and inherited the packet default,
    so an answer written against 496 items was checked against 200 and
    packet-wide uniqueness was computed over a prefix. `None` here is the fix,
    not an unset field.
    """
    cmd = parse_args(["validate", "answer.md"])
    assert isinstance(cmd, ValidateCommand)
    assert cmd.limit is None


def test_validate_accepts_an_explicit_limit() -> None:
    cmd = parse_args(["validate", "answer.md", "--limit", "496"])
    assert isinstance(cmd, ValidateCommand)
    assert cmd.limit == 496


def test_validate_limit_zero_is_rejected() -> None:
    with pytest.raises(CliUsageError) as exc:
        parse_args(["validate", "answer.md", "--limit", "0"])
    assert "--limit must be between 1 and 1000" in str(exc.value)


def test_validate_limit_above_the_ceiling_is_rejected() -> None:
    with pytest.raises(CliUsageError):
        parse_args(["validate", "answer.md", "--limit", "1001"])
