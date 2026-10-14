from __future__ import annotations

import argparse
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from intelligence_engine.cli.commands import (
    ENTITY_TYPE_VALUES,
    MAX_LIST_LIMIT,
    ROLE_VALUES,
    SHOW_TARGETS,
    AnalyzeCommand,
    CliUsageError,
    Command,
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
from intelligence_engine.cli.values import parse_meta_pair, parse_timestamp
from intelligence_engine.packet.schemas import MAX_PACKET_LIMIT


def _parse_uuid(value: str) -> uuid.UUID:
    """Parse a UUID string, raising CliUsageError on failure."""
    try:
        return uuid.UUID(value)
    except ValueError as err:
        raise CliUsageError(f"not a valid id: {value}") from err


class PieParser(argparse.ArgumentParser):
    """ArgumentParser that raises CliUsageError instead of calling sys.exit."""

    def error(self, message: str) -> NoReturn:
        raise CliUsageError(message)


def parse_args(argv: Sequence[str]) -> Command:
    """Parse command-line arguments into a Command dataclass."""
    if not argv:
        raise CliUsageError("expected command")

    parser = PieParser(prog="pie")
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=PieParser)

    # Common metadata flags for process and ingest
    def _add_metadata_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--source-type", default=None)
        p.add_argument("--source-label", default=None)
        p.add_argument("--source-url", default=None)
        p.add_argument("--captured-at", default=None)
        p.add_argument("--meta", action="append", default=[])

    # pie process
    process_parser = subparsers.add_parser(
        "process", help="Process a file through the full pipeline"
    )
    process_parser.add_argument("path", type=Path)
    _add_metadata_flags(process_parser)
    process_parser.add_argument(
        "--no-analyze", action="store_true", help="Skip analysis after extraction"
    )
    process_parser.add_argument(
        "--non-interactive", action="store_true", help="Run without prompting for disambiguation"
    )
    process_parser.add_argument(
        "--force-new", action="store_true", help="Force creation of new evidence even if duplicate"
    )
    process_parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of formatted text"
    )

    # pie ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a file without extraction")
    ingest_parser.add_argument("path", type=Path)
    _add_metadata_flags(ingest_parser)
    ingest_parser.add_argument(
        "--force-new", action="store_true", help="Force creation of new evidence even if duplicate"
    )
    ingest_parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of formatted text"
    )

    # pie extract
    extract_parser = subparsers.add_parser(
        "extract", help="Extract entities from existing evidence"
    )
    extract_parser.add_argument("evidence_id", type=str, metavar="EVIDENCE_ID")
    extract_parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of formatted text"
    )

    # pie entity
    entity_parser = subparsers.add_parser("entity", help="Manage entities")
    entity_subparsers = entity_parser.add_subparsers(
        dest="subcommand", required=True, parser_class=PieParser
    )

    # pie entity new
    entity_new_parser = entity_subparsers.add_parser("new", help="Create a new entity")
    entity_new_parser.add_argument(
        "--type",
        required=True,
        choices=ENTITY_TYPE_VALUES,
        help="Entity type (one of: %(choices)s)",
    )
    entity_new_parser.add_argument(
        "--name",
        required=True,
        help="Canonical name of the entity",
    )
    entity_new_parser.add_argument(
        "--disambiguator",
        default="",
        help="Optional disambiguator for the entity name",
    )
    entity_new_parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of formatted text"
    )

    # pie entity list
    entity_list_parser = entity_subparsers.add_parser("list", help="List entities")
    entity_list_parser.add_argument(
        "--type",
        choices=ENTITY_TYPE_VALUES,
        help="Filter by entity type (one of: %(choices)s)",
    )
    entity_list_parser.add_argument(
        "--limit",
        type=int,
        help=f"Maximum number of results (1-{MAX_LIST_LIMIT})",
    )
    entity_list_parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of formatted text"
    )

    # pie resolve
    resolve_parser = subparsers.add_parser(
        "resolve", help="Resolve an extraction item to an entity"
    )
    resolve_parser.add_argument("extraction_item_id", type=str, metavar="ITEM_ID")
    resolve_parser.add_argument(
        "--role",
        required=True,
        choices=ROLE_VALUES,
        help="Role of the resolution (one of: %(choices)s)",
    )
    resolve_parser.add_argument(
        "--entity",
        type=str,
        metavar="ENTITY_ID",
        help="Entity ID to resolve to (if omitted, uses deterministic matcher)",
    )
    resolve_parser.add_argument(
        "--note",
        default="",
        help="Optional note for the resolution",
    )
    resolve_parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of formatted text"
    )

    # pie analyze
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis on an entity")
    analyze_parser.add_argument("entity_id", type=str, metavar="ENTITY_ID")
    analyze_parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of formatted text"
    )

    # pie packet
    packet_parser = subparsers.add_parser(
        "packet",
        help="Emit the grounded evidence packet (read-only, no model call)",
    )
    packet_parser.add_argument(
        "--limit",
        type=int,
        help=f"Maximum number of grounded items (1-{MAX_PACKET_LIMIT})",
    )
    packet_parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of formatted text"
    )

    # pie validate
    validate_parser = subparsers.add_parser(
        "validate",
        help="Check an answer's citations against the packet (read-only, no model call)",
    )
    validate_parser.add_argument("path", type=Path, metavar="ANSWER_FILE")
    validate_parser.add_argument(
        "--limit",
        type=int,
        help=(
            f"Validate against a packet of this many items (1-{MAX_PACKET_LIMIT}) "
            "instead of the whole corpus. Use it to reproduce the packet an answer "
            "was written against; uniqueness is then scoped to those items"
        ),
    )
    validate_parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of formatted text"
    )

    # pie show
    show_parser = subparsers.add_parser("show", help="Show various types of data")
    show_parser.add_argument(
        "target",
        choices=SHOW_TARGETS,
        help="What to show (one of: %(choices)s)",
    )
    show_parser.add_argument("target_id", type=str, metavar="ID")
    show_parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of formatted text"
    )

    # pie reject
    reject_parser = subparsers.add_parser("reject", help="Reject a proposed relation")
    reject_parser.add_argument("relation_id", type=str, metavar="RELATION_ID")
    reject_parser.add_argument(
        "--note",
        default="",
        help="Optional note for the rejection",
    )
    reject_parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of formatted text"
    )

    # Parse arguments
    args = parser.parse_args(argv)

    # Process metadata flags for process and ingest
    def _process_meta(meta_list: list[str]) -> dict[str, str]:
        return dict(parse_meta_pair(item, field="--meta") for item in meta_list)

    meta_dict: dict[str, str] = {}
    if hasattr(args, "meta"):
        meta_dict = _process_meta(args.meta)

    # Dispatch to appropriate Command
    if args.command == "process":
        captured_at = (
            parse_timestamp(args.captured_at, field="--captured-at")
            if args.captured_at is not None
            else None
        )
        return ProcessCommand(
            path=args.path,
            meta=EvidenceMetadata(
                source_type=args.source_type,
                source_label=args.source_label,
                source_url=args.source_url,
                captured_at=captured_at,
                metadata=meta_dict,
            ),
            analyze=not args.no_analyze,
            interactive=not args.non_interactive,
            force_new=args.force_new,
            as_json=args.json,
        )

    if args.command == "ingest":
        captured_at = (
            parse_timestamp(args.captured_at, field="--captured-at")
            if args.captured_at is not None
            else None
        )
        return IngestCommand(
            path=args.path,
            meta=EvidenceMetadata(
                source_type=args.source_type,
                source_label=args.source_label,
                source_url=args.source_url,
                captured_at=captured_at,
                metadata=meta_dict,
            ),
            force_new=args.force_new,
            as_json=args.json,
        )

    if args.command == "extract":
        return ExtractCommand(
            evidence_id=_parse_uuid(args.evidence_id),
            as_json=args.json,
        )

    if args.command == "entity":
        if args.subcommand == "new":
            name = args.name.strip()
            if not name:
                raise CliUsageError("--name must not be empty")
            disambiguator = args.disambiguator.strip() or None
            return EntityNewCommand(
                entity_type=args.type,
                canonical_name=name,
                disambiguator=disambiguator,
                as_json=args.json,
            )
        elif args.subcommand == "list":
            limit = args.limit
            if limit is not None and not (1 <= limit <= MAX_LIST_LIMIT):
                raise CliUsageError(f"--limit must be between 1 and {MAX_LIST_LIMIT}")
            return EntityListCommand(
                entity_type=args.type,
                limit=limit,
                as_json=args.json,
            )
        else:
            raise AssertionError(f"unknown entity subcommand: {args.subcommand}")

    if args.command == "resolve":
        note = args.note.strip() or None
        entity_id = None
        if args.entity is not None:
            entity_id = _parse_uuid(args.entity)
        return ResolveCommand(
            extraction_item_id=_parse_uuid(args.extraction_item_id),
            role=args.role,
            entity_id=entity_id,
            note=note,
            as_json=args.json,
        )

    if args.command == "analyze":
        return AnalyzeCommand(
            entity_id=_parse_uuid(args.entity_id),
            as_json=args.json,
        )

    if args.command == "packet":
        packet_limit: int | None = args.limit
        if packet_limit is not None and not (1 <= packet_limit <= MAX_PACKET_LIMIT):
            raise CliUsageError(f"--limit must be between 1 and {MAX_PACKET_LIMIT}")
        return PacketCommand(limit=packet_limit, as_json=args.json)

    if args.command == "validate":
        validate_limit: int | None = args.limit
        if validate_limit is not None and not (1 <= validate_limit <= MAX_PACKET_LIMIT):
            raise CliUsageError(f"--limit must be between 1 and {MAX_PACKET_LIMIT}")
        return ValidateCommand(path=args.path, limit=validate_limit, as_json=args.json)

    if args.command == "show":
        return ShowCommand(
            target=args.target,
            target_id=_parse_uuid(args.target_id),
            as_json=args.json,
        )

    if args.command == "reject":
        note = args.note.strip() or None
        return RejectCommand(
            relation_id=_parse_uuid(args.relation_id),
            note=note,
            as_json=args.json,
        )

    raise AssertionError(f"unknown command: {args.command}")
