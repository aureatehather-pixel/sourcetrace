"""The regression that keeps packet inclusion and M4 relation eligibility apart.

This file exists because of a specific near-miss in the handoff that produced the
packet: widening `RELATABLE_ITEM_KINDS` was proposed as *the* implementation
step. It is not, and doing it would have been a schema change wearing a filter's
clothes — that constant is rendered into CHECK constraints on
`intelligence_relations`, so widening it makes rows storable that the database
currently refuses.

The invariant, stated once:

    an item kind may enter a grounded evidence packet without thereby becoming
    eligible to stand in an M4 relation.

Three independent guards below, because one is not enough. Pinning the constant
catches a direct edit; pinning the rendered SQL catches a widening that reaches
the database; pinning the absence of the import catches the slow version, where
`packet/` starts reading M4's constant "for consistency" and the two sets fuse
into one that can only ever be widened together.
"""

from __future__ import annotations

import ast
import pathlib

from intelligence_engine.extraction.schemas import ItemKind
from intelligence_engine.intelligence.models import _RELATABLE_KIND_SQL
from intelligence_engine.intelligence.schemas import RELATABLE_ITEM_KINDS
from intelligence_engine.packet.schemas import PACKET_ITEM_KINDS

PACKET_PACKAGE = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "intelligence_engine" / "packet"
)

#: Kinds the packet must carry that M4 must still refuse. Named individually
#: rather than derived as a set difference, so this test fails if either set
#: moves — a derived expectation would quietly follow the code it is checking.
PACKET_ONLY_KINDS = frozenset(
    {ItemKind.IDEA, ItemKind.OPPORTUNITY, ItemKind.PROJECT, ItemKind.EVENT}
)


def test_m4_relation_eligibility_has_not_widened() -> None:
    """M4 still admits exactly two kinds. The literal value, not a property of it."""
    expected = frozenset({ItemKind.REPORTED_CLAIM, ItemKind.OBSERVATION})
    assert set(RELATABLE_ITEM_KINDS) == set(expected)


def test_relation_check_constraint_sql_still_names_only_two_kinds() -> None:
    """The database's own guard, not just the Python constant.

    `_RELATABLE_KIND_SQL` is interpolated into the `intelligence_relations` CHECK
    constraints. If this string grows, migration `0006` no longer describes the
    schema the models declare and `alembic check` starts reporting drift.
    """
    assert _RELATABLE_KIND_SQL == "'observation', 'reported_claim'"


def test_packet_carries_kinds_m4_must_keep_refusing() -> None:
    assert PACKET_ONLY_KINDS <= PACKET_ITEM_KINDS
    assert not (PACKET_ONLY_KINDS & RELATABLE_ITEM_KINDS)


def test_packet_is_a_strict_superset_of_m4_eligibility() -> None:
    """Everything M4 can relate is showable; the reverse does not hold."""
    assert RELATABLE_ITEM_KINDS < PACKET_ITEM_KINDS


def test_packet_excludes_kinds_that_assert_nothing() -> None:
    """`entity_mention` and `resource` name things without recording anything."""
    assert ItemKind.ENTITY_MENTION not in PACKET_ITEM_KINDS
    assert ItemKind.RESOURCE not in PACKET_ITEM_KINDS


def test_no_packet_module_imports_m4_eligibility_or_its_read_path() -> None:
    """Structural independence, checked structurally.

    Parsed rather than grepped: a comment or a docstring mentioning
    `RELATABLE_ITEM_KINDS` is fine and this file itself is full of them. An
    actual import is what fuses the two sets together.
    """
    forbidden_names = {"RELATABLE_ITEM_KINDS", "RELATABLE_ROLES"}
    forbidden_modules = {
        "intelligence_engine.intelligence.schemas",
        "intelligence_engine.intelligence.repository",
        "intelligence_engine.intelligence.candidates",
        "intelligence_engine.intelligence.models",
    }

    offenders: list[str] = []
    for path in sorted(PACKET_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module in forbidden_modules:
                    offenders.append(f"{path.name}: from {node.module}")
                for alias in node.names:
                    if alias.name in forbidden_names:
                        offenders.append(f"{path.name}: imports {alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        offenders.append(f"{path.name}: import {alias.name}")

    assert offenders == [], (
        "packet inclusion must not be coupled to M4 relation eligibility: " + "; ".join(offenders)
    )
