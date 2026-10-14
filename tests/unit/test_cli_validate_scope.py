"""What packet `pie validate` checks an answer against, and why it matters.

`_validate` used to call `assemble()` with no argument, inheriting
`DEFAULT_PACKET_LIMIT = 200`, while `pie packet --limit 1000` could hand the
model 496 items. The two ends of the loop therefore disagreed about what "the
packet" was, and nothing said so.

The first test here is the reason that is a defect rather than an inconvenience.
Unique binding is a **packet-wide** property: a fragment is accepted only if no
*other* item in the packet quotes it too. Shrink the packet and the collision set
shrinks with it, so the same answer, unchanged, passes over a prefix and fails
over the corpus. A validator whose verdict depends on an invisible default is
reporting a weaker guarantee under the stronger one's name - failure 3 in
`citations.py`, reintroduced by the validator instead of by the model.

The remaining tests pin the fix: the whole corpus by default, an explicit
`--limit` when an answer must be checked against the packet it was really
written against, and a refusal - never a quiet prefix - when neither is possible.
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from intelligence_engine.cli import _validate
from intelligence_engine.cli.commands import ValidateCommand
from intelligence_engine.cli.context import Services
from intelligence_engine.extraction.schemas import ItemKind
from intelligence_engine.packet.errors import CorpusExceedsPacketLimit, InvalidCitation
from intelligence_engine.packet.schemas import (
    MAX_PACKET_LIMIT,
    EvidencePacket,
    PacketItem,
    PacketSource,
)

EVIDENCE_ID = uuid.UUID("0df5980f-cbe6-4e99-881e-8e135f77c970")
RUN_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")

# Long enough to be a legal citation fragment on its own, and shared verbatim by
# two items that sit on opposite sides of the old 200-item cut.
COLLIDING = "the same sentence in two different rows"


def _item(index: int, quote: str) -> PacketItem:
    return PacketItem(
        item_id=uuid.UUID(int=index, version=4),
        kind=ItemKind.IDEA,
        text="the item's statement",
        stance=None,
        claimed_by=None,
        subject=None,
        attributes=(),
        grounding_quote=quote,
        extraction_item_position=index,
        extraction_run_id=RUN_ID,
        evidence_id=EVIDENCE_ID,
        source_label=None,
        captured_at=None,
        entities=(),
    )


def _packet(items: tuple[PacketItem, ...], *, total: int) -> EvidencePacket:
    return EvidencePacket(
        items=items,
        sources=(
            PacketSource(
                evidence_id=EVIDENCE_ID,
                source_type=None,
                source_label=None,
                source_url=None,
                captured_at=None,
                created_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
                item_count=len(items),
            ),
        ),
        total_items_available=total,
        truncated=total > len(items),
    )


def _corpus() -> tuple[PacketItem, ...]:
    """250 items where E6 and E210 share a quote - one on each side of 200."""
    items = [_item(i, f"grounding quote number {i} for this item") for i in range(250)]
    items[5] = _item(5, COLLIDING)
    items[209] = _item(209, COLLIDING)
    return tuple(items)


def _plain(count: int) -> tuple[PacketItem, ...]:
    return tuple(_item(i, f"grounding quote number {i} for this item") for i in range(count))


class _StubPacketService:
    """Records the limit it was asked for. The defect was that number."""

    def __init__(self, items: tuple[PacketItem, ...]) -> None:
        self._items = items
        self.limits: list[int] = []

    async def assemble(self, *, limit: int = 200) -> EvidencePacket:
        self.limits.append(limit)
        return _packet(self._items[:limit], total=len(self._items))


def _services(items: tuple[PacketItem, ...]) -> tuple[Services, _StubPacketService]:
    packet_service = _StubPacketService(items)
    return cast(Services, SimpleNamespace(packet=packet_service)), packet_service


def _answer(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "answer.md"
    path.write_text(text, encoding="utf-8")
    return path


UNCITED = "No citations, nothing to bind.\n"


# ============================================================================
# The defect: the same answer gets two different verdicts
# ============================================================================


async def test_a_truncated_packet_accepts_a_citation_the_corpus_rejects(tmp_path: Path) -> None:
    """The false accept, demonstrated rather than argued.

    `E6` and `E210` carry byte-identical quotes. Over the first 200 items `E210`
    does not exist, so the fragment looks unique and the citation binds. Over all
    250 it collides, and the citation is exactly the ambiguous kind the
    uniqueness check exists to reject. Nothing about the answer changed.
    """
    corpus = _corpus()
    answer = _answer(tmp_path, f'The finding stands. [E6 "{COLLIDING}"]\n')

    services, _ = _services(corpus)
    assert await _validate(services, ValidateCommand(path=answer, limit=200)) == 0

    services, packet_service = _services(corpus)
    with pytest.raises(InvalidCitation):
        await _validate(services, ValidateCommand(path=answer))
    assert packet_service.limits == [MAX_PACKET_LIMIT]


# ============================================================================
# The fix
# ============================================================================


async def test_the_default_scope_is_the_whole_corpus(tmp_path: Path) -> None:
    """Pins the regression. `200` here would be the bug, not a tuning choice."""
    services, packet_service = _services(_corpus())

    assert await _validate(services, ValidateCommand(path=_answer(tmp_path, UNCITED))) == 0
    assert packet_service.limits == [MAX_PACKET_LIMIT]


async def test_an_explicit_limit_is_honoured(tmp_path: Path) -> None:
    """Narrowing the scope on purpose stays possible - it is how an answer gets
    checked against the packet it was actually written against."""
    services, packet_service = _services(_corpus())

    assert (
        await _validate(services, ValidateCommand(path=_answer(tmp_path, UNCITED), limit=46)) == 0
    )
    assert packet_service.limits == [46]


async def test_a_narrowed_scope_is_named_in_the_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A verdict over a prefix must not read like a verdict over the corpus."""
    answer = _answer(tmp_path, UNCITED)

    services, _ = _services(_corpus())
    await _validate(services, ValidateCommand(path=answer, limit=46))
    assert "TRUNCATED" in capsys.readouterr().out

    services, _ = _services(_plain(10))
    await _validate(services, ValidateCommand(path=answer))
    assert "complete corpus" in capsys.readouterr().out


async def test_a_corpus_too_large_to_assemble_is_refused_not_truncated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The negative case. Past `MAX_PACKET_LIMIT` there is no whole-corpus
    packet, and this must refuse rather than return the weaker verdict silently.
    """
    services, _ = _services(_plain(MAX_PACKET_LIMIT + 1))

    with pytest.raises(CorpusExceedsPacketLimit) as exc:
        await _validate(services, ValidateCommand(path=_answer(tmp_path, UNCITED)))

    message = str(exc.value)
    assert str(MAX_PACKET_LIMIT) in message
    assert "--limit" in message
    assert capsys.readouterr().out == ""


async def test_an_explicit_limit_still_validates_an_oversized_corpus(tmp_path: Path) -> None:
    """The refusal is about the *silent* default, not about truncation itself."""
    services, packet_service = _services(_plain(MAX_PACKET_LIMIT + 1))
    command = ValidateCommand(path=_answer(tmp_path, UNCITED), limit=MAX_PACKET_LIMIT)

    assert await _validate(services, command) == 0
    assert packet_service.limits == [MAX_PACKET_LIMIT]
