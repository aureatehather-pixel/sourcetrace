"""The example notes are the demo's evidence, read the way `pie` reads them.

`examples/notes/` is what a reader with an API key runs `pie process` on, and
`demo/run_demo.py` is what everyone else runs. If the two drifted, the README
would describe one corpus and the demo would show another.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from demo.run_demo import FOLLOW_UP, REVIEW, WORKSHOP

from intelligence_engine.cli.commands import EvidenceMetadata
from intelligence_engine.cli.source import read_source

NOTES = Path(__file__).resolve().parents[2] / "examples" / "notes"


@pytest.mark.parametrize(
    ("name", "content", "label"),
    [
        ("01-workshop.md", WORKSHOP, "Harbour Hall maker workshop"),
        ("02-follow-up.md", FOLLOW_UP, "Follow-up call"),
        ("03-design-review.md", REVIEW, "Design review"),
    ],
    ids=["workshop", "follow-up", "design-review"],
)
def test_each_example_note_is_the_demo_evidence(name: str, content: str, label: str) -> None:
    document = read_source(NOTES / name, EvidenceMetadata())

    assert document.content == content
    assert document.meta.source_label == label
    assert isinstance(document.meta.captured_at, dt.datetime)
    assert document.meta.captured_at.tzinfo is not None
