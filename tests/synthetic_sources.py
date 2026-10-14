"""Synthetic long-form evidence, shaped like the export that broke extraction.

The windowing tests were written against a real 60,241-character conversation
export. That document is private and is not distributed, so a checkout without
it silently skipped the only tests that exercised windowing at real scale.

This module generates a stand-in that keeps the properties that made the real
document hard, and none of its content:

- exactly 60,241 characters, so it divides into several windows at the shipped
  6,000-character budget;
- LF line endings, ~1,000 lines, ~280 paragraph breaks and lines up to a few
  hundred characters, so window boundaries land mid-paragraph and mid-line;
- several hundred multi-byte characters (em and en dashes, curly quotes,
  arrows), so a character offset and a byte offset disagree throughout, which is
  exactly what a slicing bug would get wrong.

Every name, place and event in it is invented. Generation is seeded and pure:
the same call always returns the same string.
"""

from __future__ import annotations

import functools
import random

LONG_TRANSCRIPT_CHARS = 60_241

_SPEAKERS = ("Speaker A", "Speaker B", "Facilitator")

_SUBJECTS = (
    "the sensor pipeline",
    "the booking service",
    "our demo deck",
    "the offline cache",
    "the scoring rubric",
    "the onboarding flow",
    "the queue consumer",
    "the map tiles",
)

_VERBS = (
    "kept failing under load",
    "needed another rewrite",
    "was mostly reading, not writing",
    "took longer than the estimate",
    "worked once we stopped guessing",
    "broke when the network dropped",
    "looked fine until the judges asked",
)

_ASIDES = (
    "\u201cship it and see\u201d",
    "\u201cmeasure before you tune\u201d",
    "\u201cnobody reads the logs\u201d",
    "\u201cthe first version is a question\u201d",
)


def _sentence(rng: random.Random) -> str:
    subject = rng.choice(_SUBJECTS)
    verb = rng.choice(_VERBS)
    shape = rng.randrange(5)
    if shape == 0:
        return f"{subject.capitalize()} {verb} \u2014 or so it felt at the time."
    if shape == 1:
        return f"Someone said {rng.choice(_ASIDES)}, and {subject} {verb}."
    if shape == 2:
        start = rng.randrange(1, 9)
        return f"Between hours {start}\u2013{start + rng.randrange(1, 4)}, {subject} {verb}."
    if shape == 3:
        return f"Plan: fix {subject} \u2192 rerun the demo \u2192 write it down."
    return f"{subject.capitalize()} {verb}."


def _line(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return f"- {rng.choice(_SUBJECTS)} {rng.choice(_VERBS)}"
    speaker = rng.choice(_SPEAKERS)
    count = rng.choice((1, 1, 1, 1, 2, 2, 3, 4))
    return f"{speaker}: " + " ".join(_sentence(rng) for _ in range(count))


@functools.cache
def long_transcript() -> str:
    """A deterministic, invented transcript of exactly `LONG_TRANSCRIPT_CHARS`."""
    rng = random.Random(60_241)  # noqa: S311 - a fixture, not a secret
    parts: list[str] = ["# Workshop transcript (synthetic)\n\n"]
    length = len(parts[0])
    while length < LONG_TRANSCRIPT_CHARS:
        paragraph = "\n".join(_line(rng) for _ in range(rng.randrange(1, 5))) + "\n\n"
        parts.append(paragraph)
        length += len(paragraph)
    text = "".join(parts)[: LONG_TRANSCRIPT_CHARS - 1] + "\n"
    assert len(text) == LONG_TRANSCRIPT_CHARS
    return text
