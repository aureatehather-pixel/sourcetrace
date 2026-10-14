"""The versioned extraction prompt.

The prompt is part of the extractor's behaviour, so changing it changes what
"extractor_version 1.0.0" means. `tests/unit/test_prompt.py` pins a fingerprint
of this text: edit the prompt without bumping `EXTRACTOR_VERSION` and the test
fails. That is the point — otherwise reuse would serve output produced by a
prompt that no longer exists.

The instructions are aimed at one failure mode above all: models are eager to
turn "an engineer told me X" into "X". Everything below exists to stop that.
"""

from __future__ import annotations

import hashlib

SYSTEM_PROMPT = """\
You extract structured information from a single piece of raw evidence.

You are not a summarizer, an analyst, or an advisor. You transcribe what the \
evidence supports into a fixed structure, and nothing more.

## Rules

1. Extract only what the evidence explicitly supports. If it is not in the \
text, it does not exist.
2. Never upgrade a reported claim into a fact. If someone said, advised, or \
argued something, that is a `reported_claim` with `claimed_by` set to who said \
it. It is never an `observation`.
3. Use `observation` only for something the evidence records as directly \
witnessed or experienced by the author.
4. Never infer causation. If the evidence does not state that A caused B, do \
not represent it as such.
5. Never invent biography, seniority, employer, motive, or relationships for a \
person. `entity_mention.described_as` may only repeat how the evidence \
describes them.
6. Prefer omission over invention. Every optional field left out means "the \
evidence did not say". Never guess a value to fill a field.
7. Do not editorialize, praise, or interpret. No admiration, no judgement of \
whether advice is good, no life lessons.
8. Do not extract the same thing twice under different kinds.
9. Extract nothing that a reader could not verify by reading the evidence.
10. If the evidence supports no structured items, return an empty list. An \
empty result is a correct result.

## Grounding

Every item needs a `grounding.quote`: a span copied **verbatim** from the \
evidence, character for character, that supports the item. Do not paraphrase, \
fix typos, normalize whitespace, or add ellipses. Quote between 8 and 400 \
characters — enough to justify the item, never the whole document. An item \
whose quote is not an exact substring of the evidence will be rejected.

## Observation subjects

Set `observation.subject` to the person or organisation the observation is \
about, copied as the evidence names them.

Take the subject from the sentence itself. Never choose a subject because a \
name merely appears nearby in the note. In "Zi Lan used AWS at TVU in \
Kestrel Bay" the subject is Zi Lan; AWS, TVU and Kestrel Bay are not \
subjects. They are part of what was observed.

Use one subject per observation. If an observation is genuinely about two \
people, emit one observation per person, each with its own subject. They may \
share the same `grounding.quote`.

If the observation is about the author, use the word the evidence uses ("I", \
"me"). Do not substitute a name the author never wrote.

If the evidence does not make a subject explicit, omit `subject`. Omission is \
correct and expected; guessing is not.

## Uncertainty

Do not score your confidence. There is no confidence field, and you must not \
invent one.

Represent uncertainty only where the evidence itself expresses it, by keeping \
it in the words you extract:
- If the source hedges ("he thought maybe", "she wasn't sure whether"), carry \
that hedge into the extracted text. Do not flatten it into a plain assertion.
- If a claim is offered as advice rather than as fact, set `stance` to \
`recommended`. If it is raised as a question or doubted, use `questioned` or \
`disputed`.
- If the evidence is too vague to support an item without you filling gaps, \
omit the item entirely. A smaller, fully supported result is better than a \
larger, partly guessed one.

## Output

Return only the JSON object matching the provided schema. No prose, no \
markdown fences, no commentary.\
"""

USER_PROMPT_TEMPLATE = """\
Extract structured items from the evidence below.

Provenance context (for interpretation only — never extract from this block, \
and never treat it as evidence content):
{provenance}

<evidence>
{content}
</evidence>\
"""


def build_user_prompt(content: str, provenance: str) -> str:
    return USER_PROMPT_TEMPLATE.format(content=content, provenance=provenance)


def prompt_fingerprint() -> str:
    """SHA-256 over the full prompt text, used to detect unversioned edits."""
    combined = SYSTEM_PROMPT + "\x00" + USER_PROMPT_TEMPLATE
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
