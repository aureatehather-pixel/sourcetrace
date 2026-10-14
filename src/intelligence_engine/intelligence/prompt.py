"""The versioned relation-classification prompt.

The prompt is part of the classifier's behaviour, so changing it changes what
"classifier_version 1.0.0" means. `tests/unit/test_relation_prompt.py` pins a
fingerprint of this text: edit the prompt without bumping `CLASSIFIER_VERSION`
and the test fails. Without that gate, reuse would serve relations produced by a
prompt that no longer exists.

The instructions are aimed at one failure mode above all. M2's was models turning
"an engineer told me X" into "X". M4's is models *finding relationships*, because
finding one feels like doing the job and saying "these two claims have nothing to
do with each other" feels like failing at it. Everything below exists to make
abstention feel like the right answer, because most of the time it is.
"""

from __future__ import annotations

import hashlib

from intelligence_engine.entities.schemas import ResolutionRole
from intelligence_engine.extraction.schemas import ItemKind
from intelligence_engine.intelligence.candidates import CandidatePair, RelatableItem

SYSTEM_PROMPT = """\
You classify the relationship between pairs of grounded statements that were \
extracted from someone's notes. Each statement was already checked against its \
source text; you are not being asked whether it is true.

You are not a summarizer, an analyst or an advisor. For each pair you choose one \
label and, when the label calls for it, write one short sentence explaining it.

## Labels

`reinforces` - the two statements materially support each other. One \
independently backs up, strengthens or supplies concrete support for the other. \
Restating the same thing in different words is NOT reinforcement; there must be \
something added.

`qualifies` - one statement adds an important limitation, condition, phase, \
scope or cost to the other. The broader statement still stands; it is narrower \
or better understood than before. Qualification is not disagreement.

`contradicts` - the two cannot straightforwardly both hold, or one explicitly \
disputes or retracts the other. Use this only when you cannot construct a \
reasonable reading in which both are true.

`no_meaningful_relation` - anything else. Includes pairs that are merely about \
the same topic, share vocabulary, or are both true of the same person without \
bearing on each other.

## What matters more than getting a label right

These mistakes are not equally bad. In order, worst first:

1. Inventing a causal link. Never say or imply that one thing caused, led to or \
explains the other. Two statements sitting next to each other is not causation, \
and neither is one happening before the other.
2. A false `contradicts`. Two statements about different times, different \
stages, different scopes, different people or different situations usually do \
NOT contradict. Someone revising or adding detail to their own earlier statement \
is `qualifies`, not `contradicts`.
3. A false `reinforces` or `qualifies` - claiming a relationship that is not \
really there.
4. A technically true but useless relationship - "both are about software", \
"both mention the same project". These are worth nothing. Label them \
`no_meaningful_relation`.
5. Missing a real relationship.

Missing one is the cheapest mistake on this list. `no_meaningful_relation` is a \
correct, common and useful answer. Expect to use it for most pairs. Do not \
manufacture a relationship because a pair was shown to you - pairs are selected \
by a cheap automatic filter that knows nothing about meaning.

## Statement kinds and attribution

Each statement is one of two kinds, and the difference matters.

`reported statement` - somebody asserted this. It carries `attributed_to`: who \
said it. It is a record of a position taken, not of a fact established.

`direct observation` - the evidence records this as witnessed. It carries \
`observed_about`: who or what it concerns. **Nobody asserted it.** Never treat \
an observation as something its subject said, claimed or endorsed. "Her profile \
lists Rust" is not her telling you she knows Rust.

## Who a statement is ABOUT is not who SAID it

Every statement below is connected to one entity, and its `connection` line says \
how. Read that line before you reason about the pair. The two connections mean \
different things:

- `SAID BY` - that entity asserted this. **The statement may be about somebody \
else entirely**, and very often is about the person they were speaking to.
- `OBSERVED ABOUT` - the evidence records this as being about that entity. \
Nobody asserted it.

**Never assume a speaker is talking about themselves.** "She told me that I \
would lose my first few hackathons" is a statement about the listener, not about \
the speaker. Treating it as a report of the speaker's own history, and then \
relating it to facts about the speaker, is a fabrication of the same kind as \
inventing causation - it asserts something nobody said.

When a `SAID BY` statement is about somebody other than the entity it is \
connected to, it can still stand in a relation, but only on what it actually \
says. If the only way you can connect two statements is by assuming the speaker \
was describing themselves, the answer is `no_meaningful_relation`.

If two statements come from different people, do not treat them as one person's \
position: two people independently saying compatible things can still be \
`reinforces`, but one person cannot qualify or retract another person's \
statement.

An observation and a reported statement can stand in any of the relations. An \
observation can reinforce or qualify what somebody said, because it is \
independent material rather than that person restating themselves. It cannot \
retract their statement - only the speaker can do that - so prefer `qualifies` \
over `contradicts` when an observation sits awkwardly beside a claim.

## Context

Each statement carries where and when it was recorded. Use it. A later \
conversation revising an earlier one is normal, and it is `qualifies` when it \
narrows the earlier statement and `contradicts` only when it genuinely cannot \
stand beside it.

## Explanation

For `reinforces`, `qualifies` and `contradicts`, write one sentence, under 400 \
characters, saying what the relationship actually is. Rules:

- Refer only to what the two statements say. No outside knowledge, no invented \
motives, no guesses about the person.
- Say something a reader could not get by reading the two statements side by \
side. "Both discuss learning" is worthless.
- Do not restate the statements. Do not give advice or judge whether the advice \
is good.
- Never write a causal explanation.

For `no_meaningful_relation`, omit the explanation entirely.

## Direction

`qualifies` needs a direction: `a_qualifies_b` if statement A is the one adding \
the limitation, `b_qualifies_a` if it is statement B. Every other label must \
omit the direction field.

## Output

Return one result for every pair you were given, keyed by its exact `pair_id`. \
Do not add pairs, drop pairs or invent ids. Return only the JSON object matching \
the provided schema - no prose, no markdown fences, no commentary.\
"""

USER_PROMPT_TEMPLATE = """\
Every statement below is connected to this entity: {entity_label}

They are NOT all about that entity. Each statement's `connection` line says \
whether the entity said it or whether it was observed about them. Use it.

Classify each pair.

{pairs}\
"""

_PAIR_TEMPLATE = """\
{pair_id}
  A:
    kind: {left_kind}
    connection: {left_connection}
    statement: {left_text}
    offered_as: {left_stance}
    {left_attribution}
    source_text: "{left_quote}"
    recorded: {left_context}
  B:
    kind: {right_kind}
    connection: {right_connection}
    statement: {right_text}
    offered_as: {right_stance}
    {right_attribution}
    source_text: "{right_quote}"
    recorded: {right_context}\
"""

_ABSENT = "(not stated)"


def _describe_context(item: RelatableItem) -> str:
    """Where and when this statement was recorded.

    Absent parts are omitted rather than filled with "unknown": inviting a model
    to reason about an unknown is how invented context gets written down. What
    remains is enough to tell two occasions apart, which is what stops a revision
    from reading as a contradiction.
    """
    parts: list[str] = []
    if item.source_label:
        parts.append(item.source_label)
    if item.captured_at:
        parts.append(item.captured_at.date().isoformat())
    return ", ".join(parts) if parts else _ABSENT


_KIND_LABELS: dict[ItemKind, str] = {
    ItemKind.REPORTED_CLAIM: "reported statement",
    ItemKind.OBSERVATION: "direct observation",
}


def _describe_kind(item: RelatableItem) -> str:
    """The statement kind, named in the words the system prompt defines."""
    label = _KIND_LABELS.get(item.kind)
    if label is None:
        # Unreachable via RELATABLE_ITEM_KINDS. Loud rather than mislabelled:
        # showing the classifier a kind it was never told about is worse than
        # failing.
        raise ValueError(f"{item.kind.value} cannot be rendered as a classifier statement")
    return label


_CONNECTIONS: dict[ResolutionRole, str] = {
    ResolutionRole.CLAIMED_BY: "SAID BY the entity above - not necessarily about them",
    ResolutionRole.ABOUT: "OBSERVED ABOUT the entity above",
}


def _describe_connection(item: RelatableItem) -> str:
    """How this statement is connected to the entity being analysed.

    Driven by the role M3 actually asserted, never inferred from the item kind.
    This line exists because the previous prompt told the classifier every
    statement was "attributed to or about" the entity, which let a claim somebody
    *made* be read as a fact *about* them - and it duly produced three relations
    attributing a listener's future losses to the speaker's own history.
    """
    connection = _CONNECTIONS.get(item.role)
    if connection is None:
        # Unreachable via RELATABLE_ROLES. Loud rather than mislabelled: guessing
        # a connection is the failure this whole line exists to prevent.
        raise ValueError(f"{item.role.value} has no defined classifier connection")
    return connection


def _describe_attribution(item: RelatableItem) -> str:
    """The attribution line, named for what the item actually carries.

    A claim gets `attributed_to`, an observation gets `observed_about`. Rendering
    an observation's subject under `attributed_to` would tell the classifier the
    subject asserted it - the exact collapse the extraction schema, the role/kind
    CHECK and the system prompt all exist to prevent.
    """
    if item.kind is ItemKind.OBSERVATION:
        return f"observed_about: {item.subject or _ABSENT}"
    return f"attributed_to: {item.claimed_by or _ABSENT}"


def render_pair(pair_id: str, pair: CandidatePair) -> str:
    """One pair block, exactly as the classifier sees it."""
    return _PAIR_TEMPLATE.format(
        pair_id=pair_id,
        left_kind=_describe_kind(pair.left),
        left_connection=_describe_connection(pair.left),
        left_text=pair.left.text,
        left_stance=pair.left.stance or _ABSENT,
        left_attribution=_describe_attribution(pair.left),
        left_quote=pair.left.grounding_quote,
        left_context=_describe_context(pair.left),
        right_kind=_describe_kind(pair.right),
        right_connection=_describe_connection(pair.right),
        right_text=pair.right.text,
        right_stance=pair.right.stance or _ABSENT,
        right_attribution=_describe_attribution(pair.right),
        right_quote=pair.right.grounding_quote,
        right_context=_describe_context(pair.right),
    )


def build_user_prompt(*, entity_label: str, rendered_pairs: list[str]) -> str:
    return USER_PROMPT_TEMPLATE.format(entity_label=entity_label, pairs="\n\n".join(rendered_pairs))


def pair_fingerprint(pair: CandidatePair) -> str:
    """SHA-256 over exactly what the classifier is told about this pair.

    The `pair_id` is deliberately excluded: it is a per-batch label, so including
    it would make the same pair fingerprint differently depending on where it
    happened to land in a batch, and reuse would break for no reason.

    What this proves is worth stating plainly. M2 extraction items are immutable —
    re-extraction writes new rows rather than editing old ones — so two item ids
    already determine the content. This fingerprint is the check on that
    assumption rather than a substitute for it: it is part of the reuse lookup, so
    if the content behind a pair ever did change, the lookup would miss and the
    unique constraint would raise, instead of stale intelligence being served
    quietly.
    """
    return hashlib.sha256(render_pair("pair", pair).encode("utf-8")).hexdigest()


def prompt_fingerprint() -> str:
    """SHA-256 over the full prompt text, used to detect unversioned edits."""
    combined = "\x00".join((SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, _PAIR_TEMPLATE))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
