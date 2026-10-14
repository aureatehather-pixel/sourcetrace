"""Deterministic entity matching. Pure functions, no database, no model.

Two responsibilities, both deliberately dull:

1. **Normalisation** — reduce a name to a comparison key.
2. **Decision** — given the candidates that key found, decide whether to resolve.

The design goal is **precision, not recall**. A missed match costs one manual
resolution. A false merge silently fuses two real people into one identity and
contaminates every profile and every inference built on it afterwards. Those
costs are not symmetric, so the matcher declines whenever it is not certain:
zero candidates and two candidates both mean "leave it unresolved".

What normalisation deliberately does NOT do:

- ``Zi Lan`` -> ``Zilan`` — removing spaces inside a name guesses at a naming
  convention rather than observing one.
- ``Bob`` -> ``Robert`` — a nickname table is a body of claims, not normalisation.
- ``OpenAI Inc.`` -> ``OpenAI`` — stripping a legal suffix assumes the suffix is
  noise; sometimes it is the only thing separating two organisations.
- transliteration, accent folding, punctuation stripping, initial expansion.

Each would raise recall and would eventually merge two entities that are not the
same. When two spellings genuinely are one identity, that is a human assertion
recorded as a manual resolution — never a string transformation applied behind
the operator's back.
"""

from __future__ import annotations

import unicodedata
import uuid
from collections.abc import Sequence
from typing import Any, NamedTuple

from intelligence_engine.entities.schemas import ResolutionRole, UnresolvedReason
from intelligence_engine.extraction.schemas import (
    EntityMentionData,
    ItemKind,
    ObservationData,
    ReportedClaimData,
)

# The item kind each role is allowed to describe. Enforced here, and again by a
# CHECK constraint in the database - see `entities/models.py`. A role that could
# point at any kind would let an `observation` acquire an attribution the
# evidence never gave it.
ROLE_SOURCE_KINDS: dict[ResolutionRole, ItemKind] = {
    ResolutionRole.MENTION: ItemKind.ENTITY_MENTION,
    ResolutionRole.CLAIMED_BY: ItemKind.REPORTED_CLAIM,
    ResolutionRole.ABOUT: ItemKind.OBSERVATION,
}


class MatchDecision(NamedTuple):
    """Either an entity to resolve to, or a reason not to. Never both."""

    entity_id: uuid.UUID | None
    reason: UnresolvedReason | None


def normalize_name(raw: str) -> str:
    """Reduce a name to its comparison key.

    Three operations, each independently defensible:

    - **NFKC** — compatibility-composed Unicode, so a name typed with combining
      marks or full-width Latin characters compares equal to the same name typed
      normally. That is about encoding, not about the name.
    - **Whitespace collapse** — leading, trailing and repeated internal
      whitespace is typography. ``"Zi  Lan"`` is ``"Zi Lan"`` with a stray
      keystroke; it is not a different name.
    - **Casefold** — Unicode-aware lowering. ``DevGuild`` and ``devguild`` are
      the same organisation written by two people.

    Returns ``""`` for input with no non-whitespace content. Callers must treat
    an empty key as unmatchable rather than as a key that matches other empties.
    """
    composed = unicodedata.normalize("NFKC", raw)
    collapsed = " ".join(composed.split())
    return collapsed.casefold()


def matchable_name(kind: ItemKind, payload: dict[str, Any]) -> str | None:
    """The name this extraction item offers for matching, if it offers one.

    Only three kinds carry something a human would resolve to an identity:

    - ``entity_mention.name`` — an explicitly named entity.
    - ``reported_claim.claimed_by`` — an attribution string, which is often not a
      name at all. The live M2 smoke test produced ``"Engineer at DevGuild"``:
      it describes a role and identifies nobody. Such a string matches nothing
      and the claim stays unattributed, which is correct. Unnamed people remain
      unresolved, permanently if that is the truth.
    - ``observation.subject`` — who the evidence says the observation is about,
      in the evidence's own words. Like ``claimed_by`` it is a string the model
      proposed, not an identity: ``"I"`` and ``"the other team"`` are both
      legitimate values that will match nothing and must stay unresolved. It is
      read from the sentence, never inferred from names co-occurring in the same
      note — that inference is the false-merge failure this module exists to
      refuse.

    Returns ``None`` when the item offers no name — different from offering a
    name that matched nothing.

    The payload is re-validated through the M2 model rather than indexed
    directly: JSONB written under an older schema version must fail loudly here
    instead of yielding a silently wrong name.
    """
    if kind is ItemKind.ENTITY_MENTION:
        return EntityMentionData.model_validate(payload).name
    if kind is ItemKind.REPORTED_CLAIM:
        return ReportedClaimData.model_validate(payload).claimed_by
    if kind is ItemKind.OBSERVATION:
        return ObservationData.model_validate(payload).subject
    return None


def decide(match_key: str, candidate_ids: Sequence[uuid.UUID]) -> MatchDecision:
    """Turn a candidate list into a resolve-or-decline decision.

    Exactly one candidate resolves. Anything else does not:

    - **zero** — nothing to point at. The operator creates the entity and
      resolves by hand, which is a deliberate assertion that this identity
      exists rather than a side effect of a string appearing in text.
    - **two or more** — two entities share a type and a normalised name. This is
      the false-merge case: two distinct people really can both be "Alex Lee".
      The string cannot tell them apart, so neither can this function. Picking
      one by any tiebreak (first, newest, most-resolved) would be inventing an
      answer and presenting it as a finding.
    """
    if not match_key:
        return MatchDecision(None, UnresolvedReason.NO_MATCHABLE_NAME)
    if not candidate_ids:
        return MatchDecision(None, UnresolvedReason.NO_CANDIDATE)
    if len(candidate_ids) > 1:
        return MatchDecision(None, UnresolvedReason.AMBIGUOUS_CANDIDATES)
    return MatchDecision(candidate_ids[0], None)
