"""Relation classifier version constants.

The M2 pattern, applied to a second derived layer. These three values decide
whether a previously persisted relation still describes what the classifier
would produce now. Everything that changes the output must move one of them, or
stale assertions are served as current intelligence.

- `CLASSIFIER_VERSION` — bump when *behaviour* changes: the prompt, the semantic
  checks, the candidate rules that decide what a pair even means.
  `tests/unit/test_relation_prompt.py` fails if the prompt text changes without
  a bump, so this cannot drift unnoticed.
- `SCHEMA_VERSION` — bump when the *output structure* changes incompatibly: a
  new relation type, a removed field, a changed field meaning.

A bump is also the only way to re-open a pair a human has rejected: the
rejection is recorded against the exact classifier version that produced the bad
answer, and a new version gets a fresh attempt.

History:

- 1.2.0 / schema 1 — the prompt stopped telling the classifier that every
  statement was "attributed to or about" the entity. A claim reaches an entity
  through `claimed_by`, which says the entity *said* it and nothing about who it
  is about; the old wording let that be read as aboutness, and on real data it
  produced three relations attributing a listener's future losses to the
  speaker's own history. Each side now carries an explicit `connection` line
  driven by the asserted resolution role, plus a rule that a speaker is not the
  subject of their own statement unless it says so. This corrects the semantics
  supplied to the classifier; it does not change the label vocabulary, the
  relation thresholds, or how conservative the classifier is asked to be.
- 1.1.0 / schema 1 — observations became reachable, so the classifier now sees a
  second statement shape and is told what it is. Each side carries an explicit
  `kind`, and the attribution line is named for what the item actually holds:
  `attributed_to` for a claim's speaker, `observed_about` for an observation's
  subject. Rendering a subject as an attribution would have told the classifier
  the subject asserted the observation — the collapse the whole reported-claim
  design prevents. Output structure is unchanged, so schema 1 stands; the
  behaviour bump is what stops relations classified under the claim-only prompt
  from being reused.
- 1.0.0 / schema 1 — initial relation classifier. Vocabulary:
  reinforces | qualifies | contradicts | no_meaningful_relation.
"""

from __future__ import annotations

CLASSIFIER_NAME = "relation_classifier"
CLASSIFIER_VERSION = "1.2.0"
SCHEMA_VERSION = 1
