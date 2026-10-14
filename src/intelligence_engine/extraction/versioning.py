"""Extraction pipeline version constants.

These three values, together with the evidence's `content_hash`, decide whether
a previous successful extraction can be reused instead of paying for another
model call. Everything that changes what the pipeline produces must be
reflected here, or stale output will be silently served as if it were current.

- `EXTRACTOR_VERSION` — bump when extraction *behaviour* changes: the prompt,
  the semantic checks, the provider defaults. `tests/unit/test_prompt.py` fails
  if the prompt text changes without a bump, so this cannot drift unnoticed.
- `SCHEMA_VERSION` — bump when the *output structure* changes incompatibly:
  a new required field, a removed kind, a changed field meaning.

Both are recorded on every run, so old output stays interpretable under the
version that produced it.

History:

- 3.0.1 / schema 2 - the worked example of choosing an observation's subject
  now uses invented names instead of a real person and institution. The rule it
  illustrates is unchanged, but the prompt bytes changed, so the version moves
  and runs recorded under 3.0.0 are not reused.
- 3.0.0 / schema 2 - added an optional evidence-worded subject to observations
  so downstream identity resolution has a grounded candidate to match. Schema
  2 remains current because the field defaults to `None`, so payloads written
  without it remain valid; the extractor bump prevents stale runs from reuse.
- 2.0.0 / schema 2 - removed `model_confidence`. A model's self-rating is not a
  calibrated quantity, so storing it presented precision the system does not
  have. Uncertainty is now carried only where the evidence itself expresses it.
  Both versions move: the prompt changed (behaviour) and a field was removed
  from the output (structure). Runs recorded under 1.0.0 / schema 1 are
  therefore never reused, which is exactly the intended effect.
- 1.0.0 / schema 1 - initial extractor.
"""

from __future__ import annotations

EXTRACTOR_NAME = "evidence_structurer"
EXTRACTOR_VERSION = "3.0.1"
SCHEMA_VERSION = 2
