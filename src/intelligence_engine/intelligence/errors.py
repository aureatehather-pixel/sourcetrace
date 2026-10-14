"""Intelligence error model.

Same discipline as `extraction/errors.py` and `entities/errors.py`: short stable
`error_code`s, messages authored here and bounded, never built from provider or
driver text.

Note what is *not* an error. "No candidate pairs" and "every candidate was
already settled" are ordinary outcomes with telemetry attached, not failures.
`no_meaningful_relation` is likewise a correct classification, not a decline.
"""

from __future__ import annotations

import uuid

from intelligence_engine.core.errors import ApplicationError


class IntelligenceError(ApplicationError):
    """Base for anything the intelligence layer raises deliberately."""

    error_code = "intelligence_error"


class RelationNotFound(IntelligenceError):
    error_code = "relation_not_found"

    def __init__(self, relation_id: uuid.UUID) -> None:
        super().__init__(f"Intelligence relation {relation_id} not found")
        self.relation_id = relation_id


class RelationAlreadyRejected(IntelligenceError):
    """Only a current relation can be rejected.

    Rejecting twice would overwrite the first rejection's note and timestamp,
    losing the record of when the assertion actually stopped being served.
    """

    error_code = "relation_already_rejected"


class InvalidRelationOutput(IntelligenceError):
    """Classifier output did not survive validation.

    Covers non-JSON, unknown relation types, missing or mistyped fields, extra
    keys, and every batch-identity failure: an unknown pair id, a missing pair
    id, a duplicate result, a direction on a symmetric relation. Never retried
    automatically, for the M2 reason — a model that just emitted garbage usually
    emits more garbage, at full price.
    """

    error_code = "invalid_relation_output"


class IntelligenceAssemblyError(IntelligenceError):
    """A stored extraction payload could not be read back under the current schema.

    Loud on purpose, exactly as `ProfileAssemblyError` is: silently skipping the
    item would drop grounded material out of a derived assertion's provenance,
    which is the one thing this layer must never do.
    """

    error_code = "intelligence_assembly_failed"
