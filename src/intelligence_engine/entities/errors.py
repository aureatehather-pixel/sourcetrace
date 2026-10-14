"""Entity/resolution error model.

Same discipline as `extraction/errors.py`: every error carries a short, stable
`error_code` that clients can branch on. Messages are authored here, bounded,
and never built from SQL, driver or stack-trace text.

Note what is *not* an error. "The matcher found nothing" and "the matcher found
two equally good candidates" are ordinary outcomes returned as
`ResolutionOutcome(status='unresolved')` with a reason. Unresolved is a valid
state of the world, not a failure of the request.
"""

from __future__ import annotations

import uuid

from intelligence_engine.core.errors import ApplicationError


class EntityError(ApplicationError):
    """Base for anything the entity layer raises deliberately."""

    error_code = "entity_error"


class EntityNotFound(EntityError):
    error_code = "entity_not_found"

    def __init__(self, entity_id: uuid.UUID) -> None:
        super().__init__(f"Entity {entity_id} not found")
        self.entity_id = entity_id


class ExtractionItemNotFound(EntityError):
    error_code = "extraction_item_not_found"

    def __init__(self, extraction_item_id: uuid.UUID) -> None:
        super().__init__(f"Extraction item {extraction_item_id} not found")
        self.extraction_item_id = extraction_item_id


class ResolutionNotFound(EntityError):
    error_code = "resolution_not_found"

    def __init__(self, resolution_id: uuid.UUID) -> None:
        super().__init__(f"Entity resolution {resolution_id} not found")
        self.resolution_id = resolution_id


class IncompatibleResolutionRole(EntityError):
    """The role does not describe anything this item kind has.

    `claimed_by` on an `observation`, for instance: observations have no
    attribution field by design, so there is nothing to resolve.
    """

    error_code = "incompatible_resolution_role"


class ResolutionAlreadyExists(EntityError):
    """This item already has an active resolution in this role.

    Refused rather than overwritten. Changing an existing assertion is a
    correction, and corrections go through supersession so the original decision
    survives in the audit trail.
    """

    error_code = "resolution_already_exists"


class ResolutionAlreadySuperseded(EntityError):
    """Only the currently active resolution can be superseded.

    Correcting an already-corrected row would fork the history into two chains
    claiming to be current.
    """

    error_code = "resolution_already_superseded"


class ProfileAssemblyError(EntityError):
    """Stored extraction payload could not be read back under the current schema.

    Loud on purpose. Skipping the offending item would silently drop grounded
    information from a profile, which is the one thing a profile must never do.
    """

    error_code = "profile_assembly_failed"
