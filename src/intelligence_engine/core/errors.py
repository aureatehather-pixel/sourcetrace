"""Application error model.

Small and explicit on purpose. Each error exists because a route needs to
translate it into a specific HTTP response.
"""

from __future__ import annotations

import uuid


class ApplicationError(Exception):
    """Base class for errors this application raises deliberately."""


class EvidenceNotFound(ApplicationError):
    """The requested evidence item does not exist."""

    def __init__(self, evidence_id: uuid.UUID) -> None:
        super().__init__(f"Evidence item {evidence_id} not found")
        self.evidence_id = evidence_id


class PersistenceError(ApplicationError):
    """A write that was expected to succeed did not commit."""


class InvalidCursor(ApplicationError):
    """A supplied pagination cursor could not be decoded."""
