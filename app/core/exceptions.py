"""Domain exception hierarchy.

Rule: services and the ML layer raise *these* exceptions, never
``fastapi.HTTPException``. The web layer translates them in
``app/api/error_handlers.py``. This keeps the service layer importable and
testable without a web framework, and guarantees every failure reaches the
client in one consistent JSON envelope.
"""

from __future__ import annotations

from typing import Any


class MedistockError(Exception):
    """Base class for all expected, handler-mapped application failures."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class NotFoundError(MedistockError):
    """Requested entity does not exist (unknown product, batch, ...)."""

    status_code = 404
    error_code = "not_found"


class InvalidRequestError(MedistockError):
    """Semantically invalid request that schema validation cannot express."""

    status_code = 422
    error_code = "invalid_request"


class ArtifactNotReadyError(MedistockError):
    """No trained artifact bundle is available to serve predictions.

    Returned as 503 (not 500) because it is a deployment-state problem, and
    not 404 because the endpoint exists and will work once training has run.
    """

    status_code = 503
    error_code = "artifact_not_ready"


class ArtifactIntegrityError(MedistockError):
    """Artifact bundle exists but fails validation (checksum/shape/version)."""

    status_code = 500
    error_code = "artifact_integrity_error"


class FeatureNotAvailableError(MedistockError):
    """Endpoint is part of the published contract but not yet implemented.

    Used by the Phase-1 stub routes so the OpenAPI contract is complete
    without ever returning fabricated data.
    """

    status_code = 503
    error_code = "feature_not_available"


class DatabaseUnavailableError(MedistockError):
    """Database is unreachable or uninitialised."""

    status_code = 503
    error_code = "database_unavailable"
