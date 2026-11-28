"""Structured domain exceptions for the MLOps Governance module."""

from __future__ import annotations

from typing import Any


class GovernanceError(Exception):
    """Base exception mapped to consistent JSON error envelopes."""

    status_code: int = 400
    code: str = "GOVERNANCE_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AuthenticationError(GovernanceError):
    status_code = 401
    code = "AUTHENTICATION_REQUIRED"


class PermissionDeniedError(GovernanceError):
    status_code = 403
    code = "PERMISSION_DENIED"


class ResourceNotFoundError(GovernanceError):
    status_code = 404
    code = "RESOURCE_NOT_FOUND"


class ConflictError(GovernanceError):
    status_code = 409
    code = "CONFLICT"


class InvalidTransitionError(GovernanceError):
    status_code = 409
    code = "INVALID_LIFECYCLE_TRANSITION"


class PolicyViolationError(GovernanceError):
    status_code = 422
    code = "POLICY_VIOLATION"


class ValidationError(GovernanceError):
    status_code = 422
    code = "VALIDATION_ERROR"


class AuditImmutabilityError(GovernanceError):
    status_code = 403
    code = "AUDIT_IMMUTABLE"
