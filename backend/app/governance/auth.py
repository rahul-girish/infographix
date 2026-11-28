"""Pluggable authentication & RBAC enforcement for governance routes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.governance.config import governance_settings
from backend.app.governance.constants import ROLE_PERMISSIONS, ROLES
from backend.app.governance.exceptions import AuthenticationError, PermissionDeniedError
from backend.app.governance.models import GovernanceRoleAssignment
from backend.app.governance.services.audit_service import AuditService


@dataclass(frozen=True)
class ActorContext:
    """Authenticated caller identity, effective role, and permissions."""

    actor_id: str
    role: str
    auth_source: str
    permissions: frozenset[str]
    correlation_id: str


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * ((4 - len(segment) % 4) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def verify_hs256_jwt(token: str, secret: str, expected_audience: str = "") -> dict:
    """Verify an HS256 JWT without external dependencies (Supabase/standard JWT compatible)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthenticationError("Malformed Bearer JWT.")
    header_b64, payload_b64, sig_b64 = parts

    try:
        header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        provided_sig = _b64url_decode(sig_b64)
    except Exception as exc:
        raise AuthenticationError("Invalid JWT encoding.") from exc

    if header.get("alg") != "HS256":
        raise AuthenticationError("Unsupported JWT algorithm; expected HS256.")

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise AuthenticationError("Invalid JWT signature.")

    now = time.time()
    exp = payload.get("exp")
    if exp is not None:
        try:
            exp_val = float(exp)
        except (TypeError, ValueError) as exc:
            raise AuthenticationError("Invalid JWT 'exp' claim.") from exc
        if exp_val < now:
            raise AuthenticationError("JWT has expired.")

    nbf = payload.get("nbf")
    if nbf is not None:
        try:
            nbf_val = float(nbf)
        except (TypeError, ValueError) as exc:
            raise AuthenticationError("Invalid JWT 'nbf' claim.") from exc
        if nbf_val > now:
            raise AuthenticationError("JWT is not yet valid ('nbf').")

    if expected_audience:
        aud = payload.get("aud")
        valid_aud = (aud == expected_audience) or (
            isinstance(aud, list) and expected_audience in aud
        )
        if not valid_aud:
            raise AuthenticationError("JWT audience mismatch.")

    sub = payload.get("sub")
    if not sub or not isinstance(sub, str):
        raise AuthenticationError("JWT missing 'sub' subject claim.")

    return payload


def _resolve_db_role(db: Session, actor_id: str) -> Optional[str]:
    row = (
        db.query(GovernanceRoleAssignment)
        .filter(
            GovernanceRoleAssignment.actor_id == actor_id,
            GovernanceRoleAssignment.is_active.is_(True),
        )
        .first()
    )
    if row and row.role in ROLES:
        return row.role
    return None


def get_current_actor(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_governance_token: Optional[str] = Header(default=None),
    x_actor_id: Optional[str] = Header(default=None),
    x_actor_role: Optional[str] = Header(default=None),
    x_correlation_id: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> ActorContext:
    """Resolve authenticated actor from Service Token, Bearer JWT, or explicit dev-mode headers."""
    correlation_id = (x_correlation_id or "").strip() or str(uuid.uuid4())
    request.state.correlation_id = correlation_id

    token: Optional[str] = None
    if x_governance_token and x_governance_token.strip():
        token = x_governance_token.strip()
    elif authorization and authorization.strip().lower().startswith("bearer "):
        token = authorization.strip()[7:].strip()

    # 1. Scoped Service API Token
    if token:
        service_tokens = governance_settings.parsed_service_tokens()
        for configured_token, meta in service_tokens.items():
            if hmac.compare_digest(token, configured_token):
                role = meta["role"]
                if role not in ROLES:
                    raise AuthenticationError("Configured service token has invalid role.")
                return ActorContext(
                    actor_id=meta["actor_id"],
                    role=role,
                    auth_source="service_token",
                    permissions=ROLE_PERMISSIONS[role],
                    correlation_id=correlation_id,
                )

        # 2. HS256 Bearer JWT
        jwt_secret = (governance_settings.GOV_JWT_SECRET or "").strip()
        if jwt_secret and "." in token:
            claims = verify_hs256_jwt(
                token,
                secret=jwt_secret,
                expected_audience=(governance_settings.GOV_JWT_AUDIENCE or "").strip(),
            )
            actor_id = str(claims["sub"])
            db_role = _resolve_db_role(db, actor_id)
            claim_role = claims.get("gov_role") or claims.get("role")
            effective_role = db_role or (claim_role if claim_role in ROLES else "viewer")
            return ActorContext(
                actor_id=actor_id,
                role=effective_role,
                auth_source="jwt",
                permissions=ROLE_PERMISSIONS[effective_role],
                correlation_id=correlation_id,
            )

        raise AuthenticationError("Invalid governance token.")

    # 3. Explicit dev/test header mode (disabled when GOV_AUTH_MODE=strict)
    if governance_settings.GOV_AUTH_MODE == "dev" and x_actor_id and x_actor_id.strip():
        actor_id = x_actor_id.strip()
        db_role = _resolve_db_role(db, actor_id)
        if db_role:
            role = db_role
        elif x_actor_role and x_actor_role.strip() in ROLES:
            role = x_actor_role.strip()
        else:
            role = "viewer"
        return ActorContext(
            actor_id=actor_id,
            role=role,
            auth_source="dev_header",
            permissions=ROLE_PERMISSIONS[role],
            correlation_id=correlation_id,
        )

    raise AuthenticationError(
        "Missing governance authentication. Provide Bearer JWT or X-Governance-Token."
    )


def require_permission(permission: str) -> Callable[..., ActorContext]:
    """Dependency factory enforcing RBAC permission and auditing denied attempts."""

    def _dependency(
        request: Request,
        actor: ActorContext = Depends(get_current_actor),
        db: Session = Depends(get_db),
    ) -> ActorContext:
        if permission not in actor.permissions:
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="auth.access_denied",
                entity_type="endpoint",
                entity_id=request.url.path,
                correlation_id=actor.correlation_id,
                outcome="denied",
                reason=f"Missing required permission '{permission}'.",
                new_state={"required_permission": permission, "method": request.method},
            )
            raise PermissionDeniedError(
                f"Role '{actor.role}' lacks required permission '{permission}'.",
                details={"required_permission": permission, "role": actor.role},
            )
        return actor

    return _dependency
