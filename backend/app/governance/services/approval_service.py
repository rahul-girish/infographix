"""Approval workflow service with automated shadow comparison & CI/CD promotion gate."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.app.governance.adapters.clearml_adapter import ClearMLRegistryProtocol
from backend.app.governance.auth import ActorContext
from backend.app.governance.exceptions import (
    ConflictError,
    InvalidTransitionError,
    PermissionDeniedError,
    PolicyViolationError,
    ResourceNotFoundError,
    ValidationError,
)
from backend.app.governance.models import ApprovalDecision, ApprovalRequest, GovernedModel
from backend.app.governance.schemas import (
    ApprovalDecisionCreate,
    ApprovalDecisionOut,
    ApprovalRequestCreate,
    ApprovalRequestOut,
    PromotionGateOut,
)
from backend.app.governance.services.audit_service import AuditService
from backend.app.governance.services.model_service import ModelService
from backend.app.governance.services.policy_service import PolicyService


def _parse_uuid(raw_id: str, label: str = "id") -> uuid.UUID:
    try:
        return uuid.UUID(str(raw_id))
    except ValueError as exc:
        raise ValidationError(f"Invalid UUID for '{label}': {raw_id}") from exc


class ApprovalService:
    """Manages approval requests, shadow comparison reports, decisions, and CI gates."""

    @staticmethod
    def to_out(req: ApprovalRequest) -> ApprovalRequestOut:
        model_name = req.model.model_name if req.model else ""
        decisions_out = [
            ApprovalDecisionOut(
                id=str(d.id),
                request_id=str(d.request_id),
                decision=d.decision,
                decided_by=d.decided_by,
                decided_role=d.decided_role,
                comment=d.comment,
                policy_override=d.policy_override,
                decided_at=d.decided_at.isoformat(),
            )
            for d in (req.decisions or [])
        ]
        return ApprovalRequestOut(
            id=str(req.id),
            model_id=str(req.model_id),
            model_name=model_name,
            target_stage=req.target_stage,
            request_type=req.request_type,
            status=req.status,
            requested_by=req.requested_by,
            reason=req.reason,
            automated_check_passed=req.automated_check_passed,
            automated_check_summary=req.automated_check_summary or {},
            baseline_model_id=str(req.baseline_model_id) if req.baseline_model_id else None,
            shadow_comparison_report=req.shadow_comparison_report or {},
            decisions=decisions_out,
            created_at=req.created_at.isoformat(),
            resolved_at=req.resolved_at.isoformat() if req.resolved_at else None,
        )

    @staticmethod
    def create_request(
        db: Session,
        model_id: str,
        payload: ApprovalRequestCreate,
        actor: ActorContext,
        clearml_adapter: ClearMLRegistryProtocol,
    ) -> ApprovalRequest:
        model = ModelService.get_model(db, model_id)

        if payload.request_type == "promotion" and model.lifecycle_status != "staging":
            raise InvalidTransitionError(
                f"Promotion approval requires model in 'staging'; '{model.model_name}' is '{model.lifecycle_status}'."
            )
        if payload.request_type == "rollback" and (
            model.lifecycle_status != "archived" or not model.was_ever_production
        ):
            raise InvalidTransitionError(
                f"Rollback approval requires an 'archived' model that previously held 'production'."
            )

        existing_pending = (
            db.query(ApprovalRequest)
            .filter(
                ApprovalRequest.model_id == model.id,
                ApprovalRequest.status == "pending",
            )
            .first()
        )
        if existing_pending:
            raise ConflictError(
                f"Model '{model.model_name}' already has a pending approval request ({existing_pending.id})."
            )

        # Invalidate any unconsumed prior approved requests before creating a new request
        stale_approved = (
            db.query(ApprovalRequest)
            .filter(
                ApprovalRequest.model_id == model.id,
                ApprovalRequest.status == "approved",
            )
            .all()
        )
        now = datetime.now(timezone.utc)
        for old_req in stale_approved:
            old_req.status = "invalidated"
            old_req.resolved_at = now
            AuditService.record(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="approval.invalidated",
                entity_type="approval_request",
                entity_id=str(old_req.id),
                correlation_id=actor.correlation_id,
                outcome="success",
                reason="Superseded by newer approval request.",
                related_model_version=model.model_name,
                previous_state={"status": "approved"},
                new_state={"status": "invalidated"},
            )

        policy_ok, checks, blocking = PolicyService.evaluate_model(
            db, model, target_stage="production", clearml_adapter=clearml_adapter
        )
        baseline, shadow_report, regression_blocked = PolicyService.compute_shadow_comparison(
            db, model, shadow_eval_payload=payload.shadow_eval_payload
        )
        if regression_blocked:
            blocking.extend(shadow_report.get("regression_reasons", []))

        automated_ok = policy_ok and not regression_blocked
        req = ApprovalRequest(
            model_id=model.id,
            target_stage=payload.target_stage,
            request_type=payload.request_type,
            status="pending",
            requested_by=actor.actor_id,
            reason=payload.reason.strip(),
            automated_check_passed=automated_ok,
            automated_check_summary={
                "passed": automated_ok,
                "blocking_reasons": blocking,
                "checks": [c.model_dump() for c in checks],
            },
            baseline_model_id=baseline.id if baseline else None,
            shadow_comparison_report=shadow_report,
        )
        db.add(req)
        db.flush()

        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="approval.requested",
            entity_type="approval_request",
            entity_id=str(req.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=payload.reason,
            related_model_version=model.model_name,
            new_state={
                "request_type": req.request_type,
                "automated_check_passed": automated_ok,
                "regression_detected": shadow_report.get("regression_detected", False),
            },
        )
        db.commit()
        db.refresh(req)
        return req

    @staticmethod
    def cancel_request(
        db: Session,
        request_id: str,
        *,
        reason: str,
        actor: ActorContext,
    ) -> ApprovalRequest:
        req = ApprovalService.get_request(db, request_id)
        model_name = req.model.model_name if req.model else ""

        if actor.actor_id != req.requested_by and actor.role != "admin":
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="auth.access_denied",
                entity_type="approval_request",
                entity_id=str(req.id),
                correlation_id=actor.correlation_id,
                outcome="denied",
                reason="Only the original requester or an admin can cancel an approval request.",
                related_model_version=model_name,
            )
            raise PermissionDeniedError(
                "Only the original requester or an admin can cancel this approval request."
            )

        if req.status != "pending":
            raise ConflictError(
                f"Approval request '{request_id}' is '{req.status}' and cannot be cancelled."
            )

        req.status = "cancelled"
        req.resolved_at = datetime.now(timezone.utc)
        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="approval.cancelled",
            entity_type="approval_request",
            entity_id=str(req.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=reason,
            related_model_version=model_name,
            previous_state={"status": "pending"},
            new_state={"status": "cancelled"},
        )
        db.commit()
        db.refresh(req)
        return req

    @staticmethod
    def list_requests(
        db: Session,
        *,
        status: Optional[str] = None,
        model_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ApprovalRequest]:
        q = db.query(ApprovalRequest)
        if status:
            q = q.filter(ApprovalRequest.status == status)
        if model_id:
            uid = _parse_uuid(model_id, "model_id")
            q = q.filter(ApprovalRequest.model_id == uid)
        return (
            q.order_by(desc(ApprovalRequest.created_at))
            .offset(max(0, offset))
            .limit(min(200, max(1, limit)))
            .all()
        )

    @staticmethod
    def get_request(db: Session, request_id: str) -> ApprovalRequest:
        uid = _parse_uuid(request_id, "request_id")
        req = db.query(ApprovalRequest).filter(ApprovalRequest.id == uid).first()
        if not req:
            raise ResourceNotFoundError(f"Approval request '{request_id}' not found.")
        return req

    @staticmethod
    def decide_request(
        db: Session,
        request_id: str,
        payload: ApprovalDecisionCreate,
        actor: ActorContext,
    ) -> ApprovalRequest:
        req = ApprovalService.get_request(db, request_id)
        model_name = req.model.model_name if req.model else ""

        if req.status != "pending":
            raise ConflictError(
                f"Approval request '{request_id}' is '{req.status}' and can no longer be decided."
            )

        _, _, prod_cfg = PolicyService.get_policy_config(db, "production_approval_policy")
        allow_self = bool(prod_cfg.get("allow_self_approval", False))
        if not allow_self and actor.actor_id == req.requested_by:
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="auth.access_denied",
                entity_type="approval_request",
                entity_id=str(req.id),
                correlation_id=actor.correlation_id,
                outcome="denied",
                reason="Separation-of-duties violation: requester cannot approve/reject own request.",
                related_model_version=model_name,
            )
            raise PermissionDeniedError(
                "Separation of duties: requester cannot decide their own approval request."
            )

        if payload.decision == "approved" and not req.automated_check_passed:
            raise PolicyViolationError(
                "Cannot approve request because automated policy or shadow comparison checks failed.",
                details=req.automated_check_summary or {},
            )

        decision_row = ApprovalDecision(
            request_id=req.id,
            decision=payload.decision,
            decided_by=actor.actor_id,
            decided_role=actor.role,
            comment=payload.comment.strip(),
            policy_override=False,
        )
        db.add(decision_row)

        prev_status = req.status
        req.status = payload.decision
        req.resolved_at = datetime.now(timezone.utc)

        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action=f"approval.{payload.decision}",
            entity_type="approval_request",
            entity_id=str(req.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=payload.comment,
            related_model_version=model_name,
            previous_state={"status": prev_status},
            new_state={
                "status": req.status,
                "policy_override": decision_row.policy_override,
            },
        )
        db.commit()
        db.refresh(req)
        return req

    @staticmethod
    def cancel_request(
        db: Session,
        request_id: str,
        *,
        reason: str,
        actor: ActorContext,
    ) -> ApprovalRequest:
        req = ApprovalService.get_request(db, request_id)
        model_name = req.model.model_name if req.model else ""

        if actor.actor_id != req.requested_by and actor.role != "admin":
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="auth.access_denied",
                entity_type="approval_request",
                entity_id=str(req.id),
                correlation_id=actor.correlation_id,
                outcome="denied",
                reason="Only the original requester or an admin can cancel an approval request.",
                related_model_version=model_name,
            )
            raise PermissionDeniedError(
                "Only the original requester or an admin can cancel this approval request."
            )

        if req.status != "pending":
            raise ConflictError(
                f"Approval request '{request_id}' is '{req.status}' and cannot be cancelled."
            )

        req.status = "cancelled"
        req.resolved_at = datetime.now(timezone.utc)
        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="approval.cancelled",
            entity_type="approval_request",
            entity_id=str(req.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=reason,
            related_model_version=model_name,
            previous_state={"status": "pending"},
            new_state={"status": "cancelled"},
        )
        db.commit()
        db.refresh(req)
        return req

    @staticmethod
    def get_promotion_gate(
        db: Session,
        model_id: str,
        clearml_adapter: ClearMLRegistryProtocol,
    ) -> PromotionGateOut:
        """Read-only CI/CD pre-flight check before promotion & Render deployment."""
        model = ModelService.get_model(db, model_id)
        passed, checks, blocking = PolicyService.evaluate_model(
            db, model, target_stage="production", clearml_adapter=clearml_adapter
        )
        reasons = list(blocking)

        if model.lifecycle_status != "staging":
            reasons.append(
                f"Model must be in 'staging' before production promotion (current: '{model.lifecycle_status}')."
            )

        enabled, _, prod_cfg = PolicyService.get_policy_config(db, "production_approval_policy")
        approved_req_id: Optional[str] = None
        if enabled and prod_cfg.get("require_human_approval", True):
            approved_req = (
                db.query(ApprovalRequest)
                .filter(
                    ApprovalRequest.model_id == model.id,
                    ApprovalRequest.target_stage == "production",
                    ApprovalRequest.status == "approved",
                )
                .order_by(desc(ApprovalRequest.resolved_at))
                .first()
            )
            if approved_req:
                approved_req_id = str(approved_req.id)
            else:
                reasons.append("No approved ApprovalRequest exists for production promotion.")

        return PromotionGateOut(
            model_id=str(model.id),
            model_name=model.model_name,
            current_stage=model.lifecycle_status,
            target_stage="production",
            eligible=passed and len(reasons) == 0,
            approved_request_id=approved_req_id,
            checks=checks,
            blocking_reasons=reasons,
        )
