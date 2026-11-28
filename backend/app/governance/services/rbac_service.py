"""RBAC role assignment management and dashboard overview summary service."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.governance.auth import ActorContext
from backend.app.governance.constants import LIFECYCLE_STAGES, MODEL_FAMILIES
from backend.app.governance.exceptions import ResourceNotFoundError
from backend.app.governance.models import (
    ApprovalRequest,
    AuditEvent,
    GovernanceRoleAssignment,
    GovernedDataset,
    GovernedModel,
)
from backend.app.governance.schemas import (
    GovernanceOverviewOut,
    RoleAssignmentOut,
    RoleAssignRequest,
)
from backend.app.governance.services.audit_service import AuditService


class RBACService:
    """Manages actor role bindings and governance dashboard summary metrics."""

    @staticmethod
    def to_out(row: GovernanceRoleAssignment) -> RoleAssignmentOut:
        return RoleAssignmentOut(
            id=str(row.id),
            actor_id=row.actor_id,
            role=row.role,
            assigned_by=row.assigned_by,
            notes=row.notes,
            is_active=row.is_active,
            created_at=row.created_at.isoformat(),
            updated_at=row.updated_at.isoformat(),
        )

    @staticmethod
    def assign_role(
        db: Session,
        actor_id: str,
        payload: RoleAssignRequest,
        actor: ActorContext,
    ) -> GovernanceRoleAssignment:
        clean_actor = actor_id.strip()
        existing = (
            db.query(GovernanceRoleAssignment)
            .filter(GovernanceRoleAssignment.actor_id == clean_actor)
            .first()
        )
        prev_state = (
            {"role": existing.role, "is_active": existing.is_active} if existing else None
        )

        if existing:
            existing.role = payload.role
            existing.assigned_by = actor.actor_id
            existing.notes = payload.notes.strip()
            existing.is_active = True
            existing.updated_at = datetime.now(timezone.utc)
            row = existing
        else:
            row = GovernanceRoleAssignment(
                actor_id=clean_actor,
                role=payload.role,
                assigned_by=actor.actor_id,
                notes=payload.notes.strip(),
                is_active=True,
            )
            db.add(row)
        db.flush()

        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="role.assigned",
            entity_type="user_role",
            entity_id=clean_actor,
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=payload.notes or f"Assigned role '{payload.role}'.",
            previous_state=prev_state,
            new_state={"role": row.role, "is_active": True},
        )
        db.commit()
        db.refresh(row)
        return row

    @staticmethod
    def revoke_role(
        db: Session,
        actor_id: str,
        actor: ActorContext,
    ) -> GovernanceRoleAssignment:
        row = (
            db.query(GovernanceRoleAssignment)
            .filter(GovernanceRoleAssignment.actor_id == actor_id.strip())
            .first()
        )
        if not row:
            raise ResourceNotFoundError(f"No role assignment found for actor '{actor_id}'.")

        prev_state = {"role": row.role, "is_active": row.is_active}
        row.is_active = False
        row.updated_at = datetime.now(timezone.utc)

        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="role.revoked",
            entity_type="user_role",
            entity_id=row.actor_id,
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=f"Revoked role '{row.role}'.",
            previous_state=prev_state,
            new_state={"role": row.role, "is_active": False},
        )
        db.commit()
        db.refresh(row)
        return row

    @staticmethod
    def list_roles(db: Session) -> list[GovernanceRoleAssignment]:
        return (
            db.query(GovernanceRoleAssignment)
            .order_by(GovernanceRoleAssignment.actor_id.asc())
            .all()
        )

    @staticmethod
    def get_overview(db: Session) -> GovernanceOverviewOut:
        total_datasets = int(db.query(func.count(GovernedDataset.id)).scalar() or 0)
        models = db.query(GovernedModel).all()
        by_stage = {stage: 0 for stage in LIFECYCLE_STAGES}
        by_compliance = {"pending": 0, "compliant": 0, "non_compliant": 0}
        critical_drift = 0
        active_prod: dict[str, dict | None] = {fam: None for fam in MODEL_FAMILIES}
        active_staging: dict[str, dict | None] = {fam: None for fam in MODEL_FAMILIES}

        for m in models:
            by_stage[m.lifecycle_status] = by_stage.get(m.lifecycle_status, 0) + 1
            by_compliance[m.compliance_status] = by_compliance.get(m.compliance_status, 0) + 1
            if m.drift_status == "critical":
                critical_drift += 1
            brief = {
                "id": str(m.id),
                "model_name": m.model_name,
                "semantic_version": m.semantic_version,
                "clearml_model_id": m.clearml_model_id,
                "compliance_status": m.compliance_status,
                "drift_status": m.drift_status,
            }
            if m.lifecycle_status == "production":
                active_prod[m.model_family] = brief
            elif m.lifecycle_status == "staging":
                active_staging[m.model_family] = brief

        pending_approvals = int(
            db.query(func.count(ApprovalRequest.id))
            .filter(ApprovalRequest.status == "pending")
            .scalar()
            or 0
        )
        audit_count = int(db.query(func.count(AuditEvent.id)).scalar() or 0)

        return GovernanceOverviewOut(
            total_datasets=total_datasets,
            total_models=len(models),
            models_by_stage=by_stage,
            models_by_compliance=by_compliance,
            pending_approval_requests=pending_approvals,
            critical_drift_models=critical_drift,
            active_production_models=active_prod,
            active_staging_models=active_staging,
            recent_audit_events_count=audit_count,
        )
