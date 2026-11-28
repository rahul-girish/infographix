"""FastAPI routes for the Infographix MLOps Governance module (/api/v1/governance)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.governance.adapters.clearml_adapter import get_clearml_adapter
from backend.app.governance.auth import ActorContext, require_permission
from backend.app.governance.constants import LIFECYCLE_STAGES
from backend.app.governance.schemas import (
    ActorMeOut,
    ApprovalDecisionCreate,
    ApprovalRequestCreate,
    ApprovalRequestOut,
    AuditEventOut,
    DatasetCreateRequest,
    DatasetIngestManifestRequest,
    DatasetOut,
    DriftEventCreateRequest,
    DriftEventOut,
    GovernanceOverviewOut,
    GovernancePolicyOut,
    GovernedModelOut,
    LifecycleActionRequest,
    ModelMetadataUpdateRequest,
    ModelRegisterRequest,
    PolicyEvaluationOut,
    PolicyUpdateRequest,
    PromotionGateOut,
    RoleAssignmentOut,
    RoleAssignRequest,
)
from backend.app.governance.services.approval_service import ApprovalService
from backend.app.governance.services.audit_service import AuditService
from backend.app.governance.services.dataset_service import DatasetService
from backend.app.governance.services.drift_service import DriftService
from backend.app.governance.services.model_service import ModelService
from backend.app.governance.services.policy_service import PolicyService
from backend.app.governance.services.rbac_service import RBACService

router = APIRouter(prefix="/governance", tags=["MLOps Governance"])


# ---------------------------------------------------------------------------
# Identity & Overview
# ---------------------------------------------------------------------------

@router.get("/me", response_model=ActorMeOut)
def get_me(actor: ActorContext = Depends(require_permission("governance:read"))):
    """Return authenticated actor identity, effective role, and permissions."""
    return ActorMeOut(
        actor_id=actor.actor_id,
        role=actor.role,  # type: ignore[arg-type]
        auth_source=actor.auth_source,
        permissions=sorted(actor.permissions),
    )


@router.get("/overview", response_model=GovernanceOverviewOut)
def get_overview(
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    """Dashboard summary across datasets, models, approvals, drift, and audits."""
    return RBACService.get_overview(db)


# ---------------------------------------------------------------------------
# Dataset Governance Endpoints
# ---------------------------------------------------------------------------

@router.post("/datasets", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
def register_dataset(
    payload: DatasetCreateRequest,
    actor: ActorContext = Depends(require_permission("dataset:write")),
    db: Session = Depends(get_db),
):
    ds = DatasetService.register_dataset(db, payload, actor)
    return DatasetService.to_out(db, ds)


@router.post(
    "/datasets/ingest-manifest",
    response_model=DatasetOut,
    status_code=status.HTTP_201_CREATED,
)
def ingest_dataset_manifest(
    payload: DatasetIngestManifestRequest,
    actor: ActorContext = Depends(require_permission("dataset:write")),
    db: Session = Depends(get_db),
):
    ds = DatasetService.ingest_from_manifest(db, payload, actor)
    return DatasetService.to_out(db, ds)


@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(
    active_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    rows = DatasetService.list_datasets(db, active_only=active_only, limit=limit, offset=offset)
    return [DatasetService.to_out(db, r) for r in rows]


@router.get("/datasets/{dataset_id}", response_model=DatasetOut)
def get_dataset(
    dataset_id: str,
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    ds = DatasetService.get_dataset(db, dataset_id)
    return DatasetService.to_out(db, ds)


# ---------------------------------------------------------------------------
# Model Governance & Lifecycle Endpoints
# ---------------------------------------------------------------------------

@router.post("/models", response_model=GovernedModelOut, status_code=status.HTTP_201_CREATED)
def register_model(
    payload: ModelRegisterRequest,
    actor: ActorContext = Depends(require_permission("model:register")),
    db: Session = Depends(get_db),
):
    model = ModelService.register_model(db, payload, actor, get_clearml_adapter())
    return ModelService.to_out(model)


@router.get("/models", response_model=list[GovernedModelOut])
def list_models(
    model_family: Optional[str] = Query(default=None),
    lifecycle_status: Optional[str] = Query(default=None),
    compliance_status: Optional[str] = Query(default=None),
    drift_status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    rows = ModelService.list_models(
        db,
        model_family=model_family,
        lifecycle_status=lifecycle_status,
        compliance_status=compliance_status,
        drift_status=drift_status,
        limit=limit,
        offset=offset,
    )
    return [ModelService.to_out(m) for m in rows]


@router.get("/models/{model_id}", response_model=GovernedModelOut)
def get_model(
    model_id: str,
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    return ModelService.to_out(ModelService.get_model(db, model_id))


@router.patch("/models/{model_id}", response_model=GovernedModelOut)
def update_model_metadata(
    model_id: str,
    payload: ModelMetadataUpdateRequest,
    actor: ActorContext = Depends(require_permission("model:update_metadata")),
    db: Session = Depends(get_db),
):
    model = ModelService.update_metadata(db, model_id, payload, actor)
    return ModelService.to_out(model)


@router.post("/models/{model_id}/evaluate", response_model=PolicyEvaluationOut)
def evaluate_model(
    model_id: str,
    target_stage: str = Query(default="staging", pattern=f"^({'|'.join(LIFECYCLE_STAGES)})$"),
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    return ModelService.evaluate_and_persist(
        db, model_id, target_stage=target_stage, actor=actor, clearml_adapter=get_clearml_adapter()
    )


@router.post("/models/{model_id}/promote-staging", response_model=GovernedModelOut)
def promote_to_staging(
    model_id: str,
    payload: LifecycleActionRequest,
    actor: ActorContext = Depends(require_permission("model:promote_staging")),
    db: Session = Depends(get_db),
):
    model = ModelService.promote_to_staging(db, model_id, payload, actor, get_clearml_adapter())
    return ModelService.to_out(model)


@router.get("/models/{model_id}/promotion-gate", response_model=PromotionGateOut)
def get_promotion_gate(
    model_id: str,
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    return ApprovalService.get_promotion_gate(db, model_id, get_clearml_adapter())


@router.post("/models/{model_id}/promote-production", response_model=GovernedModelOut)
def promote_to_production(
    model_id: str,
    payload: LifecycleActionRequest,
    actor: ActorContext = Depends(require_permission("model:promote_production")),
    db: Session = Depends(get_db),
):
    model = ModelService.promote_to_production(db, model_id, payload, actor, get_clearml_adapter())
    return ModelService.to_out(model)


@router.post("/models/{model_id}/demote-staging", response_model=GovernedModelOut)
def demote_to_staging(
    model_id: str,
    payload: LifecycleActionRequest,
    actor: ActorContext = Depends(require_permission("model:demote_production")),
    db: Session = Depends(get_db),
):
    model = ModelService.demote_to_staging(db, model_id, payload, actor, get_clearml_adapter())
    return ModelService.to_out(model)


@router.post("/models/{model_id}/rollback", response_model=GovernedModelOut)
def rollback_model(
    model_id: str,
    payload: LifecycleActionRequest,
    actor: ActorContext = Depends(require_permission("model:rollback")),
    db: Session = Depends(get_db),
):
    model = ModelService.rollback_to_production(db, model_id, payload, actor, get_clearml_adapter())
    return ModelService.to_out(model)


@router.post("/models/{model_id}/archive", response_model=GovernedModelOut)
def archive_model(
    model_id: str,
    payload: LifecycleActionRequest,
    actor: ActorContext = Depends(require_permission("model:archive_non_prod")),
    db: Session = Depends(get_db),
):
    model = ModelService.archive_model(db, model_id, payload, actor, get_clearml_adapter())
    return ModelService.to_out(model)


# ---------------------------------------------------------------------------
# Approval & Shadow Comparison Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/models/{model_id}/approval-requests",
    response_model=ApprovalRequestOut,
    status_code=status.HTTP_201_CREATED,
)
def create_approval_request(
    model_id: str,
    payload: ApprovalRequestCreate,
    actor: ActorContext = Depends(require_permission("approval:request")),
    db: Session = Depends(get_db),
):
    req = ApprovalService.create_request(db, model_id, payload, actor, get_clearml_adapter())
    return ApprovalService.to_out(req)


@router.get("/approval-requests", response_model=list[ApprovalRequestOut])
def list_approval_requests(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    model_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    rows = ApprovalService.list_requests(
        db, status=status_filter, model_id=model_id, limit=limit, offset=offset
    )
    return [ApprovalService.to_out(r) for r in rows]


@router.get("/approval-requests/{request_id}", response_model=ApprovalRequestOut)
def get_approval_request(
    request_id: str,
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    return ApprovalService.to_out(ApprovalService.get_request(db, request_id))


@router.post("/approval-requests/{request_id}/decide", response_model=ApprovalRequestOut)
def decide_approval_request(
    request_id: str,
    payload: ApprovalDecisionCreate,
    actor: ActorContext = Depends(require_permission("approval:decide")),
    db: Session = Depends(get_db),
):
    req = ApprovalService.decide_request(db, request_id, payload, actor)
    return ApprovalService.to_out(req)


@router.post("/approval-requests/{request_id}/cancel", response_model=ApprovalRequestOut)
def cancel_approval_request(
    request_id: str,
    payload: LifecycleActionRequest,
    actor: ActorContext = Depends(require_permission("approval:request")),
    db: Session = Depends(get_db),
):
    req = ApprovalService.cancel_request(db, request_id, reason=payload.reason, actor=actor)
    return ApprovalService.to_out(req)


# ---------------------------------------------------------------------------
# Drift Integration Endpoints
# ---------------------------------------------------------------------------

@router.post("/drift-events", response_model=DriftEventOut, status_code=status.HTTP_201_CREATED)
def record_drift_event(
    payload: DriftEventCreateRequest,
    actor: ActorContext = Depends(require_permission("drift:record")),
    db: Session = Depends(get_db),
):
    ev = DriftService.record_drift_event(db, payload, actor)
    return DriftService.to_out(ev)


@router.get("/drift-events", response_model=list[DriftEventOut])
def list_drift_events(
    model_id: Optional[str] = Query(default=None),
    drift_status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    rows = DriftService.list_drift_events(
        db, model_id=model_id, drift_status=drift_status, limit=limit, offset=offset
    )
    return [DriftService.to_out(r) for r in rows]


# ---------------------------------------------------------------------------
# Policy Configuration Endpoints
# ---------------------------------------------------------------------------

@router.get("/policies", response_model=list[GovernancePolicyOut])
def list_policies(
    model_family: Optional[str] = Query(default=None),
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    return [PolicyService.to_out(p) for p in PolicyService.list_policies(db, model_family=model_family)]


@router.get("/policies/{policy_key}", response_model=GovernancePolicyOut)
def get_policy(
    policy_key: str,
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    return PolicyService.to_out(PolicyService.get_policy(db, policy_key))


@router.patch("/policies/{policy_key}", response_model=GovernancePolicyOut)
def update_policy(
    policy_key: str,
    payload: PolicyUpdateRequest,
    actor: ActorContext = Depends(require_permission("policy:write")),
    db: Session = Depends(get_db),
):
    return PolicyService.to_out(PolicyService.update_policy(db, policy_key, payload, actor))


# ---------------------------------------------------------------------------
# RBAC Role Management Endpoints
# ---------------------------------------------------------------------------

@router.get("/roles", response_model=list[RoleAssignmentOut])
def list_roles(
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    return [RBACService.to_out(r) for r in RBACService.list_roles(db)]


@router.put("/roles/{actor_id}", response_model=RoleAssignmentOut)
def assign_role(
    actor_id: str,
    payload: RoleAssignRequest,
    actor: ActorContext = Depends(require_permission("rbac:manage")),
    db: Session = Depends(get_db),
):
    return RBACService.to_out(RBACService.assign_role(db, actor_id, payload, actor))


@router.delete("/roles/{actor_id}", response_model=RoleAssignmentOut)
def revoke_role(
    actor_id: str,
    actor: ActorContext = Depends(require_permission("rbac:manage")),
    db: Session = Depends(get_db),
):
    return RBACService.to_out(RBACService.revoke_role(db, actor_id, actor))


# ---------------------------------------------------------------------------
# Read-Only Audit Trail Endpoint (Append-Only; No PUT/PATCH/DELETE)
# ---------------------------------------------------------------------------

@router.get("/audit-events", response_model=list[AuditEventOut])
def list_audit_events(
    action: Optional[str] = Query(default=None),
    entity_type: Optional[str] = Query(default=None),
    entity_id: Optional[str] = Query(default=None),
    actor_id: Optional[str] = Query(default=None),
    outcome: Optional[str] = Query(default=None),
    correlation_id: Optional[str] = Query(default=None),
    related_model_version: Optional[str] = Query(default=None),
    since: Optional[datetime] = Query(default=None),
    until: Optional[datetime] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    actor: ActorContext = Depends(require_permission("governance:read")),
    db: Session = Depends(get_db),
):
    rows = AuditService.query(
        db,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        actor_id=actor_id,
        outcome=outcome,
        correlation_id=correlation_id,
        related_model_version=related_model_version,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )
    return [
        AuditEventOut(
            id=str(e.id),
            timestamp=e.timestamp.isoformat(),
            actor_id=e.actor_id,
            actor_role=e.actor_role,
            action=e.action,
            entity_type=e.entity_type,
            entity_id=e.entity_id,
            related_model_version=e.related_model_version,
            related_dataset_version=e.related_dataset_version,
            previous_state=e.previous_state,
            new_state=e.new_state,
            reason=e.reason,
            outcome=e.outcome,
            correlation_id=e.correlation_id,
        )
        for e in rows
    ]
