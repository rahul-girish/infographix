"""Model governance registration, metadata management, and lifecycle transition service."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.app.governance.adapters.clearml_adapter import ClearMLRegistryProtocol
from backend.app.governance.auth import ActorContext
from backend.app.governance.constants import (
    ALLOWED_STAGE_TRANSITIONS,
    FAMILY_ARCHITECTURES,
    LIFECYCLE_STAGES,
)
from backend.app.governance.exceptions import (
    ConflictError,
    InvalidTransitionError,
    PermissionDeniedError,
    PolicyViolationError,
    ResourceNotFoundError,
    ValidationError,
)
from backend.app.governance.models import ApprovalRequest, GovernedDataset, GovernedModel
from backend.app.governance.schemas import (
    GovernedModelOut,
    LifecycleActionRequest,
    ModelMetadataUpdateRequest,
    ModelRegisterRequest,
    PolicyEvaluationOut,
)
from backend.app.governance.services.audit_service import AuditService
from backend.app.governance.services.policy_service import PolicyService


def _parse_uuid(raw_id: str, label: str = "id") -> uuid.UUID:
    try:
        return uuid.UUID(str(raw_id))
    except ValueError as exc:
        raise ValidationError(f"Invalid UUID for '{label}': {raw_id}") from exc


def _invalidate_open_approvals(
    db: Session,
    *,
    model: GovernedModel,
    actor: ActorContext,
    reason: str,
    include_approved: bool = False,
) -> None:
    statuses = ["pending", "approved"] if include_approved else ["pending"]
    open_reqs = (
        db.query(ApprovalRequest)
        .filter(
            ApprovalRequest.model_id == model.id,
            ApprovalRequest.status.in_(statuses),
        )
        .all()
    )
    now = datetime.now(timezone.utc)
    for req in open_reqs:
        prev_status = req.status
        req.status = "invalidated"
        req.resolved_at = now
        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="approval.invalidated",
            entity_type="approval_request",
            entity_id=str(req.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=reason,
            related_model_version=model.model_name,
            previous_state={"status": prev_status},
            new_state={"status": "invalidated"},
        )


def _commit_with_clearml_sync(
    db: Session,
    clearml_adapter: ClearMLRegistryProtocol,
    sync_ops: list[tuple[str, str, str, str]],
) -> None:
    """Flush DB changes, sync ClearML stages, and compensate ClearML if any step fails."""
    db.flush()
    completed: list[tuple[str, str, str]] = []
    try:
        for clearml_model_id, prev_stage, target_stage, model_family in sync_ops:
            clearml_adapter.sync_lifecycle_status(
                clearml_model_id=clearml_model_id,
                target_status=target_stage,
                model_family=model_family,
            )
            completed.append((clearml_model_id, prev_stage, model_family))
        db.commit()
    except Exception:
        for cid, prev_stage, fam in reversed(completed):
            try:
                clearml_adapter.sync_lifecycle_status(
                    clearml_model_id=cid,
                    target_status=prev_stage,
                    model_family=fam,
                )
            except Exception:
                pass
        db.rollback()
        raise


class ModelService:
    """Enforces model registration rules, immutable provenance, and lifecycle state transitions."""

    @staticmethod
    def to_out(model: GovernedModel) -> GovernedModelOut:
        return GovernedModelOut(
            id=str(model.id),
            clearml_model_id=model.clearml_model_id,
            model_name=model.model_name,
            model_family=model.model_family,
            semantic_version=model.semantic_version,
            architecture=model.architecture,
            parent_model_id=str(model.parent_model_id) if model.parent_model_id else None,
            clearml_training_task_id=model.clearml_training_task_id,
            artifact_uri=model.artifact_uri,
            checkpoint_sha256=model.checkpoint_sha256,
            git_commit=model.git_commit,
            dataset_id=str(model.dataset_id) if model.dataset_id else None,
            legacy_dataset_identifier=model.legacy_dataset_identifier,
            retriever_model_id=model.retriever_model_id,
            retriever_version=model.retriever_version,
            schema_version=model.schema_version,
            environment_summary=model.environment_summary or {},
            dependencies_summary=model.dependencies_summary or {},
            metrics_summary=model.metrics_summary or {},
            lifecycle_status=model.lifecycle_status,
            was_ever_production=model.was_ever_production,
            compliance_status=model.compliance_status,
            drift_status=model.drift_status,
            owner=model.owner,
            description=model.description,
            tags=list(model.tags or []),
            registered_at=model.registered_at.isoformat(),
            updated_at=model.updated_at.isoformat(),
        )

    @staticmethod
    def get_model(db: Session, model_id: str) -> GovernedModel:
        uid = _parse_uuid(model_id, "model_id")
        model = db.query(GovernedModel).filter(GovernedModel.id == uid).first()
        if not model:
            raise ResourceNotFoundError(f"Governed model '{model_id}' not found.")
        return model

    @staticmethod
    def list_models(
        db: Session,
        *,
        model_family: Optional[str] = None,
        lifecycle_status: Optional[str] = None,
        compliance_status: Optional[str] = None,
        drift_status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[GovernedModel]:
        q = db.query(GovernedModel)
        if model_family:
            q = q.filter(GovernedModel.model_family == model_family)
        if lifecycle_status:
            q = q.filter(GovernedModel.lifecycle_status == lifecycle_status)
        if compliance_status:
            q = q.filter(GovernedModel.compliance_status == compliance_status)
        if drift_status:
            q = q.filter(GovernedModel.drift_status == drift_status)
        return (
            q.order_by(desc(GovernedModel.registered_at))
            .offset(max(0, offset))
            .limit(min(200, max(1, limit)))
            .all()
        )

    @staticmethod
    def register_model(
        db: Session,
        payload: ModelRegisterRequest,
        actor: ActorContext,
        clearml_adapter: ClearMLRegistryProtocol,
    ) -> GovernedModel:
        PolicyService.ensure_default_policies(db)
        model_name = f"{payload.model_family}-v{payload.semantic_version}"

        dup_clearml = (
            db.query(GovernedModel)
            .filter(GovernedModel.clearml_model_id == payload.clearml_model_id)
            .first()
        )
        if dup_clearml:
            raise ConflictError(
                f"ClearML model ID '{payload.clearml_model_id}' is already registered in governance."
            )

        dup_ver = (
            db.query(GovernedModel)
            .filter(
                GovernedModel.model_family == payload.model_family,
                GovernedModel.semantic_version == payload.semantic_version,
            )
            .first()
        )
        if dup_ver:
            raise ConflictError(
                f"Model '{model_name}' already exists; registered semantic versions are immutable."
            )

        dataset_uuid: Optional[uuid.UUID] = None
        if payload.dataset_id:
            dataset_uuid = _parse_uuid(payload.dataset_id, "dataset_id")
            ds = db.query(GovernedDataset).filter(GovernedDataset.id == dataset_uuid).first()
            if not ds:
                raise ResourceNotFoundError(f"Dataset '{payload.dataset_id}' not found.")

        parent_uuid: Optional[uuid.UUID] = None
        if payload.parent_model_id:
            parent_uuid = _parse_uuid(payload.parent_model_id, "parent_model_id")
            if not db.query(GovernedModel).filter(GovernedModel.id == parent_uuid).first():
                raise ResourceNotFoundError(f"Parent model '{payload.parent_model_id}' not found.")

        architecture = payload.architecture or FAMILY_ARCHITECTURES[payload.model_family]
        model = GovernedModel(
            clearml_model_id=payload.clearml_model_id.strip(),
            model_name=model_name,
            model_family=payload.model_family,
            semantic_version=payload.semantic_version,
            architecture=architecture,
            parent_model_id=parent_uuid,
            clearml_training_task_id=payload.clearml_training_task_id.strip(),
            artifact_uri=payload.artifact_uri.strip(),
            checkpoint_sha256=payload.checkpoint_sha256,
            git_commit=payload.git_commit.strip(),
            dataset_id=dataset_uuid,
            legacy_dataset_identifier=payload.legacy_dataset_identifier,
            retriever_model_id=payload.retriever_model_id,
            retriever_version=payload.retriever_version,
            schema_version=payload.schema_version,
            environment_summary=payload.environment_summary,
            dependencies_summary=payload.dependencies_summary,
            metrics_summary=payload.metrics_summary,
            lifecycle_status="candidate",
            was_ever_production=False,
            compliance_status="pending",
            drift_status="unknown",
            owner=(payload.owner or actor.actor_id).strip(),
            description=payload.description.strip(),
            tags=list(payload.tags),
        )
        db.add(model)
        db.flush()

        passed, _, blocking = PolicyService.evaluate_model(
            db,
            model,
            target_stage="candidate",
            clearml_adapter=clearml_adapter,
        )
        if not passed:
            db.rollback()
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="model.registered",
                entity_type="model",
                entity_id=model_name,
                correlation_id=actor.correlation_id,
                outcome="failed",
                reason="; ".join(blocking),
                related_model_version=model_name,
            )
            raise PolicyViolationError(
                "Model registration blocked by governance policy.",
                details={"blocking_reasons": blocking},
            )

        model.compliance_status = "compliant"
        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="model.registered",
            entity_type="model",
            entity_id=str(model.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            related_model_version=model.model_name,
            new_state={
                "lifecycle_status": "candidate",
                "clearml_model_id": model.clearml_model_id,
                "checkpoint_sha256": model.checkpoint_sha256,
                "compliance_status": model.compliance_status,
            },
        )
        db.commit()
        db.refresh(model)
        return model

    @staticmethod
    def update_metadata(
        db: Session,
        model_id: str,
        payload: ModelMetadataUpdateRequest,
        actor: ActorContext,
    ) -> GovernedModel:
        model = ModelService.get_model(db, model_id)
        prev = {
            "owner": model.owner,
            "description": model.description,
            "tags": list(model.tags or []),
        }
        if payload.owner is not None:
            model.owner = payload.owner.strip()
        if payload.description is not None:
            model.description = payload.description.strip()
        if payload.tags is not None:
            model.tags = list(payload.tags)

        model.updated_at = datetime.now(timezone.utc)
        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="model.metadata_updated",
            entity_type="model",
            entity_id=str(model.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=payload.reason,
            related_model_version=model.model_name,
            previous_state=prev,
            new_state={
                "owner": model.owner,
                "description": model.description,
                "tags": list(model.tags or []),
            },
        )
        db.commit()
        db.refresh(model)
        return model

    @staticmethod
    def evaluate_and_persist(
        db: Session,
        model_id: str,
        *,
        target_stage: str,
        actor: ActorContext,
        clearml_adapter: ClearMLRegistryProtocol,
    ) -> PolicyEvaluationOut:
        if target_stage not in LIFECYCLE_STAGES:
            raise ValidationError(f"Invalid target_stage '{target_stage}'.")
        model = ModelService.get_model(db, model_id)
        passed, checks, _ = PolicyService.evaluate_model(
            db, model, target_stage=target_stage, clearml_adapter=clearml_adapter
        )
        prev_status = model.compliance_status
        model.compliance_status = "compliant" if passed else "non_compliant"
        model.updated_at = datetime.now(timezone.utc)
        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="model.evaluated",
            entity_type="model",
            entity_id=str(model.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            related_model_version=model.model_name,
            previous_state={"compliance_status": prev_status},
            new_state={"compliance_status": model.compliance_status, "target_stage": target_stage},
        )
        db.commit()
        db.refresh(model)
        return PolicyEvaluationOut(
            model_id=str(model.id),
            model_name=model.model_name,
            target_stage=target_stage,
            passed=passed,
            compliance_status=model.compliance_status,
            checks=checks,
        )

    @staticmethod
    def _validate_transition(
        db: Session,
        *,
        model: GovernedModel,
        target_stage: str,
        actor: ActorContext,
        reason: str,
    ) -> None:
        current = model.lifecycle_status
        if current == target_stage:
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="model.transition_denied",
                entity_type="model",
                entity_id=str(model.id),
                correlation_id=actor.correlation_id,
                outcome="denied",
                reason=f"Model is already in '{target_stage}'.",
                related_model_version=model.model_name,
            )
            raise InvalidTransitionError(f"Model '{model.model_name}' is already in stage '{target_stage}'.")

        allowed = ALLOWED_STAGE_TRANSITIONS.get(current, set())
        if target_stage not in allowed:
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="model.transition_denied",
                entity_type="model",
                entity_id=str(model.id),
                correlation_id=actor.correlation_id,
                outcome="denied",
                reason=f"Transition {current} -> {target_stage} is not allowed. ({reason})",
                related_model_version=model.model_name,
                previous_state={"lifecycle_status": current},
                new_state={"lifecycle_status": target_stage},
            )
            raise InvalidTransitionError(
                f"Invalid lifecycle transition from '{current}' to '{target_stage}'."
            )

    @staticmethod
    def _archive_existing_in_stage(
        db: Session,
        *,
        model_family: str,
        stage: str,
        exclude_model_id: uuid.UUID,
        actor: ActorContext,
        reason: str,
    ) -> list[tuple[str, str, str, str]]:
        existing = (
            db.query(GovernedModel)
            .filter(
                GovernedModel.model_family == model_family,
                GovernedModel.lifecycle_status == stage,
                GovernedModel.id != exclude_model_id,
            )
            .all()
        )
        now = datetime.now(timezone.utc)
        sync_ops: list[tuple[str, str, str, str]] = []
        for old_model in existing:
            prev_stage = old_model.lifecycle_status
            old_model.lifecycle_status = "archived"
            old_model.updated_at = now
            sync_ops.append(
                (old_model.clearml_model_id, prev_stage, "archived", old_model.model_family)
            )
            _invalidate_open_approvals(
                db,
                model=old_model,
                actor=actor,
                reason=f"Model auto-archived on promotion of newer version ({reason}).",
                include_approved=True,
            )
            AuditService.record(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="model.archived",
                entity_type="model",
                entity_id=str(old_model.id),
                correlation_id=actor.correlation_id,
                outcome="success",
                reason=reason,
                related_model_version=old_model.model_name,
                previous_state={"lifecycle_status": prev_stage},
                new_state={"lifecycle_status": "archived"},
            )
        return sync_ops

    @staticmethod
    def promote_to_staging(
        db: Session,
        model_id: str,
        payload: LifecycleActionRequest,
        actor: ActorContext,
        clearml_adapter: ClearMLRegistryProtocol,
    ) -> GovernedModel:
        model = ModelService.get_model(db, model_id)
        ModelService._validate_transition(
            db, model=model, target_stage="staging", actor=actor, reason=payload.reason
        )

        passed, _, blocking = PolicyService.evaluate_model(
            db, model, target_stage="staging", clearml_adapter=clearml_adapter
        )
        if not passed:
            model.compliance_status = "non_compliant"
            db.commit()
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="model.transition_denied",
                entity_type="model",
                entity_id=str(model.id),
                correlation_id=actor.correlation_id,
                outcome="denied",
                reason="; ".join(blocking),
                related_model_version=model.model_name,
            )
            raise PolicyViolationError(
                "Staging promotion blocked by governance policies.",
                details={"blocking_reasons": blocking},
            )

        sync_ops: list[tuple[str, str, str, str]] = []
        _, _, prod_cfg = PolicyService.get_policy_config(db, "production_approval_policy")
        if prod_cfg.get("single_active_staging", True):
            sync_ops.extend(
                ModelService._archive_existing_in_stage(
                    db,
                    model_family=model.model_family,
                    stage="staging",
                    exclude_model_id=model.id,
                    actor=actor,
                    reason=f"Superseded in staging by {model.model_name}",
                )
            )

        prev_stage = model.lifecycle_status
        model.lifecycle_status = "staging"
        model.compliance_status = "compliant"
        model.updated_at = datetime.now(timezone.utc)
        sync_ops.append((model.clearml_model_id, prev_stage, "staging", model.model_family))

        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="model.lifecycle_transition",
            entity_type="model",
            entity_id=str(model.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=payload.reason,
            related_model_version=model.model_name,
            previous_state={"lifecycle_status": prev_stage},
            new_state={"lifecycle_status": "staging"},
        )
        _commit_with_clearml_sync(db, clearml_adapter, sync_ops)
        db.refresh(model)
        return model

    @staticmethod
    def promote_to_production(
        db: Session,
        model_id: str,
        payload: LifecycleActionRequest,
        actor: ActorContext,
        clearml_adapter: ClearMLRegistryProtocol,
    ) -> GovernedModel:
        model = ModelService.get_model(db, model_id)
        ModelService._validate_transition(
            db, model=model, target_stage="production", actor=actor, reason=payload.reason
        )

        passed, _, blocking = PolicyService.evaluate_model(
            db, model, target_stage="production", clearml_adapter=clearml_adapter
        )
        if not passed:
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="model.transition_denied",
                entity_type="model",
                entity_id=str(model.id),
                correlation_id=actor.correlation_id,
                outcome="denied",
                reason="; ".join(blocking),
                related_model_version=model.model_name,
            )
            raise PolicyViolationError(
                "Production promotion blocked by governance policies.",
                details={"blocking_reasons": blocking},
            )

        enabled, _, prod_cfg = PolicyService.get_policy_config(db, "production_approval_policy")
        require_human = enabled and bool(prod_cfg.get("require_human_approval", True))

        approved_req: Optional[ApprovalRequest] = None
        if require_human:
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
            if not approved_req:
                AuditService.record_isolated(
                    db,
                    actor_id=actor.actor_id,
                    actor_role=actor.role,
                    action="model.transition_denied",
                    entity_type="model",
                    entity_id=str(model.id),
                    correlation_id=actor.correlation_id,
                    outcome="denied",
                    reason="Missing approved ApprovalRequest for production promotion.",
                    related_model_version=model.model_name,
                )
                raise PolicyViolationError(
                    "Production promotion requires an approved ApprovalRequest first."
                )

        sync_ops: list[tuple[str, str, str, str]] = []
        if prod_cfg.get("single_active_production", True):
            sync_ops.extend(
                ModelService._archive_existing_in_stage(
                    db,
                    model_family=model.model_family,
                    stage="production",
                    exclude_model_id=model.id,
                    actor=actor,
                    reason=f"Superseded in production by {model.model_name}",
                )
            )

        prev_stage = model.lifecycle_status
        model.lifecycle_status = "production"
        model.was_ever_production = True
        model.compliance_status = "compliant"
        model.updated_at = datetime.now(timezone.utc)
        if approved_req is not None:
            approved_req.status = "executed"
        sync_ops.append((model.clearml_model_id, prev_stage, "production", model.model_family))

        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="model.promoted_to_production",
            entity_type="model",
            entity_id=str(model.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=payload.reason,
            related_model_version=model.model_name,
            previous_state={"lifecycle_status": prev_stage},
            new_state={"lifecycle_status": "production", "was_ever_production": True},
        )
        _commit_with_clearml_sync(db, clearml_adapter, sync_ops)
        db.refresh(model)
        return model

    @staticmethod
    def demote_to_staging(
        db: Session,
        model_id: str,
        payload: LifecycleActionRequest,
        actor: ActorContext,
        clearml_adapter: ClearMLRegistryProtocol,
    ) -> GovernedModel:
        if "model:demote_production" not in actor.permissions:
            raise PermissionDeniedError("Demoting a production model requires 'admin' role.")
        model = ModelService.get_model(db, model_id)
        ModelService._validate_transition(
            db, model=model, target_stage="staging", actor=actor, reason=payload.reason
        )

        prev_stage = model.lifecycle_status
        model.lifecycle_status = "staging"
        model.updated_at = datetime.now(timezone.utc)
        _invalidate_open_approvals(
            db,
            model=model,
            actor=actor,
            reason=f"Model demoted from production to staging: {payload.reason}",
            include_approved=True,
        )
        sync_ops = [(model.clearml_model_id, prev_stage, "staging", model.model_family)]

        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="model.lifecycle_transition",
            entity_type="model",
            entity_id=str(model.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=payload.reason,
            related_model_version=model.model_name,
            previous_state={"lifecycle_status": prev_stage},
            new_state={"lifecycle_status": "staging"},
        )
        _commit_with_clearml_sync(db, clearml_adapter, sync_ops)
        db.refresh(model)
        return model

    @staticmethod
    def rollback_to_production(
        db: Session,
        model_id: str,
        payload: LifecycleActionRequest,
        actor: ActorContext,
        clearml_adapter: ClearMLRegistryProtocol,
    ) -> GovernedModel:
        model = ModelService.get_model(db, model_id)
        if model.lifecycle_status != "archived":
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="model.transition_denied",
                entity_type="model",
                entity_id=str(model.id),
                correlation_id=actor.correlation_id,
                outcome="denied",
                reason="Rollback target must be in 'archived' stage.",
                related_model_version=model.model_name,
            )
            raise InvalidTransitionError(
                f"Rollback requires an 'archived' model; '{model.model_name}' is '{model.lifecycle_status}'."
            )

        if not model.was_ever_production:
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="model.transition_denied",
                entity_type="model",
                entity_id=str(model.id),
                correlation_id=actor.correlation_id,
                outcome="denied",
                reason="Controlled rollback blocked: model never held production status.",
                related_model_version=model.model_name,
            )
            raise InvalidTransitionError(
                f"Model '{model.model_name}' never held 'production' status and cannot be rolled back."
            )

        ver_res = clearml_adapter.verify_model_artifact(
            clearml_model_id=model.clearml_model_id,
            expected_sha256=model.checkpoint_sha256,
        )
        if not ver_res.verified:
            raise PolicyViolationError("Rollback blocked: artifact SHA-256 verification failed.")

        approved_rollback = (
            db.query(ApprovalRequest)
            .filter(
                ApprovalRequest.model_id == model.id,
                ApprovalRequest.request_type == "rollback",
                ApprovalRequest.status == "approved",
            )
            .first()
        )
        if not approved_rollback and not payload.emergency_override:
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="model.transition_denied",
                entity_type="model",
                entity_id=str(model.id),
                correlation_id=actor.correlation_id,
                outcome="denied",
                reason="Rollback requires an approved rollback request or emergency_override by admin.",
                related_model_version=model.model_name,
            )
            raise PolicyViolationError(
                "Rollback requires an approved rollback ApprovalRequest or emergency_override=true."
            )

        sync_ops = ModelService._archive_existing_in_stage(
            db,
            model_family=model.model_family,
            stage="production",
            exclude_model_id=model.id,
            actor=actor,
            reason=f"Replaced by rollback to {model.model_name}: {payload.reason}",
        )

        model.lifecycle_status = "production"
        model.updated_at = datetime.now(timezone.utc)
        if approved_rollback is not None:
            approved_rollback.status = "executed"
        sync_ops.append((model.clearml_model_id, "archived", "production", model.model_family))

        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="model.rollback",
            entity_type="model",
            entity_id=str(model.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=payload.reason,
            related_model_version=model.model_name,
            previous_state={"lifecycle_status": "archived"},
            new_state={
                "lifecycle_status": "production",
                "emergency_override": payload.emergency_override,
            },
        )
        _commit_with_clearml_sync(db, clearml_adapter, sync_ops)
        db.refresh(model)
        return model

    @staticmethod
    def archive_model(
        db: Session,
        model_id: str,
        payload: LifecycleActionRequest,
        actor: ActorContext,
        clearml_adapter: ClearMLRegistryProtocol,
    ) -> GovernedModel:
        model = ModelService.get_model(db, model_id)
        if model.lifecycle_status == "production" and "model:archive_production" not in actor.permissions:
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="auth.access_denied",
                entity_type="model",
                entity_id=str(model.id),
                correlation_id=actor.correlation_id,
                outcome="denied",
                reason="Only admin can archive an active production model.",
                related_model_version=model.model_name,
            )
            raise PermissionDeniedError("Archiving a production model requires 'admin' role.")

        ModelService._validate_transition(
            db, model=model, target_stage="archived", actor=actor, reason=payload.reason
        )
        prev_stage = model.lifecycle_status
        model.lifecycle_status = "archived"
        model.updated_at = datetime.now(timezone.utc)
        _invalidate_open_approvals(
            db,
            model=model,
            actor=actor,
            reason=f"Model archived: {payload.reason}",
            include_approved=True,
        )
        sync_ops = [(model.clearml_model_id, prev_stage, "archived", model.model_family)]

        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="model.archived",
            entity_type="model",
            entity_id=str(model.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=payload.reason,
            related_model_version=model.model_name,
            previous_state={"lifecycle_status": prev_stage},
            new_state={"lifecycle_status": "archived"},
        )
        _commit_with_clearml_sync(db, clearml_adapter, sync_ops)
        db.refresh(model)
        return model
