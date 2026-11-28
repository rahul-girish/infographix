"""Data drift ingestion service (integration boundary for Rahul's monitoring module)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.app.governance.adapters.clearml_adapter import get_clearml_adapter
from backend.app.governance.auth import ActorContext
from backend.app.governance.exceptions import ConflictError, ResourceNotFoundError, ValidationError
from backend.app.governance.models import DriftEvent, GovernedDataset, GovernedModel
from backend.app.governance.schemas import DriftEventCreateRequest, DriftEventOut
from backend.app.governance.services.audit_service import AuditService
from backend.app.governance.services.policy_service import PolicyService


def _parse_uuid(raw_id: str, label: str = "id") -> uuid.UUID:
    try:
        return uuid.UUID(str(raw_id))
    except ValueError as exc:
        raise ValidationError(f"Invalid UUID for '{label}': {raw_id}") from exc


class DriftService:
    """Records incoming drift monitoring runs and applies configured governance policies."""

    @staticmethod
    def to_out(ev: DriftEvent) -> DriftEventOut:
        return DriftEventOut(
            id=str(ev.id),
            model_id=str(ev.model_id),
            dataset_id=str(ev.dataset_id) if ev.dataset_id else None,
            monitoring_run_id=ev.monitoring_run_id,
            monitoring_window=ev.monitoring_window,
            drift_status=ev.drift_status,
            drift_metrics=ev.drift_metrics or {},
            threshold_config=ev.threshold_config or {},
            policy_action_taken=ev.policy_action_taken,
            is_mock=ev.is_mock,
            detected_at=ev.detected_at.isoformat(),
            recorded_at=ev.recorded_at.isoformat(),
        )

    @staticmethod
    def record_drift_event(
        db: Session,
        payload: DriftEventCreateRequest,
        actor: ActorContext,
    ) -> DriftEvent:
        dup = (
            db.query(DriftEvent)
            .filter(DriftEvent.monitoring_run_id == payload.monitoring_run_id)
            .first()
        )
        if dup:
            raise ConflictError(
                f"Drift monitoring run '{payload.monitoring_run_id}' has already been recorded."
            )

        if payload.model_id:
            uid = _parse_uuid(payload.model_id, "model_id")
            model = db.query(GovernedModel).filter(GovernedModel.id == uid).first()
        elif payload.model_family and payload.semantic_version:
            ver = payload.semantic_version.lstrip("v")
            model = (
                db.query(GovernedModel)
                .filter(
                    GovernedModel.model_family == payload.model_family,
                    GovernedModel.semantic_version == ver,
                )
                .first()
            )
        else:
            raise ValidationError(
                "Provide either 'model_id' or both 'model_family' and 'semantic_version'."
            )

        if not model:
            raise ResourceNotFoundError("Target governed model for drift event not found.")

        dataset_uuid: Optional[uuid.UUID] = model.dataset_id
        if payload.dataset_id:
            dataset_uuid = _parse_uuid(payload.dataset_id, "dataset_id")
            if not db.query(GovernedDataset).filter(GovernedDataset.id == dataset_uuid).first():
                raise ResourceNotFoundError(f"Dataset '{payload.dataset_id}' not found.")

        prev_drift = model.drift_status
        prev_compliance = model.compliance_status
        model.drift_status = payload.drift_status

        enabled, _, drift_cfg = PolicyService.get_policy_config(db, "drift_response_policy")
        policy_action = "recorded_only"
        if enabled and payload.drift_status == "critical":
            if drift_cfg.get("mark_model_degraded_on_critical", True):
                model.compliance_status = "non_compliant"
                policy_action = "marked_non_compliant_critical_drift"
            elif drift_cfg.get("block_promotion_on_critical_drift", True):
                policy_action = "promotion_blocked_on_critical_drift"
        elif payload.drift_status == "none" and prev_drift in ("warning", "critical"):
            passed, _, _ = PolicyService.evaluate_model(
                db,
                model,
                target_stage=model.lifecycle_status,
                clearml_adapter=get_clearml_adapter(),
            )
            model.compliance_status = "compliant" if passed else "non_compliant"
            policy_action = "recovered_compliance_reevaluated"

        model.updated_at = datetime.now(timezone.utc)

        ev = DriftEvent(
            model_id=model.id,
            dataset_id=dataset_uuid,
            monitoring_run_id=payload.monitoring_run_id.strip(),
            monitoring_window=payload.monitoring_window.strip(),
            drift_status=payload.drift_status,
            drift_metrics=payload.drift_metrics,
            threshold_config=payload.threshold_config,
            policy_action_taken=policy_action,
            is_mock=payload.is_mock,
            detected_at=payload.detected_at,
        )
        db.add(ev)
        db.flush()

        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="drift.event_recorded",
            entity_type="drift_event",
            entity_id=str(ev.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            related_model_version=model.model_name,
            previous_state={"drift_status": prev_drift, "compliance_status": prev_compliance},
            new_state={
                "drift_status": model.drift_status,
                "compliance_status": model.compliance_status,
                "policy_action_taken": policy_action,
                "is_mock": payload.is_mock,
            },
        )
        db.commit()
        db.refresh(ev)
        return ev

    @staticmethod
    def list_drift_events(
        db: Session,
        *,
        model_id: Optional[str] = None,
        drift_status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DriftEvent]:
        q = db.query(DriftEvent)
        if model_id:
            uid = _parse_uuid(model_id, "model_id")
            q = q.filter(DriftEvent.model_id == uid)
        if drift_status:
            q = q.filter(DriftEvent.drift_status == drift_status)
        return (
            q.order_by(desc(DriftEvent.detected_at))
            .offset(max(0, offset))
            .limit(min(200, max(1, limit)))
            .all()
        )
