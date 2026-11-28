"""Configurable governance policy management, evaluator, and shadow comparison."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.app.governance.adapters.clearml_adapter import ClearMLRegistryProtocol
from backend.app.governance.auth import ActorContext
from backend.app.governance.constants import DEFAULT_POLICIES
from backend.app.governance.exceptions import ResourceNotFoundError
from backend.app.governance.models import ApprovalRequest, GovernancePolicy, GovernedDataset, GovernedModel
from backend.app.governance.schemas import (
    GovernancePolicyOut,
    PolicyCheckResult,
    PolicyUpdateRequest,
)
from backend.app.governance.services.audit_service import AuditService


GENERATOR_METRICS_HIGHER_BETTER = {
    "val_iou_mean",
    "val_giou_mean",
    "val_column_accuracy",
    "val_reading_order_kendall_tau",
    "val_macro_column_accuracy",
}
GENERATOR_METRICS_LOWER_BETTER = {
    "val_loss_total",
    "val_canvas_violation_rate",
    "val_unintended_overlap_mean_area",
    "val_unintended_overlap_rate",
    "val_aspect_ratio_log_error",
}


class PolicyService:
    """Manages DB-backed governance policies, automated policy evaluation, and shadow comparison."""

    @staticmethod
    def ensure_default_policies(db: Session) -> None:
        """Seed default governance policies idempotently."""
        existing_keys = {row[0] for row in db.query(GovernancePolicy.policy_key).all()}
        added = False
        for item in DEFAULT_POLICIES:
            if item["policy_key"] in existing_keys:
                continue
            db.add(
                GovernancePolicy(
                    policy_key=item["policy_key"],
                    name=item["name"],
                    description=item["description"],
                    model_family=item["model_family"],
                    policy_type=item["policy_type"],
                    config=dict(item["config"]),
                    is_enabled=bool(item["is_enabled"]),
                    enforcement_mode=item["enforcement_mode"],
                    updated_by="system-seed",
                )
            )
            added = True
        if added:
            db.commit()

    @staticmethod
    def to_out(policy: GovernancePolicy) -> GovernancePolicyOut:
        return GovernancePolicyOut(
            id=str(policy.id),
            policy_key=policy.policy_key,
            name=policy.name,
            description=policy.description,
            model_family=policy.model_family,
            policy_type=policy.policy_type,
            config=policy.config or {},
            is_enabled=policy.is_enabled,
            enforcement_mode=policy.enforcement_mode,
            updated_by=policy.updated_by,
            created_at=policy.created_at.isoformat(),
            updated_at=policy.updated_at.isoformat(),
        )

    @staticmethod
    def list_policies(db: Session, *, model_family: Optional[str] = None) -> list[GovernancePolicy]:
        PolicyService.ensure_default_policies(db)
        q = db.query(GovernancePolicy)
        if model_family:
            q = q.filter(
                (GovernancePolicy.model_family.is_(None))
                | (GovernancePolicy.model_family == model_family)
            )
        return q.order_by(GovernancePolicy.policy_key.asc()).all()

    @staticmethod
    def get_policy(db: Session, policy_key: str) -> GovernancePolicy:
        PolicyService.ensure_default_policies(db)
        policy = db.query(GovernancePolicy).filter(GovernancePolicy.policy_key == policy_key).first()
        if not policy:
            raise ResourceNotFoundError(f"Governance policy '{policy_key}' not found.")
        return policy

    @staticmethod
    def get_policy_config(db: Session, policy_key: str) -> tuple[bool, str, dict[str, Any]]:
        policy = PolicyService.get_policy(db, policy_key)
        return policy.is_enabled, policy.enforcement_mode, dict(policy.config or {})

    @staticmethod
    def update_policy(
        db: Session,
        policy_key: str,
        payload: PolicyUpdateRequest,
        actor: ActorContext,
    ) -> GovernancePolicy:
        policy = PolicyService.get_policy(db, policy_key)
        prev_state = {
            "is_enabled": policy.is_enabled,
            "enforcement_mode": policy.enforcement_mode,
            "config": dict(policy.config or {}),
            "description": policy.description,
        }

        if payload.is_enabled is not None:
            policy.is_enabled = payload.is_enabled
        if payload.enforcement_mode is not None:
            policy.enforcement_mode = payload.enforcement_mode
        if payload.config is not None:
            merged = dict(policy.config or {})
            merged.update(payload.config)
            policy.config = merged
        if payload.description is not None:
            policy.description = payload.description.strip()

        policy.updated_by = actor.actor_id
        policy.updated_at = datetime.now(timezone.utc)

        # Invalidate pending approval requests affected by policy changes
        pending_q = db.query(ApprovalRequest).filter(ApprovalRequest.status == "pending")
        pending_reqs: list[ApprovalRequest]
        if policy.model_family:
            family_model_ids = [
                m_id
                for (m_id,) in db.query(GovernedModel.id)
                .filter(GovernedModel.model_family == policy.model_family)
                .all()
            ]
            pending_reqs = (
                pending_q.filter(ApprovalRequest.model_id.in_(family_model_ids)).all()
                if family_model_ids
                else []
            )
        else:
            pending_reqs = pending_q.all()

        now = datetime.now(timezone.utc)
        for req in pending_reqs:
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
                reason=f"Invalidated due to update of policy '{policy_key}'.",
                previous_state={"status": "pending"},
                new_state={"status": "invalidated"},
            )

        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="policy.updated",
            entity_type="policy",
            entity_id=policy.policy_key,
            correlation_id=actor.correlation_id,
            outcome="success",
            reason=payload.reason,
            previous_state=prev_state,
            new_state={
                "is_enabled": policy.is_enabled,
                "enforcement_mode": policy.enforcement_mode,
                "config": dict(policy.config or {}),
                "description": policy.description,
            },
        )
        db.commit()
        db.refresh(policy)
        return policy

    @staticmethod
    def evaluate_model(
        db: Session,
        model: GovernedModel,
        *,
        target_stage: str,
        clearml_adapter: ClearMLRegistryProtocol,
    ) -> tuple[bool, list[PolicyCheckResult], list[str]]:
        """Evaluate all enabled policies applicable to the model and target lifecycle stage."""
        policies = [p for p in PolicyService.list_policies(db, model_family=model.model_family) if p.is_enabled]
        checks: list[PolicyCheckResult] = []
        blocking_reasons: list[str] = []

        for policy in policies:
            cfg = policy.config or {}
            key = policy.policy_key
            passed = True
            msg = "Passed"
            details: dict[str, Any] = {}

            if key == "require_dataset_lineage":
                allow_legacy = bool(cfg.get("allow_legacy_fallback", True))
                legacy_vers = set(cfg.get("legacy_versions", ["0.1.0"]))
                if model.dataset_id is not None:
                    ds = db.query(GovernedDataset).filter(GovernedDataset.id == model.dataset_id).first()
                    if ds is None or not ds.is_active:
                        passed = False
                        msg = "Linked dataset is missing or inactive."
                    else:
                        details = {"dataset": f"{ds.dataset_name}:{ds.dataset_version}"}
                elif allow_legacy and model.semantic_version in legacy_vers and model.legacy_dataset_identifier:
                    details = {"legacy_fallback": model.legacy_dataset_identifier}
                else:
                    passed = False
                    msg = "Model is missing a valid governed dataset_id reference."

            elif key == "require_artifact_integrity":
                require_sha = bool(cfg.get("require_sha256", True))
                verify_bundle = bool(cfg.get("verify_clearml_bundle", True))
                if require_sha and (not model.checkpoint_sha256 or len(model.checkpoint_sha256) != 64):
                    passed = False
                    msg = "Missing valid 64-character checkpoint SHA-256."
                elif verify_bundle and target_stage in ("staging", "production"):
                    ver_res = clearml_adapter.verify_model_artifact(
                        clearml_model_id=model.clearml_model_id,
                        expected_sha256=model.checkpoint_sha256,
                    )
                    details = {"is_mock": ver_res.is_mock, **ver_res.details}
                    if not ver_res.verified:
                        passed = False
                        msg = "ClearML artifact bundle SHA-256 verification failed."

            elif key == "require_reproducibility_metadata":
                required_fields = cfg.get(
                    "required_fields",
                    ["clearml_training_task_id", "git_commit", "environment_summary"],
                )
                missing = []
                for f in required_fields:
                    val = getattr(model, f, None)
                    if not val:
                        missing.append(f)
                if missing:
                    passed = False
                    msg = f"Missing required reproducibility fields: {missing}"
                    details = {"missing_fields": missing}

            elif key == "generator_quality_gates":
                metrics = model.metrics_summary or {}
                if target_stage in ("staging", "production"):
                    min_iou = float(cfg.get("min_val_iou_mean", 0.25))
                    max_viol = float(cfg.get("max_canvas_violation_rate", 0.05))
                    max_ovl = float(cfg.get("max_unintended_overlap_mean_area", 0.15))

                    failures = []
                    if "val_iou_mean" not in metrics:
                        failures.append("missing 'val_iou_mean'")
                    elif float(metrics["val_iou_mean"]) < min_iou:
                        failures.append(f"val_iou_mean={float(metrics['val_iou_mean']):.4f} < {min_iou}")

                    if "val_canvas_violation_rate" in metrics and float(metrics["val_canvas_violation_rate"]) > max_viol:
                        failures.append(
                            f"val_canvas_violation_rate={float(metrics['val_canvas_violation_rate']):.4f} > {max_viol}"
                        )
                    if (
                        "val_unintended_overlap_mean_area" in metrics
                        and float(metrics["val_unintended_overlap_mean_area"]) > max_ovl
                    ):
                        failures.append(
                            f"val_unintended_overlap_mean_area={float(metrics['val_unintended_overlap_mean_area']):.4f} > {max_ovl}"
                        )

                    if failures:
                        passed = False
                        msg = "; ".join(failures)
                        details = {"failures": failures}
                else:
                    if not metrics:
                        passed = False
                        msg = "Evaluation metrics_summary must not be empty at candidate registration."

            elif key == "retriever_quality_gates":
                metrics = model.metrics_summary or {}
                deps = model.dependencies_summary or {}
                if target_stage in ("staging", "production"):
                    min_posters = int(cfg.get("min_train_posters", 100))
                    require_train_only = bool(cfg.get("require_train_only", True))
                    n_posters = int(metrics.get("n_train_posters", deps.get("n_train_posters", 0)) or 0)
                    train_only_flag = metrics.get("train_only_verified", deps.get("train_only_verified", True))

                    failures = []
                    if n_posters < min_posters:
                        failures.append(f"n_train_posters={n_posters} < {min_posters}")
                    if require_train_only and train_only_flag is False:
                        failures.append("anti-leakage check failed (train_only_verified=false)")
                    if failures:
                        passed = False
                        msg = "; ".join(failures)
                        details = {"failures": failures}
                else:
                    if not metrics:
                        passed = False
                        msg = "Retriever evaluation metrics_summary must not be empty."

            elif key == "drift_response_policy":
                block_on_crit = bool(cfg.get("block_promotion_on_critical_drift", True))
                if target_stage == "production" and block_on_crit and model.drift_status == "critical":
                    passed = False
                    msg = "Model currently has unresolved critical data drift."
                    details = {"drift_status": model.drift_status}

            else:
                # Non-gate configuration policies (production_approval_policy, shadow_comparison_policy)
                continue

            checks.append(
                PolicyCheckResult(
                    policy_key=key,
                    passed=passed,
                    enforcement_mode=policy.enforcement_mode,
                    message=msg,
                    details=details,
                )
            )
            if not passed and policy.enforcement_mode == "blocking":
                blocking_reasons.append(f"[{key}] {msg}")

        return len(blocking_reasons) == 0, checks, blocking_reasons

    @staticmethod
    def compute_shadow_comparison(
        db: Session,
        candidate_model: GovernedModel,
        *,
        shadow_eval_payload: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[GovernedModel], dict[str, Any], bool]:
        """Compare candidate/staging model against active production model in same family."""
        baseline = (
            db.query(GovernedModel)
            .filter(
                GovernedModel.model_family == candidate_model.model_family,
                GovernedModel.lifecycle_status == "production",
                GovernedModel.id != candidate_model.id,
            )
            .first()
        )

        enabled, enforcement_mode, cfg = PolicyService.get_policy_config(db, "shadow_comparison_policy")

        if baseline is None:
            report = {
                "has_production_baseline": False,
                "baseline_model_id": None,
                "baseline_model_name": None,
                "metric_deltas": {},
                "shadow_eval_payload": shadow_eval_payload or {},
                "regression_detected": False,
                "regression_reasons": [],
                "summary": "No active production baseline exists for this family; initial production candidate.",
            }
            return None, report, False

        cand_metrics = candidate_model.metrics_summary or {}
        base_metrics = baseline.metrics_summary or {}
        shared_keys = sorted(set(cand_metrics.keys()) & set(base_metrics.keys()))

        deltas: dict[str, dict[str, Any]] = {}
        regression_reasons: list[str] = []

        for k in shared_keys:
            c_val = cand_metrics.get(k)
            b_val = base_metrics.get(k)
            if not isinstance(c_val, (int, float)) or not isinstance(b_val, (int, float)):
                continue
            diff = round(float(c_val) - float(b_val), 6)
            direction = "neutral"
            if k in GENERATOR_METRICS_HIGHER_BETTER:
                direction = "improved" if diff > 0 else ("regressed" if diff < 0 else "unchanged")
            elif k in GENERATOR_METRICS_LOWER_BETTER:
                direction = "improved" if diff < 0 else ("regressed" if diff > 0 else "unchanged")
            deltas[k] = {
                "candidate": float(c_val),
                "baseline": float(b_val),
                "delta": diff,
                "status": direction,
            }

        block_iou = bool(cfg.get("block_on_iou_regression", True))
        max_drop = float(cfg.get("max_allowed_iou_drop", 0.0))
        if "val_iou_mean" in deltas:
            iou_delta = deltas["val_iou_mean"]["delta"]
            if iou_delta < -abs(max_drop):
                regression_reasons.append(
                    f"val_iou_mean regressed by {abs(iou_delta):.4f} vs production {baseline.model_name} "
                    f"(max allowed drop {abs(max_drop):.4f})."
                )

        if shadow_eval_payload and shadow_eval_payload.get("regression_detected"):
            regression_reasons.append(
                str(shadow_eval_payload.get("summary", "External shadow evaluation reported regression."))
            )

        regression_detected = len(regression_reasons) > 0
        regression_blocked = (
            enabled
            and enforcement_mode == "blocking"
            and block_iou
            and regression_detected
        )

        report = {
            "has_production_baseline": True,
            "baseline_model_id": str(baseline.id),
            "baseline_model_name": baseline.model_name,
            "baseline_semantic_version": baseline.semantic_version,
            "metric_deltas": deltas,
            "shadow_eval_payload": shadow_eval_payload or {},
            "regression_detected": regression_detected,
            "regression_reasons": regression_reasons,
            "summary": (
                "Regression detected vs active production model."
                if regression_detected
                else f"No regression detected vs active production model {baseline.model_name}."
            ),
        }
        return baseline, report, regression_blocked
