"""Canonical constants, enums, lifecycle rules, and RBAC permission map."""

from __future__ import annotations

import re

# Model families & architectures aligned with src/registry/clearml_registry.py
MODEL_FAMILIES = ("poster_layout_generator", "poster_layout_retriever")
FAMILY_ARCHITECTURES = {
    "poster_layout_generator": "TemplateConditionedLayoutModel",
    "poster_layout_retriever": "LayoutRetriever",
}

SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

# Lifecycle stages matching ClearML Native Model Registry
LIFECYCLE_STAGES = ("candidate", "staging", "production", "archived")

# Allowed explicit state transitions (current_stage -> set of valid next stages)
ALLOWED_STAGE_TRANSITIONS: dict[str, set[str]] = {
    "candidate": {"staging", "archived"},
    "staging": {"production", "archived"},
    "production": {"staging", "archived"},
    "archived": {"production"},  # Controlled rollback only (requires was_ever_production=True)
}

COMPLIANCE_STATUSES = ("pending", "compliant", "non_compliant")
DRIFT_STATUSES = ("unknown", "none", "warning", "critical")

# Dataset constants
DATASET_SOURCE_TYPES = ("scipostlayout", "hitl_export", "hybrid", "mock")
DATASET_STORAGE_SYSTEMS = ("minidvc", "s3", "clearml_dataset", "local")

# Approval workflow constants
APPROVAL_REQUEST_TYPES = ("promotion", "rollback")
APPROVAL_REQUEST_STATUSES = ("pending", "approved", "executed", "rejected", "cancelled", "invalidated")
APPROVAL_DECISIONS = ("approved", "rejected")

# Policy constants
POLICY_ENFORCEMENT_MODES = ("blocking", "warning")

# Roles & Permissions
ROLES = ("viewer", "ml_engineer", "reviewer", "admin")

ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset({"governance:read"}),
    "ml_engineer": frozenset(
        {
            "governance:read",
            "dataset:write",
            "model:register",
            "model:update_metadata",
            "model:promote_staging",
            "model:archive_non_prod",
            "approval:request",
            "drift:record",
        }
    ),
    "reviewer": frozenset(
        {
            "governance:read",
            "approval:decide",
        }
    ),
    "admin": frozenset(
        {
            "governance:read",
            "dataset:write",
            "model:register",
            "model:update_metadata",
            "model:promote_staging",
            "model:archive_non_prod",
            "model:promote_production",
            "model:rollback",
            "model:archive_production",
            "model:demote_production",
            "approval:request",
            "approval:decide",
            "drift:record",
            "policy:write",
            "rbac:manage",
        }
    ),
}

# Audit constants
AUDIT_OUTCOMES = ("success", "denied", "failed")

AUDIT_ACTIONS = (
    "dataset.registered",
    "dataset.updated",
    "model.registered",
    "model.metadata_updated",
    "model.evaluated",
    "model.lifecycle_transition",
    "model.promoted_to_production",
    "model.rollback",
    "model.archived",
    "model.transition_denied",
    "approval.requested",
    "approval.approved",
    "approval.rejected",
    "approval.cancelled",
    "approval.invalidated",
    "policy.created",
    "policy.updated",
    "role.assigned",
    "role.revoked",
    "drift.event_recorded",
    "auth.access_denied",
)

# Default seeded policies (Area 2.1 & 2.3)
DEFAULT_POLICIES: list[dict] = [
    {
        "policy_key": "require_dataset_lineage",
        "name": "Require Governed Dataset Lineage",
        "description": "Every registered model must link to an active governed dataset unless explicitly exempted as a legacy baseline.",
        "model_family": None,
        "policy_type": "lineage",
        "enforcement_mode": "blocking",
        "is_enabled": True,
        "config": {
            "allow_legacy_fallback": True,
            "legacy_versions": ["0.1.0"],
        },
    },
    {
        "policy_key": "require_artifact_integrity",
        "name": "Require SHA-256 Artifact Integrity",
        "description": "Require valid 64-char SHA-256 checkpoint checksum and ClearML bundle verification prior to promotion.",
        "model_family": None,
        "policy_type": "integrity",
        "enforcement_mode": "blocking",
        "is_enabled": True,
        "config": {
            "require_sha256": True,
            "verify_clearml_bundle": True,
        },
    },
    {
        "policy_key": "require_reproducibility_metadata",
        "name": "Require Reproducibility Metadata",
        "description": "Require ClearML training task ID, git commit, and environment summary on all governed models.",
        "model_family": None,
        "policy_type": "reproducibility",
        "enforcement_mode": "blocking",
        "is_enabled": True,
        "config": {
            "required_fields": ["clearml_training_task_id", "git_commit", "environment_summary"],
        },
    },
    {
        "policy_key": "generator_quality_gates",
        "name": "Neural Generator Quality Gates",
        "description": "Minimum evaluation quality thresholds for TemplateConditionedLayoutModel promotion.",
        "model_family": "poster_layout_generator",
        "policy_type": "quality_gate",
        "enforcement_mode": "blocking",
        "is_enabled": True,
        "config": {
            "min_val_iou_mean": 0.25,
            "max_canvas_violation_rate": 0.05,
            "max_unintended_overlap_mean_area": 0.15,
        },
    },
    {
        "policy_key": "retriever_quality_gates",
        "name": "Layout Retriever Quality & Anti-Leakage Gates",
        "description": "Ensure LayoutRetriever was fitted exclusively on train split posters and meets minimum sample count.",
        "model_family": "poster_layout_retriever",
        "policy_type": "quality_gate",
        "enforcement_mode": "blocking",
        "is_enabled": True,
        "config": {
            "require_train_only": True,
            "min_train_posters": 100,
        },
    },
    {
        "policy_key": "production_approval_policy",
        "name": "Production Approval & Stage Uniqueness Policy",
        "description": "Require human approval for production promotion, enforce separation of duties, and keep staging/production unambiguous.",
        "model_family": None,
        "policy_type": "approval",
        "enforcement_mode": "blocking",
        "is_enabled": True,
        "config": {
            "require_human_approval": True,
            "allow_self_approval": False,
            "single_active_staging": True,
            "single_active_production": True,
        },
    },
    {
        "policy_key": "shadow_comparison_policy",
        "name": "Automated Shadow Comparison Before Approval",
        "description": "Compare candidate/staging model against the active production model before human approval.",
        "model_family": None,
        "policy_type": "shadow_comparison",
        "enforcement_mode": "blocking",
        "is_enabled": True,
        "config": {
            "require_comparison_when_prod_exists": True,
            "block_on_iou_regression": True,
            "max_allowed_iou_drop": 0.0,
        },
    },
    {
        "policy_key": "drift_response_policy",
        "name": "Data Drift Governance Policy",
        "description": "Control compliance and promotion behavior when critical data drift is reported.",
        "model_family": None,
        "policy_type": "drift",
        "enforcement_mode": "blocking",
        "is_enabled": True,
        "config": {
            "block_promotion_on_critical_drift": True,
            "mark_model_degraded_on_critical": True,
        },
    },
]
