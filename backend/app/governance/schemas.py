"""Versioned Pydantic v2 schemas for the Infographix MLOps Governance APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from backend.app.governance.constants import (
    APPROVAL_DECISIONS,
    APPROVAL_REQUEST_TYPES,
    DATASET_SOURCE_TYPES,
    DATASET_STORAGE_SYSTEMS,
    DRIFT_STATUSES,
    MODEL_FAMILIES,
    POLICY_ENFORCEMENT_MODES,
    ROLES,
    SEMVER_PATTERN,
    SHA256_PATTERN,
)


# ---------------------------------------------------------------------------
# Consistent Error Envelope
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Auth & RBAC Schemas
# ---------------------------------------------------------------------------

class ActorMeOut(BaseModel):
    actor_id: str
    role: Literal["viewer", "ml_engineer", "reviewer", "admin"]
    auth_source: str
    permissions: list[str]


class RoleAssignRequest(BaseModel):
    role: Literal["viewer", "ml_engineer", "reviewer", "admin"]
    notes: str = Field(default="", max_length=500)


class RoleAssignmentOut(BaseModel):
    id: str
    actor_id: str
    role: str
    assigned_by: str
    notes: str
    is_active: bool
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Dataset Schemas
# ---------------------------------------------------------------------------

class DatasetCreateRequest(BaseModel):
    dataset_name: str = Field(..., min_length=2, max_length=128)
    dataset_version: str = Field(..., min_length=1, max_length=64)
    source_type: Literal["scipostlayout", "hitl_export", "hybrid", "mock"] = "scipostlayout"
    owner: Optional[str] = Field(default=None, max_length=128)
    description: str = Field(default="", max_length=2000)
    storage_system: Literal["minidvc", "s3", "clearml_dataset", "local"] = "minidvc"
    storage_ref: str = Field(..., min_length=1, max_length=512)
    checksum_sha256: str = Field(..., min_length=64, max_length=64)
    data_format: str = Field(default="parquet", min_length=1, max_length=32)
    schema_version: str = Field(default="2.0.0", min_length=1, max_length=32)
    record_count: int = Field(default=0, ge=0)
    size_bytes: int = Field(default=0, ge=0)
    split_config: dict[str, Any] = Field(default_factory=dict)
    quality_summary: dict[str, Any] = Field(default_factory=dict)
    preprocessing_version: str = Field(default="2.0.0", min_length=1, max_length=32)
    usage_policy: str = Field(default="internal-research", min_length=1, max_length=255)
    parent_dataset_id: Optional[str] = None

    @field_validator("checksum_sha256")
    @classmethod
    def _validate_sha256(cls, v: str) -> str:
        clean = v.strip().lower()
        if not SHA256_PATTERN.match(clean):
            raise ValueError("checksum_sha256 must be a valid 64-character hexadecimal SHA-256 string.")
        return clean


class DatasetIngestManifestRequest(BaseModel):
    manifest_path: str = Field(..., min_length=1, max_length=512)
    feature_report_path: Optional[str] = Field(default=None, max_length=512)
    dataset_name: str = Field(default="scipostlayout-features", min_length=2, max_length=128)
    dataset_version: Optional[str] = Field(default=None, max_length=64)
    source_type: Literal["scipostlayout", "hitl_export", "hybrid", "mock"] = "scipostlayout"
    description: str = Field(default="", max_length=2000)
    usage_policy: str = Field(default="academic-scipostlayout-research", max_length=255)


class DatasetOut(BaseModel):
    id: str
    dataset_name: str
    dataset_version: str
    source_type: str
    owner: str
    description: str
    storage_system: str
    storage_ref: str
    checksum_sha256: str
    data_format: str
    schema_version: str
    record_count: int
    size_bytes: int
    split_config: dict[str, Any]
    quality_summary: dict[str, Any]
    preprocessing_version: str
    usage_policy: str
    parent_dataset_id: Optional[str] = None
    is_active: bool
    linked_models_count: int = 0
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Model Governance Schemas
# ---------------------------------------------------------------------------

class ModelRegisterRequest(BaseModel):
    clearml_model_id: str = Field(..., min_length=3, max_length=128)
    model_family: Literal["poster_layout_generator", "poster_layout_retriever"]
    semantic_version: str = Field(..., min_length=5, max_length=32)
    architecture: Optional[str] = Field(default=None, max_length=128)
    parent_model_id: Optional[str] = None
    clearml_training_task_id: str = Field(..., min_length=1, max_length=128)
    artifact_uri: str = Field(..., min_length=1, max_length=512)
    checkpoint_sha256: str = Field(..., min_length=64, max_length=64)
    git_commit: str = Field(..., min_length=4, max_length=64)
    dataset_id: Optional[str] = None
    legacy_dataset_identifier: Optional[str] = Field(default=None, max_length=255)
    retriever_model_id: Optional[str] = Field(default=None, max_length=128)
    retriever_version: Optional[str] = Field(default=None, max_length=32)
    schema_version: str = Field(default="v1.0.0", max_length=32)
    environment_summary: dict[str, Any] = Field(default_factory=dict)
    dependencies_summary: dict[str, Any] = Field(default_factory=dict)
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    owner: Optional[str] = Field(default=None, max_length=128)
    description: str = Field(default="", max_length=2000)
    tags: list[str] = Field(default_factory=list)

    @field_validator("semantic_version")
    @classmethod
    def _validate_semver(cls, v: str) -> str:
        clean = v.strip()
        if clean.startswith("v"):
            clean = clean[1:]
        if not SEMVER_PATTERN.match(clean):
            raise ValueError("semantic_version must follow X.Y.Z format (e.g. '0.2.0').")
        return clean

    @field_validator("checkpoint_sha256")
    @classmethod
    def _validate_ckpt_sha(cls, v: str) -> str:
        clean = v.strip().lower()
        if not SHA256_PATTERN.match(clean):
            raise ValueError("checkpoint_sha256 must be a valid 64-character hex SHA-256 string.")
        return clean


class ModelMetadataUpdateRequest(BaseModel):
    owner: Optional[str] = Field(default=None, min_length=1, max_length=128)
    description: Optional[str] = Field(default=None, max_length=2000)
    tags: Optional[list[str]] = None
    reason: str = Field(..., min_length=3, max_length=500)


class LifecycleActionRequest(BaseModel):
    reason: str = Field(..., min_length=3, max_length=1000)
    emergency_override: bool = False


class GovernedModelOut(BaseModel):
    id: str
    clearml_model_id: str
    model_name: str
    model_family: str
    semantic_version: str
    architecture: str
    parent_model_id: Optional[str] = None
    clearml_training_task_id: str
    artifact_uri: str
    checkpoint_sha256: str
    git_commit: str
    dataset_id: Optional[str] = None
    legacy_dataset_identifier: Optional[str] = None
    retriever_model_id: Optional[str] = None
    retriever_version: Optional[str] = None
    schema_version: str
    environment_summary: dict[str, Any]
    dependencies_summary: dict[str, Any]
    metrics_summary: dict[str, Any]
    lifecycle_status: str
    was_ever_production: bool
    compliance_status: str
    drift_status: str
    owner: str
    description: str
    tags: list[str]
    registered_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Policy & Shadow Comparison Schemas
# ---------------------------------------------------------------------------

class PolicyCheckResult(BaseModel):
    policy_key: str
    passed: bool
    enforcement_mode: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class PolicyEvaluationOut(BaseModel):
    model_id: str
    model_name: str
    target_stage: str
    passed: bool
    compliance_status: str
    checks: list[PolicyCheckResult]


class PolicyUpdateRequest(BaseModel):
    is_enabled: Optional[bool] = None
    enforcement_mode: Optional[Literal["blocking", "warning"]] = None
    config: Optional[dict[str, Any]] = None
    description: Optional[str] = Field(default=None, max_length=2000)
    reason: str = Field(..., min_length=3, max_length=500)


class GovernancePolicyOut(BaseModel):
    id: str
    policy_key: str
    name: str
    description: str
    model_family: Optional[str] = None
    policy_type: str
    config: dict[str, Any]
    is_enabled: bool
    enforcement_mode: str
    updated_by: str
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Approval & Promotion Gate Schemas
# ---------------------------------------------------------------------------

class ApprovalRequestCreate(BaseModel):
    target_stage: Literal["production"] = "production"
    request_type: Literal["promotion", "rollback"] = "promotion"
    reason: str = Field(..., min_length=3, max_length=1000)
    shadow_eval_payload: Optional[dict[str, Any]] = None


class ApprovalDecisionCreate(BaseModel):
    decision: Literal["approved", "rejected"]
    comment: str = Field(..., min_length=3, max_length=1000)
    policy_override: bool = False


class ApprovalDecisionOut(BaseModel):
    id: str
    request_id: str
    decision: str
    decided_by: str
    decided_role: str
    comment: str
    policy_override: bool
    decided_at: str


class ApprovalRequestOut(BaseModel):
    id: str
    model_id: str
    model_name: str = ""
    target_stage: str
    request_type: str
    status: str
    requested_by: str
    reason: str
    automated_check_passed: bool
    automated_check_summary: dict[str, Any]
    baseline_model_id: Optional[str] = None
    shadow_comparison_report: dict[str, Any]
    decisions: list[ApprovalDecisionOut] = Field(default_factory=list)
    created_at: str
    resolved_at: Optional[str] = None


class PromotionGateOut(BaseModel):
    model_id: str
    model_name: str
    current_stage: str
    target_stage: str
    eligible: bool
    approved_request_id: Optional[str] = None
    checks: list[PolicyCheckResult]
    blocking_reasons: list[str]


# ---------------------------------------------------------------------------
# Drift Integration Schemas
# ---------------------------------------------------------------------------

class DriftEventCreateRequest(BaseModel):
    model_id: Optional[str] = None
    model_family: Optional[Literal["poster_layout_generator", "poster_layout_retriever"]] = None
    semantic_version: Optional[str] = None
    dataset_id: Optional[str] = None
    monitoring_run_id: str = Field(..., min_length=2, max_length=128)
    monitoring_window: str = Field(..., min_length=2, max_length=128)
    drift_status: Literal["none", "warning", "critical"]
    drift_metrics: dict[str, Any] = Field(default_factory=dict)
    threshold_config: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime
    is_mock: bool = False


class DriftEventOut(BaseModel):
    id: str
    model_id: str
    dataset_id: Optional[str] = None
    monitoring_run_id: str
    monitoring_window: str
    drift_status: str
    drift_metrics: dict[str, Any]
    threshold_config: dict[str, Any]
    policy_action_taken: str
    is_mock: bool
    detected_at: str
    recorded_at: str


# ---------------------------------------------------------------------------
# Audit & Overview Schemas
# ---------------------------------------------------------------------------

class AuditEventOut(BaseModel):
    id: str
    timestamp: str
    actor_id: str
    actor_role: str
    action: str
    entity_type: str
    entity_id: str
    related_model_version: Optional[str] = None
    related_dataset_version: Optional[str] = None
    previous_state: Optional[dict[str, Any]] = None
    new_state: Optional[dict[str, Any]] = None
    reason: Optional[str] = None
    outcome: str
    correlation_id: str


class GovernanceOverviewOut(BaseModel):
    total_datasets: int
    total_models: int
    models_by_stage: dict[str, int]
    models_by_compliance: dict[str, int]
    pending_approval_requests: int
    critical_drift_models: int
    active_production_models: dict[str, Optional[dict[str, Any]]]
    active_staging_models: dict[str, Optional[dict[str, Any]]]
    recent_audit_events_count: int
