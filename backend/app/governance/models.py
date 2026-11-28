"""SQLAlchemy 2.0 ORM models for the Infographix MLOps Governance module."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    event,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.models.generation_run import JSONVariant


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GovernedDataset(Base):
    """Lightweight dataset governance catalog referencing minidvc / S3 / ClearML Datasets."""

    __tablename__ = "governed_datasets"
    __table_args__ = (
        UniqueConstraint("dataset_name", "dataset_version", name="uq_governed_datasets_name_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="scipostlayout")
    owner: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")

    storage_system: Mapped[str] = mapped_column(String(32), nullable=False, default="minidvc")
    storage_ref: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    data_format: Mapped[str] = mapped_column(String(32), nullable=False, default="parquet")
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False, default="2.0.0")

    record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    split_config: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)
    quality_summary: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)
    preprocessing_version: Mapped[str] = mapped_column(String(32), nullable=False, default="2.0.0")
    usage_policy: Mapped[str] = mapped_column(String(255), nullable=False, default="internal-research")

    parent_dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("governed_datasets.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    models: Mapped[list["GovernedModel"]] = relationship(
        back_populates="dataset",
        passive_deletes="all",
    )


class GovernedModel(Base):
    """Authoritative governance control record indexing a ClearML-registered model version."""

    __tablename__ = "governed_models"
    __table_args__ = (
        UniqueConstraint("clearml_model_id", name="uq_governed_models_clearml_id"),
        UniqueConstraint("model_family", "semantic_version", name="uq_governed_models_family_version"),
        Index("ix_governed_models_family_stage", "model_family", "lifecycle_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clearml_model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_name: Mapped[str] = mapped_column(String(192), nullable=False)
    model_family: Mapped[str] = mapped_column(String(64), nullable=False)
    semantic_version: Mapped[str] = mapped_column(String(32), nullable=False)
    architecture: Mapped[str] = mapped_column(String(128), nullable=False)

    parent_model_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("governed_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    clearml_training_task_id: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    checkpoint_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    git_commit: Mapped[str] = mapped_column(String(64), nullable=False)

    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("governed_datasets.id", ondelete="RESTRICT"),
        nullable=True,
    )
    legacy_dataset_identifier: Mapped[str | None] = mapped_column(String(255), nullable=True)

    retriever_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    retriever_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False, default="v1.0.0")

    environment_summary: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)
    dependencies_summary: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)
    metrics_summary: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)

    lifecycle_status: Mapped[str] = mapped_column(String(32), nullable=False, default="candidate")
    was_ever_production: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    compliance_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    drift_status: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")

    owner: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list[str]] = mapped_column(JSONVariant, nullable=False, default=list)

    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    dataset: Mapped[GovernedDataset | None] = relationship(back_populates="models")
    approval_requests: Mapped[list["ApprovalRequest"]] = relationship(
        back_populates="model",
        foreign_keys="ApprovalRequest.model_id",
        cascade="all, delete-orphan",
    )
    drift_events: Mapped[list["DriftEvent"]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
    )


class GovernancePolicy(Base):
    """Database-backed configurable governance policy rule."""

    __tablename__ = "governance_policies"
    __table_args__ = (UniqueConstraint("policy_key", name="uq_governance_policies_key"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    policy_key: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(192), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    model_family: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    policy_type: Mapped[str] = mapped_column(String(64), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    enforcement_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="blocking")
    updated_by: Mapped[str] = mapped_column(String(128), nullable=False, default="system-seed")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )


class ApprovalRequest(Base):
    """Human approval workflow request for production promotion or rollback."""

    __tablename__ = "approval_requests"
    __table_args__ = (Index("ix_approval_requests_model_status", "model_id", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("governed_models.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_stage: Mapped[str] = mapped_column(String(32), nullable=False, default="production")
    request_type: Mapped[str] = mapped_column(String(32), nullable=False, default="promotion")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    requested_by: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)

    automated_check_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    automated_check_summary: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)

    baseline_model_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("governed_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    shadow_comparison_report: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model: Mapped[GovernedModel] = relationship(
        back_populates="approval_requests",
        foreign_keys=[model_id],
    )
    decisions: Mapped[list["ApprovalDecision"]] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",
    )


class ApprovalDecision(Base):
    """Immutable human approval/rejection decision attached to an ApprovalRequest."""

    __tablename__ = "approval_decisions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("approval_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    decided_by: Mapped[str] = mapped_column(String(128), nullable=False)
    decided_role: Mapped[str] = mapped_column(String(32), nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=False)
    policy_override: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    request: Mapped[ApprovalRequest] = relationship(back_populates="decisions")


class DriftEvent(Base):
    """Idempotent monitoring/drift integration record received from Data Drift module."""

    __tablename__ = "drift_events"
    __table_args__ = (
        UniqueConstraint("monitoring_run_id", name="uq_drift_events_monitoring_run_id"),
        Index("ix_drift_events_model_detected", "model_id", "detected_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("governed_models.id", ondelete="CASCADE"),
        nullable=False,
    )
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("governed_datasets.id", ondelete="SET NULL"),
        nullable=True,
    )
    monitoring_run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    monitoring_window: Mapped[str] = mapped_column(String(128), nullable=False)
    drift_status: Mapped[str] = mapped_column(String(32), nullable=False)
    drift_metrics: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)
    threshold_config: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)
    policy_action_taken: Mapped[str] = mapped_column(String(128), nullable=False, default="recorded_only")
    is_mock: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    model: Mapped[GovernedModel] = relationship(back_populates="drift_events")


class GovernanceRoleAssignment(Base):
    """Persistent actor-to-role binding for governance RBAC."""

    __tablename__ = "governance_role_assignments"
    __table_args__ = (UniqueConstraint("actor_id", name="uq_governance_role_assignments_actor"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    assigned_by: Mapped[str] = mapped_column(String(128), nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )


class AuditEvent(Base):
    """Append-only governance audit ledger (zero foreign keys so history outlives deletions)."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_entity", "entity_type", "entity_id"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        index=True,
    )
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    actor_role: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    related_model_version: Mapped[str | None] = mapped_column(String(192), nullable=True)
    related_dataset_version: Mapped[str | None] = mapped_column(String(192), nullable=True)
    previous_state: Mapped[dict[str, Any] | None] = mapped_column(JSONVariant, nullable=True)
    new_state: Mapped[dict[str, Any] | None] = mapped_column(JSONVariant, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False, default="success")
    correlation_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)


@event.listens_for(AuditEvent, "before_update")
def _prevent_audit_update(mapper, connection, target):  # pragma: no cover - defensive guard
    from backend.app.governance.exceptions import AuditImmutabilityError

    raise AuditImmutabilityError("AuditEvent records are append-only and cannot be modified.")


@event.listens_for(AuditEvent, "before_delete")
def _prevent_audit_delete(mapper, connection, target):  # pragma: no cover - defensive guard
    from backend.app.governance.exceptions import AuditImmutabilityError

    raise AuditImmutabilityError("AuditEvent records are append-only and cannot be deleted.")
