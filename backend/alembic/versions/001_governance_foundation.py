"""Create governance foundation tables, PG append-only audit trigger, and seed policies.

Revision ID: 001_governance_foundation
Revises:
Create Date: 2026-11-28
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa

from backend.app.governance.constants import DEFAULT_POLICIES
from backend.app.models.generation_run import JSONVariant

revision = "001_governance_foundation"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Ensure existing generation_runs table exists on brand-new databases without breaking existing ones
    if not inspector.has_table("generation_runs"):
        op.create_table(
            "generation_runs",
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("title", sa.String(255), nullable=False),
            sa.Column("authors", sa.String(500), nullable=False, server_default=""),
            sa.Column("orientation", sa.String(32), nullable=False, server_default="landscape"),
            sa.Column("engine", sa.String(64), nullable=False, server_default="adaptation"),
            sa.Column("model_name", sa.String(128), nullable=False),
            sa.Column("model_version", sa.String(64), nullable=False),
            sa.Column("retrieved_template_id", sa.String(64), nullable=True),
            sa.Column("candidate_templates", JSONVariant, nullable=False),
            sa.Column("generation_metrics", JSONVariant, nullable=False),
            sa.Column("input_payload", JSONVariant, nullable=False),
            sa.Column("request_features", JSONVariant, nullable=False),
            sa.Column("generated_layout", JSONVariant, nullable=False),
            sa.Column("edited_layout", JSONVariant, nullable=True),
            sa.Column("was_edited", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("exported_formats", JSONVariant, nullable=False),
            sa.Column("feedback_rating", sa.Integer(), nullable=True),
            sa.Column("feedback_notes", sa.Text(), nullable=True),
        )
        op.create_index("ix_generation_runs_created_at", "generation_runs", ["created_at"])
        op.create_index("ix_generation_runs_was_edited", "generation_runs", ["was_edited"])

    if not inspector.has_table("governed_datasets"):
        op.create_table(
            "governed_datasets",
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column("dataset_name", sa.String(128), nullable=False),
            sa.Column("dataset_version", sa.String(64), nullable=False),
            sa.Column("source_type", sa.String(32), nullable=False),
            sa.Column("owner", sa.String(128), nullable=False),
            sa.Column("description", sa.Text(), nullable=False, server_default=""),
            sa.Column("storage_system", sa.String(32), nullable=False),
            sa.Column("storage_ref", sa.String(512), nullable=False),
            sa.Column("checksum_sha256", sa.String(64), nullable=False),
            sa.Column("data_format", sa.String(32), nullable=False),
            sa.Column("schema_version", sa.String(32), nullable=False),
            sa.Column("record_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
            sa.Column("split_config", JSONVariant, nullable=False),
            sa.Column("quality_summary", JSONVariant, nullable=False),
            sa.Column("preprocessing_version", sa.String(32), nullable=False),
            sa.Column("usage_policy", sa.String(255), nullable=False),
            sa.Column(
                "parent_dataset_id",
                sa.Uuid(as_uuid=True),
                sa.ForeignKey("governed_datasets.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("dataset_name", "dataset_version", name="uq_governed_datasets_name_version"),
        )
        op.create_index("ix_governed_datasets_dataset_name", "governed_datasets", ["dataset_name"])
        op.create_index("ix_governed_datasets_storage_ref", "governed_datasets", ["storage_ref"])

    if not inspector.has_table("governed_models"):
        op.create_table(
            "governed_models",
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column("clearml_model_id", sa.String(128), nullable=False),
            sa.Column("model_name", sa.String(192), nullable=False),
            sa.Column("model_family", sa.String(64), nullable=False),
            sa.Column("semantic_version", sa.String(32), nullable=False),
            sa.Column("architecture", sa.String(128), nullable=False),
            sa.Column(
                "parent_model_id",
                sa.Uuid(as_uuid=True),
                sa.ForeignKey("governed_models.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("clearml_training_task_id", sa.String(128), nullable=False),
            sa.Column("artifact_uri", sa.String(512), nullable=False),
            sa.Column("checkpoint_sha256", sa.String(64), nullable=False),
            sa.Column("git_commit", sa.String(64), nullable=False),
            sa.Column(
                "dataset_id",
                sa.Uuid(as_uuid=True),
                sa.ForeignKey("governed_datasets.id", ondelete="RESTRICT"),
                nullable=True,
            ),
            sa.Column("legacy_dataset_identifier", sa.String(255), nullable=True),
            sa.Column("retriever_model_id", sa.String(128), nullable=True),
            sa.Column("retriever_version", sa.String(32), nullable=True),
            sa.Column("schema_version", sa.String(32), nullable=False),
            sa.Column("environment_summary", JSONVariant, nullable=False),
            sa.Column("dependencies_summary", JSONVariant, nullable=False),
            sa.Column("metrics_summary", JSONVariant, nullable=False),
            sa.Column("lifecycle_status", sa.String(32), nullable=False),
            sa.Column("was_ever_production", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("compliance_status", sa.String(32), nullable=False),
            sa.Column("drift_status", sa.String(32), nullable=False),
            sa.Column("owner", sa.String(128), nullable=False),
            sa.Column("description", sa.Text(), nullable=False, server_default=""),
            sa.Column("tags", JSONVariant, nullable=False),
            sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("clearml_model_id", name="uq_governed_models_clearml_id"),
            sa.UniqueConstraint("model_family", "semantic_version", name="uq_governed_models_family_version"),
        )
        op.create_index("ix_governed_models_family_stage", "governed_models", ["model_family", "lifecycle_status"])
        op.create_index("ix_governed_models_compliance_status", "governed_models", ["compliance_status"])

    if not inspector.has_table("governance_policies"):
        op.create_table(
            "governance_policies",
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column("policy_key", sa.String(128), nullable=False),
            sa.Column("name", sa.String(192), nullable=False),
            sa.Column("description", sa.Text(), nullable=False, server_default=""),
            sa.Column("model_family", sa.String(64), nullable=True),
            sa.Column("policy_type", sa.String(64), nullable=False),
            sa.Column("config", JSONVariant, nullable=False),
            sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("enforcement_mode", sa.String(16), nullable=False, server_default="blocking"),
            sa.Column("updated_by", sa.String(128), nullable=False, server_default="system-seed"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("policy_key", name="uq_governance_policies_key"),
        )
        op.create_index("ix_governance_policies_model_family", "governance_policies", ["model_family"])

        now = datetime.now(timezone.utc)
        policies_table = sa.table(
            "governance_policies",
            sa.column("id", sa.Uuid(as_uuid=True)),
            sa.column("policy_key", sa.String),
            sa.column("name", sa.String),
            sa.column("description", sa.Text),
            sa.column("model_family", sa.String),
            sa.column("policy_type", sa.String),
            sa.column("config", JSONVariant),
            sa.column("is_enabled", sa.Boolean),
            sa.column("enforcement_mode", sa.String),
            sa.column("updated_by", sa.String),
            sa.column("created_at", sa.DateTime(timezone=True)),
            sa.column("updated_at", sa.DateTime(timezone=True)),
        )
        op.bulk_insert(
            policies_table,
            [
                {
                    "id": uuid.uuid4(),
                    "policy_key": p["policy_key"],
                    "name": p["name"],
                    "description": p["description"],
                    "model_family": p["model_family"],
                    "policy_type": p["policy_type"],
                    "config": p["config"],
                    "is_enabled": p["is_enabled"],
                    "enforcement_mode": p["enforcement_mode"],
                    "updated_by": "alembic-seed",
                    "created_at": now,
                    "updated_at": now,
                }
                for p in DEFAULT_POLICIES
            ],
        )

    if not inspector.has_table("approval_requests"):
        op.create_table(
            "approval_requests",
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column(
                "model_id",
                sa.Uuid(as_uuid=True),
                sa.ForeignKey("governed_models.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("target_stage", sa.String(32), nullable=False),
            sa.Column("request_type", sa.String(32), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("requested_by", sa.String(128), nullable=False),
            sa.Column("reason", sa.Text(), nullable=False),
            sa.Column("automated_check_passed", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("automated_check_summary", JSONVariant, nullable=False),
            sa.Column(
                "baseline_model_id",
                sa.Uuid(as_uuid=True),
                sa.ForeignKey("governed_models.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("shadow_comparison_report", JSONVariant, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_approval_requests_model_status", "approval_requests", ["model_id", "status"])
        op.create_index("ix_approval_requests_status", "approval_requests", ["status"])

    if not inspector.has_table("approval_decisions"):
        op.create_table(
            "approval_decisions",
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column(
                "request_id",
                sa.Uuid(as_uuid=True),
                sa.ForeignKey("approval_requests.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("decision", sa.String(32), nullable=False),
            sa.Column("decided_by", sa.String(128), nullable=False),
            sa.Column("decided_role", sa.String(32), nullable=False),
            sa.Column("comment", sa.Text(), nullable=False),
            sa.Column("policy_override", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_approval_decisions_request_id", "approval_decisions", ["request_id"])

    if not inspector.has_table("drift_events"):
        op.create_table(
            "drift_events",
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column(
                "model_id",
                sa.Uuid(as_uuid=True),
                sa.ForeignKey("governed_models.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "dataset_id",
                sa.Uuid(as_uuid=True),
                sa.ForeignKey("governed_datasets.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("monitoring_run_id", sa.String(128), nullable=False),
            sa.Column("monitoring_window", sa.String(128), nullable=False),
            sa.Column("drift_status", sa.String(32), nullable=False),
            sa.Column("drift_metrics", JSONVariant, nullable=False),
            sa.Column("threshold_config", JSONVariant, nullable=False),
            sa.Column("policy_action_taken", sa.String(128), nullable=False),
            sa.Column("is_mock", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("monitoring_run_id", name="uq_drift_events_monitoring_run_id"),
        )
        op.create_index("ix_drift_events_model_detected", "drift_events", ["model_id", "detected_at"])

    if not inspector.has_table("governance_role_assignments"):
        op.create_table(
            "governance_role_assignments",
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column("actor_id", sa.String(128), nullable=False),
            sa.Column("role", sa.String(32), nullable=False),
            sa.Column("assigned_by", sa.String(128), nullable=False),
            sa.Column("notes", sa.Text(), nullable=False, server_default=""),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("actor_id", name="uq_governance_role_assignments_actor"),
        )

    if not inspector.has_table("audit_events"):
        op.create_table(
            "audit_events",
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
            sa.Column("actor_id", sa.String(128), nullable=False),
            sa.Column("actor_role", sa.String(32), nullable=False),
            sa.Column("action", sa.String(64), nullable=False),
            sa.Column("entity_type", sa.String(64), nullable=False),
            sa.Column("entity_id", sa.String(128), nullable=False),
            sa.Column("related_model_version", sa.String(192), nullable=True),
            sa.Column("related_dataset_version", sa.String(192), nullable=True),
            sa.Column("previous_state", JSONVariant, nullable=True),
            sa.Column("new_state", JSONVariant, nullable=True),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("outcome", sa.String(16), nullable=False),
            sa.Column("correlation_id", sa.String(64), nullable=False),
        )
        op.create_index("ix_audit_events_timestamp", "audit_events", ["timestamp"])
        op.create_index("ix_audit_events_actor_id", "audit_events", ["actor_id"])
        op.create_index("ix_audit_events_action", "audit_events", ["action"])
        op.create_index("ix_audit_events_entity", "audit_events", ["entity_type", "entity_id"])
        op.create_index("ix_audit_events_correlation_id", "audit_events", ["correlation_id"])

    if bind.dialect.name == "postgresql":
        op.execute(
            """
            CREATE OR REPLACE FUNCTION prevent_audit_events_mutation()
            RETURNS TRIGGER AS $$
            BEGIN
                RAISE EXCEPTION 'audit_events table is append-only; UPDATE and DELETE are prohibited';
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        op.execute("DROP TRIGGER IF EXISTS trg_audit_events_append_only ON audit_events;")
        op.execute(
            """
            CREATE TRIGGER trg_audit_events_append_only
            BEFORE UPDATE OR DELETE ON audit_events
            FOR EACH ROW EXECUTE FUNCTION prevent_audit_events_mutation();
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS trg_audit_events_append_only ON audit_events;")
        op.execute("DROP FUNCTION IF EXISTS prevent_audit_events_mutation();")

    for tbl in (
        "audit_events",
        "governance_role_assignments",
        "drift_events",
        "approval_decisions",
        "approval_requests",
        "governance_policies",
        "governed_models",
        "governed_datasets",
    ):
        op.drop_table(tbl)
