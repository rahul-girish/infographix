"""Dataset governance catalog service."""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from backend.app.governance.adapters.dataset_manifest_adapter import (
    parse_feature_report,
    parse_minidvc_manifest,
)
from backend.app.governance.auth import ActorContext
from backend.app.governance.exceptions import ConflictError, ResourceNotFoundError, ValidationError
from backend.app.governance.models import GovernedDataset, GovernedModel
from backend.app.governance.schemas import (
    DatasetCreateRequest,
    DatasetIngestManifestRequest,
    DatasetOut,
)
from backend.app.governance.services.audit_service import AuditService


def _parse_uuid(raw_id: str, label: str = "id") -> uuid.UUID:
    try:
        return uuid.UUID(str(raw_id))
    except ValueError as exc:
        raise ValidationError(f"Invalid UUID for '{label}': {raw_id}") from exc


class DatasetService:
    """Manages governed dataset records and minidvc/feature-report lineage ingestion."""

    @staticmethod
    def to_out(db: Session, dataset: GovernedDataset) -> DatasetOut:
        linked_count = (
            db.query(func.count(GovernedModel.id))
            .filter(GovernedModel.dataset_id == dataset.id)
            .scalar()
            or 0
        )
        return DatasetOut(
            id=str(dataset.id),
            dataset_name=dataset.dataset_name,
            dataset_version=dataset.dataset_version,
            source_type=dataset.source_type,
            owner=dataset.owner,
            description=dataset.description,
            storage_system=dataset.storage_system,
            storage_ref=dataset.storage_ref,
            checksum_sha256=dataset.checksum_sha256,
            data_format=dataset.data_format,
            schema_version=dataset.schema_version,
            record_count=dataset.record_count,
            size_bytes=dataset.size_bytes,
            split_config=dataset.split_config or {},
            quality_summary=dataset.quality_summary or {},
            preprocessing_version=dataset.preprocessing_version,
            usage_policy=dataset.usage_policy,
            parent_dataset_id=str(dataset.parent_dataset_id) if dataset.parent_dataset_id else None,
            is_active=dataset.is_active,
            linked_models_count=int(linked_count),
            created_at=dataset.created_at.isoformat(),
            updated_at=dataset.updated_at.isoformat(),
        )

    @staticmethod
    def register_dataset(
        db: Session,
        payload: DatasetCreateRequest,
        actor: ActorContext,
    ) -> GovernedDataset:
        existing = (
            db.query(GovernedDataset)
            .filter(
                GovernedDataset.dataset_name == payload.dataset_name,
                GovernedDataset.dataset_version == payload.dataset_version,
            )
            .first()
        )
        if existing:
            AuditService.record_isolated(
                db,
                actor_id=actor.actor_id,
                actor_role=actor.role,
                action="dataset.registered",
                entity_type="dataset",
                entity_id=f"{payload.dataset_name}:{payload.dataset_version}",
                correlation_id=actor.correlation_id,
                outcome="failed",
                reason="Duplicate dataset_name and dataset_version.",
                related_dataset_version=f"{payload.dataset_name}:{payload.dataset_version}",
            )
            raise ConflictError(
                f"Dataset '{payload.dataset_name}' version '{payload.dataset_version}' already exists."
            )

        parent_uuid: Optional[uuid.UUID] = None
        if payload.parent_dataset_id:
            parent_uuid = _parse_uuid(payload.parent_dataset_id, "parent_dataset_id")
            parent_obj = db.query(GovernedDataset).filter(GovernedDataset.id == parent_uuid).first()
            if not parent_obj:
                raise ResourceNotFoundError(
                    f"Parent dataset '{payload.parent_dataset_id}' not found."
                )

        dataset = GovernedDataset(
            dataset_name=payload.dataset_name.strip(),
            dataset_version=payload.dataset_version.strip(),
            source_type=payload.source_type,
            owner=(payload.owner or actor.actor_id).strip(),
            description=payload.description.strip(),
            storage_system=payload.storage_system,
            storage_ref=payload.storage_ref.strip(),
            checksum_sha256=payload.checksum_sha256,
            data_format=payload.data_format.strip(),
            schema_version=payload.schema_version.strip(),
            record_count=payload.record_count,
            size_bytes=payload.size_bytes,
            split_config=payload.split_config,
            quality_summary=payload.quality_summary,
            preprocessing_version=payload.preprocessing_version.strip(),
            usage_policy=payload.usage_policy.strip(),
            parent_dataset_id=parent_uuid,
        )
        db.add(dataset)
        db.flush()

        AuditService.record(
            db,
            actor_id=actor.actor_id,
            actor_role=actor.role,
            action="dataset.registered",
            entity_type="dataset",
            entity_id=str(dataset.id),
            correlation_id=actor.correlation_id,
            outcome="success",
            related_dataset_version=f"{dataset.dataset_name}:{dataset.dataset_version}",
            new_state={
                "dataset_name": dataset.dataset_name,
                "dataset_version": dataset.dataset_version,
                "storage_system": dataset.storage_system,
                "storage_ref": dataset.storage_ref,
                "checksum_sha256": dataset.checksum_sha256,
            },
        )
        db.commit()
        db.refresh(dataset)
        return dataset

    @staticmethod
    def ingest_from_manifest(
        db: Session,
        payload: DatasetIngestManifestRequest,
        actor: ActorContext,
    ) -> GovernedDataset:
        manifest = parse_minidvc_manifest(payload.manifest_path)
        report = parse_feature_report(payload.feature_report_path)

        quality = report.get("quality_summary", {})
        record_count = int(
            quality.get("valid_elements")
            or quality.get("retained_elements")
            or quality.get("total_posters")
            or manifest.get("file_count", 0)
        )
        version = payload.dataset_version or manifest["manifest_version"]
        pipeline_ver = report.get("pipeline_version", "2.0.0")

        create_req = DatasetCreateRequest(
            dataset_name=payload.dataset_name,
            dataset_version=version,
            source_type=payload.source_type,
            owner=actor.actor_id,
            description=payload.description
            or f"Ingested from minidvc manifest '{manifest['manifest_version']}' ({manifest['file_count']} files).",
            storage_system="minidvc",
            storage_ref=f"minidvc:{manifest['manifest_version']}",
            checksum_sha256=manifest["manifest_sha256"],
            data_format="parquet",
            schema_version=pipeline_ver,
            record_count=record_count,
            size_bytes=manifest["total_size_bytes"],
            split_config=report.get("orientation_distribution", {}),
            quality_summary=quality,
            preprocessing_version=pipeline_ver,
            usage_policy=payload.usage_policy,
        )
        return DatasetService.register_dataset(db, create_req, actor)

    @staticmethod
    def list_datasets(
        db: Session,
        *,
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[GovernedDataset]:
        q = db.query(GovernedDataset)
        if active_only:
            q = q.filter(GovernedDataset.is_active.is_(True))
        return (
            q.order_by(desc(GovernedDataset.created_at))
            .offset(max(0, offset))
            .limit(min(200, max(1, limit)))
            .all()
        )

    @staticmethod
    def get_dataset(db: Session, dataset_id: str) -> GovernedDataset:
        uid = _parse_uuid(dataset_id, "dataset_id")
        dataset = db.query(GovernedDataset).filter(GovernedDataset.id == uid).first()
        if not dataset:
            raise ResourceNotFoundError(f"Governed dataset '{dataset_id}' not found.")
        return dataset
