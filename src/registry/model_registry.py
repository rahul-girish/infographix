"""Model Registry implementation for Infographix.

Provides versioning, lifecycle stage management (staging, production, archived),
metadata manifests, and loading utilities for production inference.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


STAGES = ("development", "staging", "production", "archived")


@dataclass
class ModelManifest:
    """Metadata identity card for a registered model artifact."""
    model_name: str
    version: str
    stage: str
    algorithm: str
    created_at: str
    git_commit: str | None = None
    data_version_hash: str | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_schema_columns: list[str] = field(default_factory=list)
    description: str = ""
    artifact_files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelManifest:
        return cls(**data)


class ModelRegistry:
    """File-based, version-controlled Model Registry.

    Directory structure:
        <registry_root>/
            <model_name>/
                stage_pointers.json
                v1.0.0/
                    pipeline.joblib
                    manifest.json
                    feature_schema.json
                v2.0.0/
                    ...
    """

    def __init__(self, registry_root: str | Path | None = None):
        if registry_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.registry_root = project_root / "models" / "registry"
        else:
            self.registry_root = Path(registry_root)
        self.registry_root.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_name: str) -> Path:
        m_dir = self.registry_root / model_name
        m_dir.mkdir(parents=True, exist_ok=True)
        return m_dir

    def _stage_pointers_path(self, model_name: str) -> Path:
        return self._model_dir(model_name) / "stage_pointers.json"

    def _load_stage_pointers(self, model_name: str) -> dict[str, str | None]:
        p = self._stage_pointers_path(model_name)
        if not p.exists():
            return {"development": None, "staging": None, "production": None, "archived": None}
        return json.loads(p.read_text(encoding="utf-8"))

    def _save_stage_pointers(self, model_name: str, pointers: dict[str, str | None]) -> None:
        p = self._stage_pointers_path(model_name)
        p.write_text(json.dumps(pointers, indent=2), encoding="utf-8")

    def register_model(
        self,
        model_name: str,
        version: str,
        model_object: Any,
        preprocessor: Any | None = None,
        feature_schema_columns: list[str] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        stage: str = "staging",
        description: str = "",
        git_commit: str | None = None,
        data_version_hash: str | None = None,
    ) -> ModelManifest:
        """Register a new immutable model version with its artifacts and manifest."""
        if stage not in STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {STAGES}")

        # Normalize version string (e.g. '1.0.0' -> 'v1.0.0')
        if not version.startswith("v"):
            version = f"v{version}"

        version_dir = self._model_dir(model_name) / version
        if version_dir.exists():
            raise FileExistsError(
                f"Model '{model_name}' version '{version}' is already registered and immutable at {version_dir}."
            )
        version_dir.mkdir(parents=True, exist_ok=False)

        # 1. Save serialized model + preprocessor bundle
        bundle_path = version_dir / "pipeline.joblib"
        bundle_payload = {
            "model": model_object,
            "preprocessor": preprocessor,
            "feature_columns": feature_schema_columns or [],
        }
        joblib.dump(bundle_payload, bundle_path)

        # 2. Save feature schema JSON
        schema_path = version_dir / "feature_schema.json"
        schema_path.write_text(
            json.dumps({"feature_columns": feature_schema_columns or []}, indent=2),
            encoding="utf-8",
        )

        # 3. Create manifest
        manifest = ModelManifest(
            model_name=model_name,
            version=version,
            stage=stage,
            algorithm=model_object.__class__.__name__,
            created_at=datetime.now(timezone.utc).isoformat(),
            git_commit=git_commit,
            data_version_hash=data_version_hash,
            hyperparameters=hyperparameters or {},
            metrics=metrics or {},
            feature_schema_columns=feature_schema_columns or [],
            description=description,
            artifact_files={
                "pipeline": "pipeline.joblib",
                "schema": "feature_schema.json",
                "manifest": "manifest.json",
            },
        )

        manifest_path = version_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        # 4. Update stage pointers
        pointers = self._load_stage_pointers(model_name)
        pointers[stage] = version
        self._save_stage_pointers(model_name, pointers)

        return manifest

    def promote_model(self, model_name: str, version: str, target_stage: str) -> ModelManifest:
        """Promote a registered model version to a target stage (e.g. 'production')."""
        if target_stage not in STAGES:
            raise ValueError(f"Invalid target_stage '{target_stage}'. Must be one of {STAGES}")

        if not version.startswith("v"):
            version = f"v{version}"

        version_dir = self._model_dir(model_name) / version
        manifest_path = version_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Model '{model_name}' version '{version}' not found in registry.")

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_data["stage"] = target_stage
        manifest = ModelManifest.from_dict(manifest_data)
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        # Update pointers
        pointers = self._load_stage_pointers(model_name)
        pointers[target_stage] = version
        self._save_stage_pointers(model_name, pointers)

        return manifest

    def load_model(
        self,
        model_name: str,
        version_or_stage: str = "production",
    ) -> tuple[Any, Any, ModelManifest]:
        """Load the model object, preprocessor, and manifest by version or stage name."""
        target_version = version_or_stage
        if version_or_stage in STAGES:
            pointers = self._load_stage_pointers(model_name)
            target_version = pointers.get(version_or_stage)
            if target_version is None:
                raise ValueError(f"No model currently assigned to stage '{version_or_stage}' for '{model_name}'.")

        if not target_version.startswith("v"):
            target_version = f"v{target_version}"

        version_dir = self._model_dir(model_name) / target_version
        bundle_path = version_dir / "pipeline.joblib"
        manifest_path = version_dir / "manifest.json"

        if not bundle_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Registry entry missing for '{model_name}' ({target_version}) at {version_dir}.")

        bundle = joblib.load(bundle_path)
        manifest = ModelManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))

        return bundle["model"], bundle.get("preprocessor"), manifest

    def load_production_model(self, model_name: str) -> tuple[Any, Any, ModelManifest]:
        """Convenience helper for production APIs (e.g. FastAPI)."""
        return self.load_model(model_name, version_or_stage="production")

    def list_models(self) -> dict[str, Any]:
        """Return a structured overview of all models, versions, and stage pointers."""
        result = {}
        for m_dir in self.registry_root.iterdir():
            if not m_dir.is_dir():
                continue
            name = m_dir.name
            pointers = self._load_stage_pointers(name)
            versions = [
                d.name for d in m_dir.iterdir()
                if d.is_dir() and (d / "manifest.json").exists()
            ]
            result[name] = {
                "active_stages": pointers,
                "versions": sorted(versions),
            }
        return result
