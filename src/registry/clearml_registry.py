"""ClearML Model Registry integration for Infographix.

Provides automated model versioning, artifact upload, hyperparameter & metric
tracking, and production model retrieval using ClearML, with seamless offline
fallback to the local file-based registry.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.model_registry import ModelManifest, ModelRegistry


def is_clearml_configured() -> bool:
    """Check if ClearML has API credentials configured (config file or env vars)."""
    has_env = bool(os.getenv("CLEARML_API_ACCESS_KEY") and os.getenv("CLEARML_API_SECRET_KEY"))
    conf_path = Path.home() / "clearml.conf"
    return has_env or conf_path.exists()


class ClearMLModelRegistry:
    """Manages model artifacts, versioning, and stage promotion via ClearML.

    Includes automatic offline/local fallback so pipelines and unit tests never fail
    when running without internet or credentials.
    """

    DEFAULT_PROJECT = "Infographix"

    def __init__(
        self,
        project_name: str = DEFAULT_PROJECT,
        offline: bool | None = None,
        local_fallback_root: str | Path | None = None,
    ):
        self.project_name = project_name
        self.local_registry = ModelRegistry(registry_root=local_fallback_root)

        # Determine online vs offline mode
        if offline is not None:
            self.offline = offline
        else:
            self.offline = not is_clearml_configured()

        try:
            from clearml import Task
            if self.offline:
                Task.set_offline(True)
            self._clearml_available = True
        except ImportError:
            self._clearml_available = False

    def register_model(
        self,
        model_name: str,
        version: str,
        model_object: Any,
        preprocessor: Any | None = None,
        feature_schema_columns: list[str] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        stage: str = "production",
        description: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Register, version, and publish a model pipeline to ClearML Model Registry.

        Also mirrors to the local registry for dual offline/online reliability.
        """
        if not version.startswith("v"):
            version = f"v{version}"

        # 1. First mirror to local registry to guarantee a local copy exists
        local_manifest = self.local_registry.register_model(
            model_name=model_name,
            version=version,
            model_object=model_object,
            preprocessor=preprocessor,
            feature_schema_columns=feature_schema_columns,
            hyperparameters=hyperparameters,
            metrics=metrics,
            stage=stage,
            description=description,
        )

        pipeline_file = self.local_registry._model_dir(model_name) / version / "pipeline.joblib"

        result = {
            "model_name": model_name,
            "version": version,
            "stage": stage,
            "local_path": str(pipeline_file),
            "clearml_id": None,
            "clearml_mode": "offline" if self.offline else "online",
        }

        # 2. Upload and publish via ClearML Model Registry
        if self._clearml_available:
            try:
                from clearml import OutputModel, Task

                task_name = f"register-{model_name}-{version}"
                # Get existing task or create a dedicated registration task
                task = Task.init(
                    project_name=self.project_name,
                    task_name=task_name,
                    task_type=Task.TaskTypes.training,
                    reuse_last_task_id=False,
                )

                if hyperparameters:
                    task.connect(hyperparameters, name="hyperparameters")

                if metrics:
                    logger = task.get_logger()
                    for m_name, m_val in metrics.items():
                        if isinstance(m_val, (int, float)):
                            logger.report_scalar(
                                title="Metrics",
                                series=m_name,
                                iteration=1,
                                value=float(m_val),
                            )

                # Create ClearML OutputModel in the Model Registry
                output_model = OutputModel(
                    task=task,
                    name=model_name,
                    framework="Scikit-Learn",
                )
                output_model.update_weights(weights_filename=str(pipeline_file))
                output_model.comment = description or f"{model_name} {version}"

                # Add stage and version tags
                all_tags = [version, stage]
                if tags:
                    all_tags.extend(tags)
                output_model.add_tags(list(set(all_tags)))

                # Publish model (makes it immutable in ClearML)
                if stage == "production":
                    output_model.publish()

                task.flush()
                task.close()

                result["clearml_id"] = output_model.id
                result["clearml_url"] = output_model.url
            except Exception as e:
                # Log warning and keep local manifest if ClearML network/server is unreachable
                result["clearml_error"] = str(e)

        return result

    def load_production_model(self, model_name: str) -> tuple[Any, Any, dict[str, Any]]:
        """Fetch the active production model from ClearML, falling back to local registry."""
        loaded_from_clearml = False

        if self._clearml_available and not self.offline:
            try:
                from clearml import Model
                models = Model.query_models(
                    project_name=self.project_name,
                    model_name=model_name,
                    tags=["production"],
                    only_published=True,
                )
                if models:
                    best_model = models[0]
                    local_weights = best_model.get_local_copy()
                    bundle = joblib.load(local_weights)
                    meta = {
                        "source": "clearml",
                        "model_id": best_model.id,
                        "name": best_model.name,
                        "tags": best_model.tags,
                        "comment": best_model.comment,
                    }
                    return bundle["model"], bundle.get("preprocessor"), meta
            except Exception:
                pass

        # Fallback to local registry
        model, prep, local_manifest = self.local_registry.load_production_model(model_name)
        meta = {
            "source": "local_registry",
            "version": local_manifest.version,
            "stage": local_manifest.stage,
            "metrics": local_manifest.metrics,
            "hyperparameters": local_manifest.hyperparameters,
        }
        return model, prep, meta

    def list_models(self) -> dict[str, Any]:
        """List models available in ClearML with local registry fallback."""
        if self._clearml_available and not self.offline:
            try:
                from clearml import Model
                models = Model.query_models(project_name=self.project_name)
                return {
                    m.name: {
                        "id": m.id,
                        "tags": m.tags,
                        "published": m.published,
                        "created": str(m.created),
                    }
                    for m in models
                }
            except Exception:
                pass

        return self.local_registry.list_models()
