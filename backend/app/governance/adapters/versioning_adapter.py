"""Mock/interface adapter for Rahul S's Model Versioning module (out of scope)."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol


class VersioningIntegrationProtocol(Protocol):
    """Contract expected from the external Model Versioning module."""

    def build_sample_registration_payload(
        self,
        *,
        semantic_version: str,
        dataset_id: str,
        model_family: str = "poster_layout_generator",
    ) -> dict[str, Any]: ...


class MockVersioningAdapter:
    """Provides clearly labeled mock registration payloads for tests and teammate demos."""

    @staticmethod
    def build_sample_registration_payload(
        *,
        semantic_version: str,
        dataset_id: str,
        model_family: str = "poster_layout_generator",
    ) -> dict[str, Any]:
        sha = hashlib.sha256(f"mock:{model_family}:{semantic_version}".encode("utf-8")).hexdigest()
        return {
            "clearml_model_id": f"mock-clearml-{model_family}-{semantic_version}",
            "model_family": model_family,
            "semantic_version": semantic_version,
            "clearml_training_task_id": f"mock-task-{semantic_version}",
            "artifact_uri": f"https://mock.clear.ml/models/{model_family}-v{semantic_version}.zip",
            "checkpoint_sha256": sha,
            "git_commit": "0000000mock0000000",
            "dataset_id": dataset_id,
            "environment_summary": {
                "python_version": "3.12.3",
                "torch_version": "2.5.0",
                "clearml_version": "2.2.0",
                "is_mock": True,
            },
            "dependencies_summary": (
                {}
                if model_family == "poster_layout_generator"
                else {
                    "n_train_posters": 5498,
                    "train_poster_ids_sha256": sha,
                    "train_only_verified": True,
                }
            ),
            "metrics_summary": (
                {
                    "val_iou_mean": 0.31,
                    "val_loss_total": 3.45,
                    "val_canvas_violation_rate": 0.01,
                    "val_unintended_overlap_mean_area": 0.04,
                }
                if model_family == "poster_layout_generator"
                else {
                    "ndcg_at_5": 0.78,
                }
            ),
            "description": f"[MOCK] Sample {model_family} v{semantic_version} from versioning adapter",
            "tags": ["mock-versioning"],
        }
