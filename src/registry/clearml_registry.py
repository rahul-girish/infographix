"""ClearML Native Model Registry Interface for Infographix.

Provides standardized, immutable model registration, artifact bundling with SHA-256
verification, lineage tracking, querying, and reproducible model reconstruction
using ClearML's native OutputModel, InputModel, and Model.query_models APIs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import clearml
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from clearml import InputModel, Model, OutputModel, Task
from clearml.backend_api.session.client import APIClient

from src.models.layout_retrieval import (
    POSTER_COMPOSITION_COLS,
    SHARED_CANVAS_COLS,
    SHARED_CAPACITY_COLS,
    SHARED_INVENTORY_COLS,
    SHARED_STRUCTURE_COLS,
    LayoutRetriever,
)
from src.neural_layout.dataset import (
    CONTENT_FEAT_DIM,
    KNOWN_CATEGORIES,
    TEMPLATE_FEAT_DIM,
)
from src.neural_layout.layout_model import TemplateConditionedLayoutModel


DEFAULT_PROJECT_NAME = "Infographix"
DEFAULT_MODEL_FAMILY = "poster_layout_generator"
RETRIEVER_MODEL_FAMILY = "poster_layout_retriever"
LIFECYCLE_STAGES = ("candidate", "staging", "production", "archived")
SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
FEATURE_SCHEMA_VERSION = "v1.0.0"
RETRIEVER_SCHEMA_VERSION = "v2.0.0"


def normalize_version(version: str) -> str:
    """Normalize and validate semantic version string (e.g. 'v0.1.0' -> '0.1.0')."""
    clean = version.strip()
    if clean.startswith("v"):
        clean = clean[1:]
    if not SEMVER_PATTERN.match(clean):
        raise ValueError(
            f"Invalid semantic version '{version}'. Expected 'X.Y.Z' (e.g. '0.1.0')."
        )
    return clean


def build_model_name(model_family: str, semantic_version: str) -> str:
    """Construct canonical versioned model name (e.g. 'poster_layout_generator-v0.1.0')."""
    ver = normalize_version(semantic_version)
    return f"{model_family}-v{ver}"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Return current git HEAD commit hash or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def get_feature_schema() -> dict[str, Any]:
    """Return canonical input feature and inference output schemas for poster_layout_generator."""
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "categories": list(KNOWN_CATEGORIES),
        "content_feature_dim": CONTENT_FEAT_DIM,
        "content_features": [
            {"indices": "0..8", "name": "category_one_hot", "description": "9-dim one-hot semantic category"},
            {"index": 9, "name": "log1p_word_count", "description": "log(1 + word_count) if has_text_length else 0.0"},
            {"index": 10, "name": "log1p_char_count", "description": "log(1 + char_count) if has_text_length else 0.0"},
            {"index": 11, "name": "has_text_length", "description": "1.0 if extracted text length available else 0.0"},
            {"index": 12, "name": "aspect_ratio", "description": "intrinsic width/height if has_aspect_ratio else 1.0"},
            {"index": 13, "name": "has_aspect_ratio", "description": "1.0 if intrinsic asset dimension available else 0.0"},
            {"index": 14, "name": "table_rows_scaled", "description": "table_rows / 10.0"},
            {"index": 15, "name": "table_cols_scaled", "description": "table_cols / 10.0"},
            {"index": 16, "name": "priority", "description": "element importance score in [0.1, 1.0]"},
            {"index": 17, "name": "shuffled_order_ratio", "description": "decoupled input index / total_items"},
        ],
        "template_feature_dim": TEMPLATE_FEAT_DIM,
        "template_features": [
            {"indices": "0..8", "name": "category_one_hot", "description": "9-dim one-hot semantic category"},
            {"indices": "9..12", "name": "bbox_norm", "description": "normalized [x0, y0, w, h] in [0, 1]"},
            {"indices": "13..14", "name": "center", "description": "normalized center [cx, cy] in [0, 1]"},
            {"index": 15, "name": "area_norm", "description": "normalized slot area in [0, 1]"},
        ],
        "inference_output_schema": {
            "pred_boxes": {"shape": "[B, M, 4]", "range": "(0, 1)", "format": "[x0, y0, w, h]"},
            "pred_column_logits": {"shape": "[B, M, num_columns]", "description": "column assignment logits"},
            "pred_order_scores": {"shape": "[B, M]", "description": "continuous reading-order ranking score"},
            "pred_macro_column_logits": {"shape": "[B, num_columns]", "description": "poster column count logits"},
        },
    }


def get_preprocessing_config() -> dict[str, Any]:
    """Return preprocessing and anti-leakage specification required at inference time."""
    return {
        "anti_leakage_mode": True,
        "aspect_ratio_clip_bounds": [0.25, 4.0],
        "missing_text_word_count": 0.0,
        "missing_text_char_count": 0.0,
        "missing_aspect_ratio_default": 1.0,
        "table_default_rows": 4,
        "table_default_cols": 3,
        "coordinate_normalization": "unit_canvas_[0,1]",
    }


def get_environment_info() -> dict[str, str]:
    """Collect non-sensitive dependency and runtime version information."""
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "clearml_version": clearml.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "git_commit": get_git_commit(),
    }


def extract_model_config_from_checkpoint(checkpoint_data: dict[str, Any]) -> dict[str, Any]:
    """Extract TemplateConditionedLayoutModel instantiation config from checkpoint."""
    explicit_mcfg = checkpoint_data.get("model_config")
    if isinstance(explicit_mcfg, dict) and "d_model" in explicit_mcfg:
        return dict(explicit_mcfg)

    cfg = checkpoint_data.get("config")
    state_dict = checkpoint_data.get("model_state_dict", {})

    # Determine num_columns safely from decoder head weight shape if available
    num_columns = 5
    if "decoder.column_head.3.weight" in state_dict:
        num_columns = int(state_dict["decoder.column_head.3.weight"].shape[0])

    # State dict is authoritative for whether anchor weights exist
    use_anchor = "decoder.slot_q.weight" in state_dict
    bbox_param = str(
        getattr(
            cfg,
            "bbox_parameterization",
            "bounded" if use_anchor else "independent",
        )
    )

    if cfg is not None and hasattr(cfg, "d_model"):
        return {
            "architecture": "TemplateConditionedLayoutModel",
            "content_dim": CONTENT_FEAT_DIM,
            "template_dim": TEMPLATE_FEAT_DIM,
            "d_model": int(cfg.d_model),
            "num_heads": int(cfg.num_heads),
            "encoder_layers": int(cfg.encoder_layers),
            "decoder_layers": int(cfg.decoder_layers),
            "ffn_dim": int(cfg.ffn_dim),
            "num_columns": num_columns,
            "dropout": float(cfg.dropout),
            "bbox_parameterization": bbox_param,
            "use_anchor_conditioning": use_anchor,
            "min_box_width": float(getattr(cfg, "min_box_width", 0.01)),
            "min_box_height": float(getattr(cfg, "min_box_height", 0.005)),
        }

    raise ValueError(
        "Checkpoint does not contain a valid TrainingConfig; pass 'model_config' explicitly."
    )


def extract_training_config_from_checkpoint(checkpoint_data: dict[str, Any]) -> dict[str, Any]:
    """Extract training hyperparameters from checkpoint safely."""
    cfg = checkpoint_data.get("config")
    if cfg is None:
        return {}
    loss_w = getattr(cfg, "loss_weights", None)
    loss_weights_dict = (
        {
            "w_box": float(loss_w.w_box),
            "w_size": float(getattr(loss_w, "w_size", 0.0)),
            "w_giou": float(loss_w.w_giou),
            "w_bound": float(loss_w.w_bound),
            "w_overlap": float(loss_w.w_overlap),
            "w_col": float(loss_w.w_col),
            "w_aspect": float(loss_w.w_aspect),
            "w_order": float(loss_w.w_order),
            "w_macro": float(loss_w.w_macro),
        }
        if loss_w is not None
        else {}
    )
    rinfo = checkpoint_data.get("retriever_info") or {}
    return {
        "seed": int(getattr(cfg, "seed", 42)),
        "batch_size": int(getattr(cfg, "batch_size", 8)),
        "learning_rate": float(getattr(cfg, "learning_rate", 3e-4)),
        "weight_decay": float(getattr(cfg, "weight_decay", 1e-4)),
        "num_epochs": int(getattr(cfg, "num_epochs", 1)),
        "grad_clip_norm": float(getattr(cfg, "grad_clip_norm", 1.0)),
        "optimizer": "AdamW",
        "scheduler": f"CosineAnnealingLR(T_max={getattr(cfg, 'num_epochs', 1)}, eta_min=1e-5)",
        "loss_weights": loss_weights_dict,
        "execution_env": str(getattr(cfg, "execution_env", "unknown")),
        "checkpoint_epoch": int(checkpoint_data.get("epoch", -1)),
        "retriever_model_name": str(
            rinfo.get("model_name") or getattr(cfg, "retriever_model_name", RETRIEVER_MODEL_FAMILY)
        ),
        "retriever_model_id": rinfo.get("model_id") or getattr(cfg, "retriever_model_id", None),
        "retriever_version": rinfo.get("version") or getattr(cfg, "retriever_version", None),
    }


def verify_checkpoint_provenance(
    checkpoint_data: dict[str, Any],
    clearml_task: Task,
) -> dict[str, Any]:
    """Verify that a checkpoint genuinely originates from the specified ClearML training task.

    Checks:
      1. Direct clearml_task_id inside checkpoint (if saved by updated trainer).
      2. Matching task name and hyperparameters in ClearML 'Experiment Configuration'.
      3. Exact validation loss match at checkpoint epoch+1 against ClearML reported scalars.
    """
    if "model_state_dict" not in checkpoint_data:
        raise ValueError("Checkpoint is missing 'model_state_dict'.")

    explicit_task_id = checkpoint_data.get("clearml_task_id")
    if explicit_task_id is not None and explicit_task_id != clearml_task.id:
        raise ValueError(
            f"Provenance mismatch: checkpoint records clearml_task_id='{explicit_task_id}', "
            f"but registration targeted task '{clearml_task.id}'."
        )

    cfg = checkpoint_data.get("config")
    task_params = clearml_task.get_parameters_as_dict().get("Experiment Configuration", {})
    exp_training = task_params.get("training", {})
    exp_model = task_params.get("model", {})

    if cfg is not None:
        if getattr(cfg, "task_name", None) and cfg.task_name != clearml_task.name:
            raise ValueError(
                f"Provenance mismatch: checkpoint task_name='{cfg.task_name}' "
                f"does not match ClearML task name '{clearml_task.name}'."
            )
        if exp_training:
            if str(cfg.batch_size) != str(exp_training.get("batch_size")) or str(cfg.seed) != str(exp_training.get("seed")):
                raise ValueError(
                    "Provenance mismatch: checkpoint hyperparameters do not match ClearML task configuration."
                )
        if exp_model:
            if str(cfg.d_model) != str(exp_model.get("d_model")):
                raise ValueError(
                    "Provenance mismatch: checkpoint model d_model does not match ClearML task configuration."
                )

    # Verify reported scalar match at checkpoint epoch + 1
    ckpt_epoch = checkpoint_data.get("epoch")
    ckpt_metrics = checkpoint_data.get("metrics", {})
    if ckpt_epoch is not None and "val_loss_total" in ckpt_metrics:
        scalars = clearml_task.get_reported_scalars()
        val_loss_series = scalars.get("Loss - Total", {}).get("val", {})
        xs = val_loss_series.get("x", [])
        ys = val_loss_series.get("y", [])
        expected_iter = int(ckpt_epoch) + 1
        if expected_iter in xs:
            idx = xs.index(expected_iter)
            reported_val_loss = float(ys[idx])
            ckpt_val_loss = float(ckpt_metrics["val_loss_total"])
            if abs(reported_val_loss - ckpt_val_loss) > 1e-4:
                raise ValueError(
                    f"Provenance mismatch: checkpoint epoch {expected_iter} val_loss_total="
                    f"{ckpt_val_loss:.6f} differs from ClearML reported {reported_val_loss:.6f}."
                )

    return {
        "verified": True,
        "task_id": clearml_task.id,
        "task_name": clearml_task.name,
        "checkpoint_epoch": int(ckpt_epoch) if ckpt_epoch is not None else None,
        "provenance_method": "explicit_id" if explicit_task_id else "config_and_metric_match",
    }


def build_model_bundle(
    bundle_dir: Path,
    checkpoint_data: dict[str, Any],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    training_task_id: str,
    dataset_identifier: str,
    semantic_version: str,
    validation_metrics: dict[str, Any],
    lifecycle_status: str = "candidate",
    model_family: str = DEFAULT_MODEL_FAMILY,
    parent_model_id: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Construct an immutable, self-contained model bundle directory with manifest and SHA-256 checksums.

    Excludes optimizer and scheduler states—only inference weights and reconstruction metadata are bundled.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    ver = normalize_version(semantic_version)
    full_name = build_model_name(model_family, ver)

    # 1. Save inference-only model weights (strip optimizer/scheduler states)
    weights_path = bundle_dir / "model_weights.pt"
    torch.save(checkpoint_data["model_state_dict"], weights_path)

    # 2. Save model architecture configuration
    model_cfg_path = bundle_dir / "model_config.json"
    model_cfg_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    # 3. Save feature schema
    feature_schema = get_feature_schema()
    schema_path = bundle_dir / "feature_schema.json"
    schema_path.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")

    # 4. Save preprocessing configuration
    prep_cfg = get_preprocessing_config()
    prep_path = bundle_dir / "preprocessing_config.json"
    prep_path.write_text(json.dumps(prep_cfg, indent=2), encoding="utf-8")

    # 5. Save dependency/environment metadata
    env_info = get_environment_info()
    if git_commit:
        env_info["git_commit"] = git_commit
    env_path = bundle_dir / "environment.json"
    env_path.write_text(json.dumps(env_info, indent=2), encoding="utf-8")

    # 6. Compute SHA-256 checksums across all bundle artifacts
    bundle_sha256s = {
        "model_weights.pt": compute_sha256(weights_path),
        "model_config.json": compute_sha256(model_cfg_path),
        "feature_schema.json": compute_sha256(schema_path),
        "preprocessing_config.json": compute_sha256(prep_path),
        "environment.json": compute_sha256(env_path),
    }

    retriever_provenance = {
        "model_name": training_config.get("retriever_model_name", RETRIEVER_MODEL_FAMILY),
        "model_id": training_config.get("retriever_model_id"),
        "version": training_config.get("retriever_version"),
    }

    # 7. Construct manifest.json
    manifest = {
        "project_name": "infographix",
        "model_family": model_family,
        "semantic_version": ver,
        "full_model_name": full_name,
        "clearml_training_task_id": training_task_id,
        "dataset_identifier": dataset_identifier,
        "git_commit": env_info["git_commit"],
        "checkpoint_sha256": bundle_sha256s["model_weights.pt"],
        "bundle_sha256s": bundle_sha256s,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lifecycle_status": lifecycle_status,
        "parent_model_id": parent_model_id,
        "retriever": retriever_provenance,
        "model_config": model_config,
        "training_config": training_config,
        "environment": env_info,
        "metrics": {
            k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
            for k, v in validation_metrics.items()
        },
    }

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_bundle_checksums(bundle_dir: Path) -> dict[str, Any]:
    """Validate bundle completeness and SHA-256 checksum integrity before loading weights."""
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Corrupted artifact bundle: missing manifest.json in {bundle_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest.get("bundle_sha256s", {})

    if not expected_hashes:
        # Backward fallback check on checkpoint_sha256
        expected_ckpt_hash = manifest.get("checkpoint_sha256")
        if not expected_ckpt_hash:
            raise ValueError("Manifest contains no SHA-256 checksums for verification.")
        expected_hashes = {"model_weights.pt": expected_ckpt_hash}

    for rel_file, expected_hash in expected_hashes.items():
        fpath = bundle_dir / rel_file
        if not fpath.exists():
            raise FileNotFoundError(f"Corrupted artifact bundle: missing required file '{rel_file}'.")
        actual_hash = compute_sha256(fpath)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Artifact checksum verification failed for '{rel_file}': "
                f"expected {expected_hash}, got {actual_hash}."
            )

    return manifest


def _resolve_upload_uri(task: Task) -> str:
    """Resolve remote ClearML file server URI so uploads are shared across machines."""
    for candidate in [getattr(task, "output_uri", None), getattr(task, "storage_uri", None)]:
        if candidate and str(candidate).strip():
            return str(candidate)
    files_server = task._get_default_session().get_files_server_host()
    if not files_server:
        raise RuntimeError("Could not resolve ClearML files server upload URI.")
    return str(files_server)


def register_model(
    checkpoint_path: str | Path,
    training_task_id: str,
    semantic_version: str,
    dataset_identifier: str = "SciPostLayout v2.0.0",
    model_config: dict[str, Any] | None = None,
    validation_metrics: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    lifecycle_status: str = "candidate",
    parent_model_id: str | None = None,
    project_name: str = DEFAULT_PROJECT_NAME,
    model_family: str = DEFAULT_MODEL_FAMILY,
    comment: str = "",
    verify_provenance: bool = True,
) -> str:
    """Register a versioned model bundle in ClearML's native Model Registry.

    Validates checkpoint & provenance, creates an immutable self-contained bundle with
    SHA-256 checksums, uploads synchronously to ClearML Cloud, writes tags & metadata,
    verifies registration, and returns the ClearML model ID.
    """
    if lifecycle_status not in LIFECYCLE_STAGES:
        raise ValueError(f"Invalid lifecycle_status '{lifecycle_status}'. Must be one of {LIFECYCLE_STAGES}")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_file}")

    ver = normalize_version(semantic_version)
    full_model_name = build_model_name(model_family, ver)

    # 1. Prevent silent overwrite of existing registered versions
    existing = Model.query_models(
        project_name=project_name,
        model_name=full_model_name,
        include_archived=True,
    )
    exact_matches = [m for m in existing if m.name == full_model_name]
    if exact_matches:
        raise ValueError(
            f"Model '{full_model_name}' already exists in ClearML project '{project_name}' "
            f"(ID: {exact_matches[0].id}). Registered model versions are immutable."
        )

    # 2. Load checkpoint and verify provenance against originating ClearML task
    checkpoint_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    orig_task = Task.get_task(task_id=training_task_id)
    if orig_task is None:
        raise ValueError(f"Originating ClearML training task '{training_task_id}' not found.")

    if verify_provenance:
        verify_checkpoint_provenance(checkpoint_data, orig_task)

    resolved_model_cfg = model_config or extract_model_config_from_checkpoint(checkpoint_data)
    resolved_train_cfg = training_config or extract_training_config_from_checkpoint(checkpoint_data)
    resolved_metrics = validation_metrics if validation_metrics is not None else checkpoint_data.get("metrics", {})

    # Determine git commit from task configuration if available, else local git HEAD
    task_env = orig_task.get_parameters_as_dict().get("Experiment Configuration", {}).get("environment", {})
    git_commit = task_env.get("git_commit") or get_git_commit()

    # 3. Build bundle in temporary directory & upload synchronously
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_dir = Path(tmp_dir) / full_model_name
        manifest = build_model_bundle(
            bundle_dir=bundle_dir,
            checkpoint_data=checkpoint_data,
            model_config=resolved_model_cfg,
            training_config=resolved_train_cfg,
            training_task_id=training_task_id,
            dataset_identifier=dataset_identifier,
            semantic_version=ver,
            validation_metrics=resolved_metrics,
            lifecycle_status=lifecycle_status,
            model_family=model_family,
            parent_model_id=parent_model_id,
            git_commit=git_commit,
        )

        tags = [
            f"family:{model_family}",
            f"version:v{ver}",
            f"arch:{resolved_model_cfg.get('architecture', 'TemplateConditionedLayoutModel')}",
            f"dataset:{dataset_identifier}",
            f"stage:{lifecycle_status}",
        ]

        full_comment = (
            f"Infographix {full_model_name} | Stage: {lifecycle_status} | "
            f"Training Task: {training_task_id} | Dataset: {dataset_identifier}"
        )
        if comment:
            full_comment = f"{full_comment}\n{comment}"

        label_enum = {cat: idx for idx, cat in enumerate(KNOWN_CATEGORIES)}
        output_model = OutputModel(
            task=orig_task,
            config_dict={
                "model_config": resolved_model_cfg,
                "feature_schema": get_feature_schema(),
                "preprocessing_config": get_preprocessing_config(),
                "manifest": manifest,
            },
            label_enumeration=label_enum,
            name=full_model_name,
            tags=tags,
            comment=full_comment,
            framework="PyTorch",
            base_model_id=None,
        )

        upload_uri = _resolve_upload_uri(orig_task)
        uploaded_uri = output_model.update_weights(
            weights_filename=str(bundle_dir),
            upload_uri=upload_uri,
            target_filename=f"{full_model_name}.zip",
            auto_delete_file=False,
            async_enable=False,
            is_package=True,
        )
        output_model.wait_for_uploads()

        if not uploaded_uri or str(uploaded_uri).startswith("file://") or "failed_uploading" in str(uploaded_uri):
            raise RuntimeError(f"Model artifact upload failed or remained local: {uploaded_uri}")

        # Set structured ClearML metadata (non-sensitive)
        metadata_payload: dict[str, dict[str, str]] = {
            "model_family": {"value": model_family, "type": "str"},
            "semantic_version": {"value": ver, "type": "str"},
            "architecture": {"value": str(resolved_model_cfg.get("architecture", "TemplateConditionedLayoutModel")), "type": "str"},
            "dataset_identifier": {"value": dataset_identifier, "type": "str"},
            "lifecycle_status": {"value": lifecycle_status, "type": "str"},
            "training_task_id": {"value": training_task_id, "type": "str"},
            "git_commit": {"value": str(git_commit), "type": "str"},
            "checkpoint_sha256": {"value": manifest["checkpoint_sha256"], "type": "str"},
            "python_version": {"value": manifest["environment"]["python_version"], "type": "str"},
            "torch_version": {"value": manifest["environment"]["torch_version"], "type": "str"},
            "schema_version": {"value": FEATURE_SCHEMA_VERSION, "type": "str"},
        }
        if parent_model_id:
            metadata_payload["parent_model_id"] = {"value": parent_model_id, "type": "str"}
        ret_meta = manifest.get("retriever", {})
        if ret_meta.get("model_id"):
            metadata_payload["retriever_model_id"] = {"value": str(ret_meta["model_id"]), "type": "str"}
        if ret_meta.get("version"):
            metadata_payload["retriever_version"] = {"value": str(ret_meta["version"]), "type": "str"}

        for metric_name, metric_val in manifest["metrics"].items():
            if isinstance(metric_val, (int, float)):
                metadata_payload[f"metric.{metric_name}"] = {
                    "value": f"{metric_val:.6f}",
                    "type": "float",
                }

        ok = output_model.set_all_metadata(metadata_payload, replace=True)
        if not ok:
            raise RuntimeError(f"Failed to persist metadata on ClearML model '{output_model.id}'.")

        # Link native ClearML parent lineage if derived from an earlier version
        if parent_model_id:
            client = APIClient()
            client.models.edit(model=output_model.id, parent=parent_model_id)

        # Verify registration persisted cleanly in ClearML
        verified_model = InputModel(model_id=output_model.id)
        if not verified_model.url or "failed" in verified_model.url:
            raise RuntimeError(f"Model verification failed after registration: {verified_model.url}")

        return str(output_model.id)


def query_registered_models(
    project_name: str = DEFAULT_PROJECT_NAME,
    model_family: str = DEFAULT_MODEL_FAMILY,
    semantic_version: str | None = None,
    lifecycle_status: str | None = None,
    include_archived: bool = False,
) -> list[dict[str, Any]]:
    """Query ClearML Model Registry for Infographix models."""
    query_name = build_model_name(model_family, semantic_version) if semantic_version else None
    tags: list[str] = [f"family:{model_family}"]
    if lifecycle_status:
        if lifecycle_status not in LIFECYCLE_STAGES:
            raise ValueError(f"Invalid lifecycle_status '{lifecycle_status}'.")
        tags.append(f"stage:{lifecycle_status}")

    models = Model.query_models(
        project_name=project_name,
        model_name=query_name,
        tags=tags if not semantic_version else None,
        only_published=False,
        include_archived=include_archived,
    )

    results: list[dict[str, Any]] = []
    for m in models:
        if semantic_version and m.name != query_name:
            continue
        if lifecycle_status and f"stage:{lifecycle_status}" not in (m.tags or []):
            continue
        raw_meta = m.get_all_metadata() or {}
        meta = {
            k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
            for k, v in raw_meta.items()
        }
        results.append({
            "model_id": m.id,
            "name": m.name,
            "tags": list(m.tags or []),
            "published": bool(m.published),
            "task_id": m.task,
            "uri": m.url,
            "lifecycle_status": meta.get("lifecycle_status", "unknown"),
            "semantic_version": meta.get("semantic_version", "unknown"),
            "dataset_identifier": meta.get("dataset_identifier", "unknown"),
            "git_commit": meta.get("git_commit", "unknown"),
            "checkpoint_sha256": meta.get("checkpoint_sha256", "unknown"),
        })
    return results


def set_model_lifecycle_status(
    target_status: str,
    model_id: str | None = None,
    semantic_version: str | None = None,
    project_name: str = DEFAULT_PROJECT_NAME,
    model_family: str = DEFAULT_MODEL_FAMILY,
) -> dict[str, Any]:
    """Explicitly transition lifecycle stage tag & metadata (distinct from ClearML Published status)."""
    if target_status not in LIFECYCLE_STAGES:
        raise ValueError(f"Invalid target_status '{target_status}'. Must be one of {LIFECYCLE_STAGES}")

    resolved_id = _resolve_exact_model_id(
        model_id=model_id,
        semantic_version=semantic_version,
        project_name=project_name,
        model_family=model_family,
    )
    model_obj = InputModel(model_id=resolved_id)
    new_tags = [t for t in (model_obj.tags or []) if not t.startswith("stage:")]
    new_tags.append(f"stage:{target_status}")

    client = APIClient()
    client.models.edit(model=resolved_id, tags=new_tags)
    model_obj.set_metadata("lifecycle_status", target_status, value_type="str")
    return {
        "model_id": resolved_id,
        "lifecycle_status": target_status,
        "tags": new_tags,
    }


def _resolve_exact_model_id(
    model_id: str | None = None,
    semantic_version: str | None = None,
    project_name: str = DEFAULT_PROJECT_NAME,
    model_family: str = DEFAULT_MODEL_FAMILY,
) -> str:
    """Resolve an exact ClearML model ID without falling back to latest or local files."""
    if not model_id and not semantic_version:
        raise ValueError(
            "Explicit 'model_id' or 'semantic_version' is required. Refusing to silently load latest."
        )

    if model_id:
        return model_id

    full_name = build_model_name(model_family, str(semantic_version))
    matches = Model.query_models(
        project_name=project_name,
        model_name=full_name,
        only_published=False,
        include_archived=False,
    )
    exact = [m for m in matches if m.name == full_name]
    if not exact:
        raise LookupError(
            f"No registered model '{full_name}' found in ClearML project '{project_name}'."
        )
    if len(exact) > 1:
        raise ValueError(
            f"Ambiguous version query: found {len(exact)} models named '{full_name}': "
            f"{[m.id for m in exact]}. Specify 'model_id' directly."
        )
    return str(exact[0].id)


def load_registered_model(
    model_id: str | None = None,
    semantic_version: str | None = None,
    project_name: str = DEFAULT_PROJECT_NAME,
    model_family: str = DEFAULT_MODEL_FAMILY,
    device: str | torch.device = "cpu",
    force_download: bool = False,
    return_manifest: bool = False,
) -> TemplateConditionedLayoutModel | tuple[TemplateConditionedLayoutModel, dict[str, Any]]:
    """Retrieve a registered model from ClearML, verify SHA-256 checksums, and reconstruct PyTorch model.

    Strictly refuses silent fallbacks to local checkpoints or newest versions.
    """
    resolved_id = _resolve_exact_model_id(
        model_id=model_id,
        semantic_version=semantic_version,
        project_name=project_name,
        model_family=model_family,
    )

    input_model = InputModel(model_id=resolved_id)
    local_bundle = input_model.get_local_copy(
        extract_archive=True,
        raise_on_error=True,
        force_download=force_download,
    )
    if not local_bundle:
        raise RuntimeError(f"ClearML failed to download artifact bundle for model '{resolved_id}'.")

    bundle_dir = Path(local_bundle)
    if not bundle_dir.is_dir():
        raise RuntimeError(
            f"Expected extracted bundle directory for model '{resolved_id}', got file: {bundle_dir}"
        )

    # 1. Verify bundle completeness and SHA-256 integrity
    manifest = verify_bundle_checksums(bundle_dir)

    # 2. Load architecture configuration
    cfg_path = bundle_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing model_config.json in downloaded bundle {bundle_dir}")
    model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # 3. Load feature schema
    schema_path = bundle_dir / "feature_schema.json"
    feature_schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else {}

    # 4. Instantiate architecture
    arch = model_cfg.get("architecture", "TemplateConditionedLayoutModel")
    if arch != "TemplateConditionedLayoutModel":
        raise ValueError(f"Unsupported model architecture '{arch}' in bundle.")

    model = TemplateConditionedLayoutModel(
        content_dim=int(model_cfg.get("content_dim", CONTENT_FEAT_DIM)),
        template_dim=int(model_cfg.get("template_dim", TEMPLATE_FEAT_DIM)),
        d_model=int(model_cfg.get("d_model", 128)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        encoder_layers=int(model_cfg.get("encoder_layers", 3)),
        decoder_layers=int(model_cfg.get("decoder_layers", 3)),
        ffn_dim=int(model_cfg.get("ffn_dim", 256)),
        num_columns=int(model_cfg.get("num_columns", 5)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        bbox_parameterization=str(model_cfg.get("bbox_parameterization", "independent")),
        use_anchor_conditioning=bool(model_cfg.get("use_anchor_conditioning", False)),
        min_box_width=float(model_cfg.get("min_box_width", 0.01)),
        min_box_height=float(model_cfg.get("min_box_height", 0.005)),
    )

    # 5. Load verified weights
    weights_path = bundle_dir / "model_weights.pt"
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Attach provenance metadata for downstream inspection
    model.clearml_model_id = resolved_id
    model.registry_manifest = manifest
    model.feature_schema = feature_schema

    if return_manifest:
        return model, manifest
    return model



# ---------------------------------------------------------------------------
# LayoutRetriever ClearML Native Model Registry Integration
# ---------------------------------------------------------------------------

def compute_train_poster_ids_sha256(train_posters_df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 fingerprint of the fitted training poster IDs."""
    if "poster_id" not in train_posters_df.columns or len(train_posters_df) == 0:
        return hashlib.sha256(b"").hexdigest()
    sorted_ids = sorted(int(pid) for pid in train_posters_df["poster_id"].tolist())
    payload = ",".join(str(pid) for pid in sorted_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_fitted_retriever(retriever: Any, enforce_train_only: bool = True) -> LayoutRetriever:
    """Validate that an object is a completely fitted LayoutRetriever trained without dev/test leakage."""
    if not isinstance(retriever, LayoutRetriever):
        raise TypeError(f"Expected LayoutRetriever instance, got {type(retriever).__name__}.")

    if (
        retriever.train_posters_ is None
        or not retriever.scalers_
        or not retriever.train_group_matrices_
        or retriever.train_occ_grids_ is None
    ):
        raise ValueError("LayoutRetriever is not fitted; cannot serialize or serve an unfitted retriever.")

    if enforce_train_only and "split" in retriever.train_posters_.columns:
        splits_present = set(retriever.train_posters_["split"].astype(str).unique())
        non_train = splits_present - {"train"}
        if non_train:
            raise ValueError(
                f"Anti-leakage violation: retriever contains non-train posters from splits {sorted(non_train)}."
            )

    return retriever


def get_retriever_feature_schema() -> dict[str, Any]:
    """Return canonical feature schema metadata for LayoutRetriever."""
    return {
        "schema_version": RETRIEVER_SCHEMA_VERSION,
        "feature_groups": {
            "canvas": list(SHARED_CANVAS_COLS),
            "inventory": list(SHARED_INVENTORY_COLS),
            "capacity": list(SHARED_CAPACITY_COLS),
            "structure": list(SHARED_STRUCTURE_COLS),
            "full_composition": list(POSTER_COMPOSITION_COLS),
        },
        "occupancy_grid_cols": [f"occ_grid_r{r}_c{c}" for r in range(4) for c in range(4)],
    }


def get_retriever_preprocessing_config(retriever: LayoutRetriever) -> dict[str, Any]:
    """Return preprocessing and fitting metadata for a fitted LayoutRetriever."""
    _validate_fitted_retriever(retriever, enforce_train_only=False)
    assert retriever.train_posters_ is not None
    return {
        "imputer_strategy": "median",
        "scaler": "RobustScaler",
        "sqrt_dim_normalization": True,
        "valid_poster_filter": "known_element_count > 0",
        "group_weights": {k: float(v) for k, v in retriever.group_weights.items()},
        "mmr_lambda": float(retriever.mmr_lambda),
        "strict_orientation": bool(retriever.strict_orientation),
        "n_train_posters": int(len(retriever.train_posters_)),
        "train_poster_ids_sha256": compute_train_poster_ids_sha256(retriever.train_posters_),
    }


def build_retriever_bundle(
    bundle_dir: Path,
    retriever_model: LayoutRetriever,
    training_task_id: str,
    dataset_identifier: str,
    semantic_version: str,
    validation_metrics: dict[str, Any] | None = None,
    lifecycle_status: str = "candidate",
    model_family: str = RETRIEVER_MODEL_FAMILY,
    parent_model_id: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Construct an immutable, SHA-256 verified artifact bundle for LayoutRetriever."""
    _validate_fitted_retriever(retriever_model, enforce_train_only=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    ver = normalize_version(semantic_version)
    full_name = build_model_name(model_family, ver)

    # 1. Serialize full fitted estimator state using joblib
    weights_path = bundle_dir / "retriever.joblib"
    joblib.dump(retriever_model, weights_path, compress=3)

    # 2. Save model configuration
    prep_cfg = get_retriever_preprocessing_config(retriever_model)
    model_cfg = {
        "architecture": "LayoutRetriever",
        "serialization_format": "joblib",
        "artifact_filename": "retriever.joblib",
        "group_weights": prep_cfg["group_weights"],
        "mmr_lambda": prep_cfg["mmr_lambda"],
        "strict_orientation": prep_cfg["strict_orientation"],
        "n_train_posters": prep_cfg["n_train_posters"],
        "train_poster_ids_sha256": prep_cfg["train_poster_ids_sha256"],
    }
    model_cfg_path = bundle_dir / "model_config.json"
    model_cfg_path.write_text(json.dumps(model_cfg, indent=2), encoding="utf-8")

    # 3. Save feature schema
    feature_schema = get_retriever_feature_schema()
    schema_path = bundle_dir / "feature_schema.json"
    schema_path.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")

    # 4. Save preprocessing config
    prep_path = bundle_dir / "preprocessing_config.json"
    prep_path.write_text(json.dumps(prep_cfg, indent=2), encoding="utf-8")

    # 5. Save environment metadata
    env_info = get_environment_info()
    env_info["sklearn_version"] = sklearn.__version__
    env_info["pandas_version"] = pd.__version__
    env_info["joblib_version"] = joblib.__version__
    if git_commit:
        env_info["git_commit"] = git_commit
    env_path = bundle_dir / "environment.json"
    env_path.write_text(json.dumps(env_info, indent=2), encoding="utf-8")

    # 6. Compute SHA-256 checksums
    bundle_sha256s = {
        "retriever.joblib": compute_sha256(weights_path),
        "model_config.json": compute_sha256(model_cfg_path),
        "feature_schema.json": compute_sha256(schema_path),
        "preprocessing_config.json": compute_sha256(prep_path),
        "environment.json": compute_sha256(env_path),
    }

    manifest = {
        "project_name": "infographix",
        "model_family": model_family,
        "semantic_version": ver,
        "full_model_name": full_name,
        "clearml_training_task_id": training_task_id,
        "dataset_identifier": dataset_identifier,
        "git_commit": env_info["git_commit"],
        "checkpoint_sha256": bundle_sha256s["retriever.joblib"],
        "bundle_sha256s": bundle_sha256s,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lifecycle_status": lifecycle_status,
        "parent_model_id": parent_model_id,
        "model_config": model_cfg,
        "preprocessing_config": prep_cfg,
        "environment": env_info,
        "metrics": {
            k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
            for k, v in (validation_metrics or {}).items()
        },
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def register_retriever_model(
    retriever_model: LayoutRetriever,
    training_task_id: str,
    semantic_version: str,
    dataset_identifier: str = "SciPostLayout v2 (70/15/15)",
    validation_metrics: dict[str, Any] | None = None,
    lifecycle_status: str = "candidate",
    parent_model_id: str | None = None,
    project_name: str = DEFAULT_PROJECT_NAME,
    model_family: str = RETRIEVER_MODEL_FAMILY,
    comment: str = "",
) -> str:
    """Register a fitted LayoutRetriever bundle in ClearML's native Model Registry."""
    if lifecycle_status not in LIFECYCLE_STAGES:
        raise ValueError(f"Invalid lifecycle_status '{lifecycle_status}'. Must be one of {LIFECYCLE_STAGES}")

    _validate_fitted_retriever(retriever_model, enforce_train_only=True)
    ver = normalize_version(semantic_version)
    full_model_name = build_model_name(model_family, ver)

    existing = Model.query_models(
        project_name=project_name,
        model_name=full_model_name,
        include_archived=True,
    )
    exact_matches = [m for m in existing if m.name == full_model_name]
    if exact_matches:
        raise ValueError(
            f"Model '{full_model_name}' already exists in ClearML project '{project_name}' "
            f"(ID: {exact_matches[0].id}). Registered model versions are immutable."
        )

    orig_task = Task.get_task(task_id=training_task_id)
    if orig_task is None:
        raise ValueError(f"Originating ClearML task '{training_task_id}' not found.")

    git_commit = get_git_commit()

    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_dir = Path(tmp_dir) / full_model_name
        manifest = build_retriever_bundle(
            bundle_dir=bundle_dir,
            retriever_model=retriever_model,
            training_task_id=training_task_id,
            dataset_identifier=dataset_identifier,
            semantic_version=ver,
            validation_metrics=validation_metrics,
            lifecycle_status=lifecycle_status,
            model_family=model_family,
            parent_model_id=parent_model_id,
            git_commit=git_commit,
        )

        tags = [
            f"family:{model_family}",
            f"version:v{ver}",
            "arch:LayoutRetriever",
            f"dataset:{dataset_identifier}",
            f"stage:{lifecycle_status}",
        ]
        full_comment = (
            f"Infographix {full_model_name} | Stage: {lifecycle_status} | "
            f"Task: {training_task_id} | Dataset: {dataset_identifier}"
        )
        if comment:
            full_comment = f"{full_comment}\n{comment}"

        output_model = OutputModel(
            task=orig_task,
            config_dict={
                "model_config": manifest["model_config"],
                "feature_schema": get_retriever_feature_schema(),
                "preprocessing_config": manifest["preprocessing_config"],
                "manifest": manifest,
            },
            name=full_model_name,
            tags=tags,
            comment=full_comment,
            framework="ScikitLearn",
            base_model_id=None,
        )

        upload_uri = _resolve_upload_uri(orig_task)
        uploaded_uri = output_model.update_weights(
            weights_filename=str(bundle_dir),
            upload_uri=upload_uri,
            target_filename=f"{full_model_name}.zip",
            auto_delete_file=False,
            async_enable=False,
            is_package=True,
        )
        output_model.wait_for_uploads()

        if not uploaded_uri or str(uploaded_uri).startswith("file://") or "failed_uploading" in str(uploaded_uri):
            raise RuntimeError(f"Retriever artifact upload failed or remained local: {uploaded_uri}")

        metadata_payload: dict[str, dict[str, str]] = {
            "model_family": {"value": model_family, "type": "str"},
            "semantic_version": {"value": ver, "type": "str"},
            "architecture": {"value": "LayoutRetriever", "type": "str"},
            "dataset_identifier": {"value": dataset_identifier, "type": "str"},
            "lifecycle_status": {"value": lifecycle_status, "type": "str"},
            "training_task_id": {"value": training_task_id, "type": "str"},
            "git_commit": {"value": str(git_commit), "type": "str"},
            "checkpoint_sha256": {"value": manifest["checkpoint_sha256"], "type": "str"},
            "n_train_posters": {"value": str(manifest["model_config"]["n_train_posters"]), "type": "int"},
            "train_poster_ids_sha256": {"value": manifest["model_config"]["train_poster_ids_sha256"], "type": "str"},
            "schema_version": {"value": RETRIEVER_SCHEMA_VERSION, "type": "str"},
        }
        if parent_model_id:
            metadata_payload["parent_model_id"] = {"value": parent_model_id, "type": "str"}

        for metric_name, metric_val in manifest["metrics"].items():
            if isinstance(metric_val, (int, float)):
                metadata_payload[f"metric.{metric_name}"] = {
                    "value": f"{metric_val:.6f}",
                    "type": "float",
                }

        ok = output_model.set_all_metadata(metadata_payload, replace=True)
        if not ok:
            raise RuntimeError(f"Failed to persist metadata on ClearML retriever '{output_model.id}'.")

        if parent_model_id:
            client = APIClient()
            client.models.edit(model=output_model.id, parent=parent_model_id)

        verified_model = InputModel(model_id=output_model.id)
        if not verified_model.url or "failed" in verified_model.url:
            raise RuntimeError(f"Retriever verification failed after registration: {verified_model.url}")

        return str(output_model.id)


def load_registered_retriever(
    model_id: str | None = None,
    semantic_version: str | None = None,
    project_name: str = DEFAULT_PROJECT_NAME,
    model_family: str = RETRIEVER_MODEL_FAMILY,
    force_download: bool = False,
    return_manifest: bool = False,
) -> LayoutRetriever | tuple[LayoutRetriever, dict[str, Any]]:
    """Deterministically load a SHA-256 verified LayoutRetriever from ClearML Model Registry.

    Requires an explicit `model_id` or `semantic_version`; refuses implicit latest/production
    resolution and refuses silent fallback to local `models/registry/`.
    """
    resolved_id = _resolve_exact_model_id(
        model_id=model_id,
        semantic_version=semantic_version,
        project_name=project_name,
        model_family=model_family,
    )

    input_model = InputModel(model_id=resolved_id)
    local_bundle = input_model.get_local_copy(
        extract_archive=True,
        raise_on_error=True,
        force_download=force_download,
    )
    if not local_bundle:
        raise RuntimeError(f"ClearML failed to download artifact bundle for retriever '{resolved_id}'.")

    bundle_dir = Path(local_bundle)
    if not bundle_dir.is_dir():
        raise RuntimeError(
            f"Expected extracted bundle directory for retriever '{resolved_id}', got file: {bundle_dir}"
        )

    manifest = verify_bundle_checksums(bundle_dir)

    cfg_path = bundle_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing model_config.json in downloaded bundle {bundle_dir}")
    model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    arch = model_cfg.get("architecture")
    if arch != "LayoutRetriever":
        raise ValueError(
            f"Incompatible model architecture '{arch}' in bundle; expected 'LayoutRetriever'."
        )

    schema_path = bundle_dir / "feature_schema.json"
    feature_schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else {}

    artifact_name = str(model_cfg.get("artifact_filename", "retriever.joblib"))
    joblib_path = bundle_dir / artifact_name
    if not joblib_path.exists():
        # Legacy bundle compatibility inside an extracted ClearML package
        legacy_joblib = bundle_dir / "pipeline.joblib"
        if legacy_joblib.exists():
            joblib_path = legacy_joblib
        else:
            raise FileNotFoundError(
                f"Corrupted retriever bundle: missing '{artifact_name}' in {bundle_dir}"
            )

    loaded_obj = joblib.load(joblib_path)
    if isinstance(loaded_obj, dict) and "model" in loaded_obj:
        loaded_obj = loaded_obj["model"]

    retriever = _validate_fitted_retriever(loaded_obj, enforce_train_only=True)
    retriever.clearml_model_id = resolved_id
    retriever.clearml_model_version = str(manifest.get("semantic_version", "unknown"))
    retriever.registry_manifest = manifest
    retriever.feature_schema = feature_schema

    if return_manifest:
        return retriever, manifest
    return retriever


def migrate_local_retriever_to_clearml(
    legacy_version_dir: str | Path,
    training_task_id: str,
    semantic_version: str | None = None,
    dataset_identifier: str = "SciPostLayout v2",
    lifecycle_status: str = "staging",
    project_name: str = DEFAULT_PROJECT_NAME,
    comment: str = "Migrated from local ModelRegistry artifact.",
) -> str:
    """One-time migration helper to upload a legacy local retriever bundle to ClearML."""
    vdir = Path(legacy_version_dir)
    pipeline_path = vdir / "pipeline.joblib"
    manifest_path = vdir / "manifest.json"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Legacy retriever bundle not found: {pipeline_path}")

    legacy_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    payload = joblib.load(pipeline_path)
    retriever = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    ver = semantic_version or str(legacy_manifest.get("version", "2.0.0"))

    return register_retriever_model(
        retriever_model=retriever,
        training_task_id=training_task_id,
        semantic_version=ver,
        dataset_identifier=dataset_identifier,
        validation_metrics=legacy_manifest.get("metrics", {}),
        lifecycle_status=lifecycle_status,
        project_name=project_name,
        comment=comment,
    )
