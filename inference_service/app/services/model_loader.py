from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from clearml import Task

from inference_service.app.config import Settings
from src.models.layout_retrieval import (
    SHARED_CANVAS_COLS,
    SHARED_CAPACITY_COLS,
    SHARED_INVENTORY_COLS,
    SHARED_STRUCTURE_COLS,
    LayoutRetriever,
)
from src.neural_layout.dataset import CONTENT_FEAT_DIM, TEMPLATE_FEAT_DIM
from src.neural_layout.layout_model import TemplateConditionedLayoutModel
from src.registry.clearml_registry import (
    load_registered_model,
    load_registered_retriever,
    query_registered_models,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedModelMetadata:
    """Authoritative metadata resolved from ClearML for an in-memory model."""

    role: str
    model_name: str
    model_family: str
    clearml_model_id: str
    semantic_version: str
    architecture: str
    checkpoint_sha256: str
    lifecycle_status: str
    dataset_identifier: str
    supported_engines: list[str]


def configure_clearml_credentials(settings: Settings) -> None:
    """Initialize ClearML SDK authentication strictly from environment settings."""
    key = settings.CLEARML_API_ACCESS_KEY.strip()
    secret = settings.CLEARML_API_SECRET_KEY.strip()
    if not key or not secret:
        raise RuntimeError(
            "ClearML credentials missing: CLEARML_API_ACCESS_KEY and "
            "CLEARML_API_SECRET_KEY environment variables are required."
        )

    os.environ["CLEARML_API_ACCESS_KEY"] = key
    os.environ["CLEARML_API_SECRET_KEY"] = secret
    os.environ["CLEARML_API_HOST"] = settings.CLEARML_API_HOST
    os.environ["CLEARML_WEB_HOST"] = settings.CLEARML_WEB_HOST
    os.environ["CLEARML_FILES_HOST"] = settings.CLEARML_FILES_HOST
    if settings.CLEARML_CACHE_DIR:
        os.environ["CLEARML_CACHE_DIR"] = settings.CLEARML_CACHE_DIR

    Task.set_credentials(
        api_host=settings.CLEARML_API_HOST,
        web_host=settings.CLEARML_WEB_HOST,
        files_host=settings.CLEARML_FILES_HOST,
        key=key,
        secret=secret,
        store_conf_file=False,
    )


def _resolve_model_selector(
    *,
    role: str,
    project_name: str,
    model_family: str,
    model_id: Optional[str],
    semantic_version: Optional[str],
    lifecycle_stage: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Resolve an explicit model ID or semantic version without silent 'latest' fallbacks."""
    if model_id and model_id.strip():
        return model_id.strip(), semantic_version.strip() if semantic_version else None

    if semantic_version and semantic_version.strip():
        return None, semantic_version.strip()

    if lifecycle_stage and lifecycle_stage.strip():
        stage = lifecycle_stage.strip()
        matches = query_registered_models(
            project_name=project_name,
            model_family=model_family,
            lifecycle_status=stage,
            include_archived=False,
        )
        if len(matches) == 0:
            raise LookupError(
                f"No ClearML model found for {role} family '{model_family}' in stage '{stage}'."
            )
        if len(matches) > 1:
            ids = [m["model_id"] for m in matches]
            raise ValueError(
                f"Ambiguous stage resolution for {role} family '{model_family}' stage '{stage}': "
                f"found {len(matches)} models {ids}. Specify an explicit model ID or version."
            )
        resolved = matches[0]
        return str(resolved["model_id"]), str(resolved["semantic_version"])

    raise ValueError(
        f"Explicit ClearML model ID, semantic version, or unambiguous lifecycle stage is required for {role}."
    )


def _verify_retriever_schema(retriever: LayoutRetriever, manifest: dict[str, Any]) -> None:
    model_cfg = manifest.get("model_config") or {}
    arch = model_cfg.get("architecture")
    if arch and arch != "LayoutRetriever":
        raise ValueError(f"Retriever manifest architecture mismatch: got '{arch}', expected 'LayoutRetriever'.")

    if retriever.train_posters_ is None:
        raise ValueError("Loaded retriever has no fitted train_posters_ dataframe.")

    required_cols = (
        SHARED_CANVAS_COLS
        + SHARED_INVENTORY_COLS
        + SHARED_CAPACITY_COLS
        + SHARED_STRUCTURE_COLS
        + ["poster_id", "orientation"]
    )
    missing = [col for col in required_cols if col not in retriever.train_posters_.columns]
    if missing:
        raise ValueError(f"Retriever train_posters_ missing schema columns: {missing}")


def _verify_neural_model_schema(model: TemplateConditionedLayoutModel, manifest: dict[str, Any]) -> None:
    model_cfg = manifest.get("model_config") or {}
    arch = model_cfg.get("architecture")
    if arch and arch != "TemplateConditionedLayoutModel":
        raise ValueError(
            f"Neural manifest architecture mismatch: got '{arch}', expected 'TemplateConditionedLayoutModel'."
        )
    if int(model.content_dim) != CONTENT_FEAT_DIM or int(model.template_dim) != TEMPLATE_FEAT_DIM:
        raise ValueError(
            f"Neural feature dimension mismatch: expected ({CONTENT_FEAT_DIM}, {TEMPLATE_FEAT_DIM}), "
            f"got ({model.content_dim}, {model.template_dim})."
        )


def load_clearml_retriever(settings: Settings) -> tuple[LayoutRetriever, LoadedModelMetadata]:
    """Download, verify, and reconstruct the LayoutRetriever from ClearML Model Registry."""
    model_id, semantic_version = _resolve_model_selector(
        role="retriever",
        project_name=settings.CLEARML_PROJECT_NAME,
        model_family=settings.RETRIEVER_MODEL_FAMILY,
        model_id=settings.RETRIEVER_MODEL_ID,
        semantic_version=settings.RETRIEVER_MODEL_VERSION,
        lifecycle_stage=settings.RETRIEVER_LIFECYCLE_STAGE,
    )

    retriever, manifest = load_registered_retriever(
        model_id=model_id,
        semantic_version=semantic_version,
        project_name=settings.CLEARML_PROJECT_NAME,
        model_family=settings.RETRIEVER_MODEL_FAMILY,
        force_download=settings.CLEARML_FORCE_DOWNLOAD,
        return_manifest=True,
    )
    _verify_retriever_schema(retriever, manifest)

    resolved_id = str(getattr(retriever, "clearml_model_id", model_id or ""))
    resolved_ver = str(manifest.get("semantic_version", getattr(retriever, "clearml_model_version", "unknown")))
    meta = LoadedModelMetadata(
        role="retriever",
        model_name=str(manifest.get("full_model_name", f"{settings.RETRIEVER_MODEL_FAMILY}-v{resolved_ver}")),
        model_family=str(manifest.get("model_family", settings.RETRIEVER_MODEL_FAMILY)),
        clearml_model_id=resolved_id,
        semantic_version=resolved_ver,
        architecture="LayoutRetriever",
        checkpoint_sha256=str(manifest.get("checkpoint_sha256", "")),
        lifecycle_status=str(manifest.get("lifecycle_status", "unknown")),
        dataset_identifier=str(manifest.get("dataset_identifier", "unknown")),
        supported_engines=["adaptation", "neural"],
    )
    return retriever, meta


def load_clearml_neural_model(
    settings: Settings,
) -> tuple[TemplateConditionedLayoutModel, LoadedModelMetadata]:
    """Download, verify, and reconstruct TemplateConditionedLayoutModel from ClearML Model Registry."""
    model_id, semantic_version = _resolve_model_selector(
        role="neural_generator",
        project_name=settings.CLEARML_PROJECT_NAME,
        model_family=settings.NEURAL_MODEL_FAMILY,
        model_id=settings.NEURAL_MODEL_ID,
        semantic_version=settings.NEURAL_MODEL_VERSION,
        lifecycle_stage=settings.NEURAL_LIFECYCLE_STAGE,
    )

    model, manifest = load_registered_model(
        model_id=model_id,
        semantic_version=semantic_version,
        project_name=settings.CLEARML_PROJECT_NAME,
        model_family=settings.NEURAL_MODEL_FAMILY,
        device=settings.TORCH_DEVICE,
        force_download=settings.CLEARML_FORCE_DOWNLOAD,
        return_manifest=True,
    )
    _verify_neural_model_schema(model, manifest)

    resolved_id = str(getattr(model, "clearml_model_id", model_id or ""))
    resolved_ver = str(manifest.get("semantic_version", "unknown"))
    meta = LoadedModelMetadata(
        role="neural_generator",
        model_name=str(manifest.get("full_model_name", f"{settings.NEURAL_MODEL_FAMILY}-v{resolved_ver}")),
        model_family=str(manifest.get("model_family", settings.NEURAL_MODEL_FAMILY)),
        clearml_model_id=resolved_id,
        semantic_version=resolved_ver,
        architecture="TemplateConditionedLayoutModel",
        checkpoint_sha256=str(manifest.get("checkpoint_sha256", "")),
        lifecycle_status=str(manifest.get("lifecycle_status", "unknown")),
        dataset_identifier=str(manifest.get("dataset_identifier", "unknown")),
        supported_engines=["neural"],
    )
    return model, meta
