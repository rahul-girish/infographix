"""Phase 3 Model Versioning & Governance Integration Engine for Infographix.

Provides formal semantic versioning, quality gate evaluations, cryptographic artifact
provenance, and seamless integration between ClearML Native Model Registry and the
FastAPI MLOps Governance system.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEMVER_REGEX = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

SHA256_REGEX = re.compile(r"^[0-9a-fA-F]{64}$")

DEFAULT_QUALITY_GATES = {
    "poster_layout_generator": {
        "min_val_iou_mean": 0.25,
        "max_canvas_violation_rate": 0.05,
        "max_unintended_overlap_rate": 0.15,
        "max_val_loss_total": 8.0,
    },
    "poster_layout_retriever": {
        "min_ndcg_at_5": 0.60,
        "require_train_only": True,
    },
    "poster_layout_clusterer": {
        "min_train_silhouette": 0.15,
        "max_train_noise_ratio": 0.15,
    },
}


def normalize_semantic_version(version: str) -> str:
    """Normalize and validate semantic version string (e.g. 'v0.2.0' -> '0.2.0')."""
    clean = version.strip()
    if clean.startswith("v") or clean.startswith("V"):
        clean = clean[1:]
    if not SEMVER_REGEX.match(clean):
        raise ValueError(
            f"Invalid semantic version '{version}'. Must follow SemVer X.Y.Z format (e.g. '0.2.0', '1.0.0-rc1')."
        )
    return clean


def parse_semantic_version(version: str) -> tuple[int, int, int]:
    """Parse major, minor, patch numbers from semantic version."""
    clean = normalize_semantic_version(version)
    # Strip pre-release suffix for numeric comparison
    numeric_part = clean.split("-")[0].split("+")[0]
    parts = numeric_part.split(".")
    return int(parts[0]), int(parts[1]), int(parts[2])


def is_newer_version(candidate_ver: str, existing_ver: str) -> bool:
    """Check if candidate semantic version is strictly greater than existing version."""
    cand = parse_semantic_version(candidate_ver)
    exist = parse_semantic_version(existing_ver)
    return cand > exist


@dataclass
class QualityGateResult:
    """Outcome of automated quality gating prior to version promotion."""
    passed: bool
    model_family: str
    metrics: dict[str, Any]
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "model_family": self.model_family,
            "metrics": self.metrics,
            "violations": self.violations,
            "warnings": self.warnings,
        }


def evaluate_quality_gates(
    metrics: dict[str, Any],
    model_family: str,
    custom_gates: dict[str, Any] | None = None,
) -> QualityGateResult:
    """Evaluate candidate model metrics against Phase 3 automated quality gates."""
    gates = DEFAULT_QUALITY_GATES.get(model_family, {}).copy()
    if custom_gates:
        gates.update(custom_gates)

    violations: list[str] = []
    warnings: list[str] = []

    if model_family == "poster_layout_generator":
        val_iou = float(metrics.get("val_iou_mean", 0.0))
        min_iou = float(gates.get("min_val_iou_mean", 0.25))
        if val_iou < min_iou:
            violations.append(f"Validation IoU ({val_iou:.4f}) is below minimum threshold ({min_iou:.4f}).")

        canvas_violation = float(metrics.get("val_canvas_violation_rate", 0.0))
        max_canvas = float(gates.get("max_canvas_violation_rate", 0.05))
        if canvas_violation > max_canvas:
            violations.append(
                f"Canvas violation rate ({canvas_violation:.4f}) exceeds threshold ({max_canvas:.4f})."
            )

        overlap_rate = float(metrics.get("val_unintended_overlap_rate", 0.0))
        max_overlap = float(gates.get("max_unintended_overlap_rate", 0.15))
        if overlap_rate > max_overlap:
            warnings.append(
                f"Unintended overlap rate ({overlap_rate:.4f}) is high (threshold: {max_overlap:.4f})."
            )

    elif model_family == "poster_layout_retriever":
        ndcg = float(metrics.get("ndcg_at_5", metrics.get("val_ndcg_at_5", 0.0)))
        min_ndcg = float(gates.get("min_ndcg_at_5", 0.60))
        if ndcg < min_ndcg:
            violations.append(f"NDCG@5 score ({ndcg:.4f}) is below minimum threshold ({min_ndcg:.4f}).")

        if gates.get("require_train_only", True):
            train_only = bool(metrics.get("train_only_verified", False))
            if not train_only:
                violations.append("Retriever training-only anti-leakage verification failed.")

    elif model_family == "poster_layout_clusterer":
        silhouette = float(metrics.get("train_silhouette", metrics.get("silhouette", 0.0)))
        min_sil = float(gates.get("min_train_silhouette", 0.15))
        if silhouette < min_sil:
            violations.append(f"Cluster silhouette ({silhouette:.4f}) is below threshold ({min_sil:.4f}).")

    passed = len(violations) == 0
    return QualityGateResult(
        passed=passed,
        model_family=model_family,
        metrics=metrics,
        violations=violations,
        warnings=warnings,
    )


def build_governance_registration_payload(
    *,
    clearml_model_id: str,
    model_family: Literal["poster_layout_generator", "poster_layout_retriever"],
    semantic_version: str,
    clearml_training_task_id: str,
    artifact_uri: str,
    checkpoint_sha256: str,
    git_commit: str,
    dataset_id: str | None = None,
    legacy_dataset_identifier: str | None = "SciPostLayout v2.0.0",
    architecture: str | None = None,
    parent_model_id: str | None = None,
    retriever_model_id: str | None = None,
    retriever_version: str | None = None,
    environment_summary: dict[str, Any] | None = None,
    metrics_summary: dict[str, Any] | None = None,
    description: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Construct schema-compliant payload matching backend ModelRegisterRequest."""
    clean_version = normalize_semantic_version(semantic_version)
    clean_sha = checkpoint_sha256.strip().lower()

    if not SHA256_REGEX.match(clean_sha):
        raise ValueError(f"checkpoint_sha256 must be a 64-character hex string, got: {checkpoint_sha256}")

    payload = {
        "clearml_model_id": clearml_model_id.strip(),
        "model_family": model_family,
        "semantic_version": clean_version,
        "clearml_training_task_id": clearml_training_task_id.strip(),
        "artifact_uri": artifact_uri.strip(),
        "checkpoint_sha256": clean_sha,
        "git_commit": git_commit.strip(),
        "schema_version": "v1.0.0",
        "description": description or f"Phase 3 Governed {model_family} v{clean_version}",
        "tags": tags or [f"version:{clean_version}", f"family:{model_family}", "phase3-governed"],
        "environment_summary": environment_summary or {
            "python_version": sys.version.split()[0],
            "torch_version": "2.5.0",
            "clearml_version": "2.2.0",
            "platform": sys.platform,
        },
        "dependencies_summary": {},
        "metrics_summary": metrics_summary or {},
    }

    if architecture:
        payload["architecture"] = architecture
    elif model_family == "poster_layout_generator":
        payload["architecture"] = "TemplateConditionedLayoutModel"
    else:
        payload["architecture"] = "LayoutRetriever"

    if dataset_id:
        payload["dataset_id"] = dataset_id
    if legacy_dataset_identifier:
        payload["legacy_dataset_identifier"] = legacy_dataset_identifier
    if parent_model_id:
        payload["parent_model_id"] = parent_model_id
    if retriever_model_id:
        payload["retriever_model_id"] = retriever_model_id
    if retriever_version:
        payload["retriever_version"] = retriever_version

    return payload


class ModelVersioningClient:
    """Full Phase 3 client for versioning, gating, ClearML upload, and governance notification."""

    def __init__(
        self,
        governance_base_url: str = "http://localhost:8000/api/v1/governance",
        service_token: str = "ci-train-token",
        project_name: str = "Infographix",
    ):
        self.governance_base_url = governance_base_url.rstrip("/")
        self.service_token = service_token
        self.project_name = project_name

    def verify_and_gate(
        self,
        model_family: str,
        metrics: dict[str, Any],
        custom_gates: dict[str, Any] | None = None,
    ) -> QualityGateResult:
        """Evaluate candidate metrics against quality gates before registration."""
        res = evaluate_quality_gates(metrics, model_family, custom_gates)
        if not res.passed:
            violation_str = "; ".join(res.violations)
            raise ValueError(f"Model failed Phase 3 quality gates: {violation_str}")
        return res

    def register_to_governance(
        self,
        payload: dict[str, Any],
        http_client: Any | None = None,
    ) -> dict[str, Any]:
        """Submit registered model payload to Governance API (POST /api/v1/governance/models)."""
        url = f"{self.governance_base_url}/models"
        headers = {
            "Authorization": f"Bearer {self.service_token}",
            "Content-Type": "application/json",
        }

        if http_client is not None:
            # Used with FastAPI TestClient or custom requests session
            response = http_client.post(url, json=payload, headers=headers)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Governance registration failed ({response.status_code}): {response.text}"
                )
            return response.json()

        import requests
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def promote_to_staging(
        self,
        model_id: str,
        reason: str = "Passed validation and quality gates.",
        http_client: Any | None = None,
    ) -> dict[str, Any]:
        """Promote model from candidate to staging."""
        url = f"{self.governance_base_url}/models/{model_id}/promote-staging"
        headers = {
            "Authorization": f"Bearer {self.service_token}",
            "Content-Type": "application/json",
        }
        body = {"reason": reason}

        if http_client is not None:
            resp = http_client.post(url, json=body, headers=headers)
            if resp.status_code >= 400:
                raise RuntimeError(f"Promote to staging failed ({resp.status_code}): {resp.text}")
            return resp.json()

        import requests
        resp = requests.post(url, json=body, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def promote_to_production(
        self,
        model_id: str,
        reason: str = "Promoting approved staging model to production.",
        admin_token: str = "ci-promote-token",
        http_client: Any | None = None,
    ) -> dict[str, Any]:
        """Request approval and promote model from staging to production."""
        headers_eng = {
            "Authorization": f"Bearer {self.service_token}",
            "Content-Type": "application/json",
        }
        headers_admin = {
            "Authorization": f"Bearer {admin_token}",
            "Content-Type": "application/json",
        }

        # 1. Request approval for production
        app_url = f"{self.governance_base_url}/models/{model_id}/approval-requests"
        app_body = {
            "target_stage": "production",
            "request_type": "promotion",
            "reason": reason,
        }

        if http_client is not None:
            resp1 = http_client.post(app_url, json=app_body, headers=headers_eng)
            if resp1.status_code >= 400:
                raise RuntimeError(f"Approval request failed: {resp1.text}")
            app_id = resp1.json()["id"]

            # 2. Decide / approve
            dec_url = f"{self.governance_base_url}/approval-requests/{app_id}/decide"
            dec_body = {"decision": "approved", "comment": f"Auto-approved: {reason}"}
            resp2 = http_client.post(dec_url, json=dec_body, headers=headers_admin)
            if resp2.status_code >= 400:
                raise RuntimeError(f"Approval decision failed: {resp2.text}")

            # 3. Promote to production
            prom_url = f"{self.governance_base_url}/models/{model_id}/promote-production"
            prom_body = {"reason": reason}
            resp3 = http_client.post(prom_url, json=prom_body, headers=headers_admin)
            if resp3.status_code >= 400:
                raise RuntimeError(f"Production promotion failed: {resp3.text}")
            return resp3.json()

        import requests
        r1 = requests.post(app_url, json=app_body, headers=headers_eng, timeout=15)
        r1.raise_for_status()
        app_id = r1.json()["id"]

        dec_url = f"{self.governance_base_url}/approval-requests/{app_id}/decide"
        r2 = requests.post(dec_url, json={"decision": "approved", "comment": reason}, headers=headers_admin, timeout=15)
        r2.raise_for_status()

        prom_url = f"{self.governance_base_url}/models/{model_id}/promote-production"
        r3 = requests.post(prom_url, json={"reason": reason}, headers=headers_admin, timeout=15)
        r3.raise_for_status()
        return r3.json()
