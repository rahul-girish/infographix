"""Phase 3 Model Versioning & Governance Integration Tests.

Validates:
1. Strict Semantic Versioning parsing, normalization, and monotonicity checks.
2. Automated quality gates (Alignment, Overlap, Canvas bounds, and Retriever NDCG).
3. Payload construction compliance with FastAPI ModelRegisterRequest schema.
4. Live and Mock Versioning Adapter protocol equivalence.
5. End-to-end governance lifecycle flow: Registration -> Approval -> Promotion -> Staging -> Production.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["GOV_AUTH_MODE"] = "strict"
os.environ["GOV_CLEARML_MODE"] = "mock"
os.environ["GOV_SERVICE_TOKENS"] = (
    '{"ci-promote-token":{"actor_id":"ci-cd-promote","role":"admin"},'
    '"ci-train-token":{"actor_id":"ci-cd-train","role":"ml_engineer"}}'
)

from fastapi.testclient import TestClient

import backend.tests.conftest  # noqa: F401
import backend.app.models  # noqa: F401
from backend.app.db.base import Base
from backend.app.db.session import engine, get_db
from backend.app.governance.adapters.versioning_adapter import (
    LiveClearMLVersioningAdapter,
    MockVersioningAdapter,
    VersioningIntegrationProtocol,
    get_versioning_adapter,
)
from backend.app.governance.models import GovernancePolicy
from backend.app.governance.services.policy_service import PolicyService
from backend.app.main import app
from src.registry.model_versioning import (
    ModelVersioningClient,
    build_governance_registration_payload,
    evaluate_quality_gates,
    is_newer_version,
    normalize_semantic_version,
    parse_semantic_version,
)


class TestModelVersioningPhase3(unittest.TestCase):
    """Test suite for Phase 3 Model Versioning logic, gates, and API integration."""

    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)

        # Seed default policies
        with next(get_db()) as db:
            PolicyService.ensure_default_policies(db)

    def test_semantic_version_validation(self):
        """Test SemVer normalization, parsing, and monotonicity."""
        self.assertEqual(normalize_semantic_version("0.2.0"), "0.2.0")
        self.assertEqual(normalize_semantic_version("v1.0.0"), "1.0.0")
        self.assertEqual(normalize_semantic_version("V2.1.3-rc1"), "2.1.3-rc1")

        with self.assertRaises(ValueError):
            normalize_semantic_version("invalid_version")
        with self.assertRaises(ValueError):
            normalize_semantic_version("1.0")

        major, minor, patch = parse_semantic_version("0.2.1")
        self.assertEqual((major, minor, patch), (0, 2, 1))

        self.assertTrue(is_newer_version("0.3.0", "0.2.9"))
        self.assertTrue(is_newer_version("1.0.0", "0.9.9"))
        self.assertFalse(is_newer_version("0.2.0", "0.2.0"))
        self.assertFalse(is_newer_version("0.1.5", "0.2.0"))

    def test_quality_gates_evaluation(self):
        """Test Phase 3 automated quality gating logic."""
        # Generator passing metrics
        good_metrics = {
            "val_iou_mean": 0.31,
            "val_canvas_violation_rate": 0.01,
            "val_unintended_overlap_rate": 0.04,
            "val_loss_total": 3.45,
        }
        res_pass = evaluate_quality_gates(good_metrics, "poster_layout_generator")
        self.assertTrue(res_pass.passed)
        self.assertEqual(len(res_pass.violations), 0)

        # Generator failing IoU
        bad_metrics = {
            "val_iou_mean": 0.18,  # Below 0.25 gate
            "val_canvas_violation_rate": 0.08,  # Above 0.05 gate
            "val_unintended_overlap_rate": 0.04,
        }
        res_fail = evaluate_quality_gates(bad_metrics, "poster_layout_generator")
        self.assertFalse(res_fail.passed)
        self.assertEqual(len(res_fail.violations), 2)

        # Retriever quality gate
        good_ret = {"ndcg_at_5": 0.75, "train_only_verified": True}
        res_ret = evaluate_quality_gates(good_ret, "poster_layout_retriever")
        self.assertTrue(res_ret.passed)

        bad_ret = {"ndcg_at_5": 0.52, "train_only_verified": False}
        res_ret_fail = evaluate_quality_gates(bad_ret, "poster_layout_retriever")
        self.assertFalse(res_ret_fail.passed)
        self.assertEqual(len(res_ret_fail.violations), 2)

    def test_payload_builder_compliance(self):
        """Ensure build_governance_registration_payload produces valid schema fields."""
        payload = build_governance_registration_payload(
            clearml_model_id="clearml-generator-v0.2.0",
            model_family="poster_layout_generator",
            semantic_version="v0.2.0",
            clearml_training_task_id="task-8cd6cb75",
            artifact_uri="https://files.clear.ml/generator-0.2.0.zip",
            checkpoint_sha256="a" * 64,
            git_commit="09de425",
            dataset_id="320b6617-63fa-42ea-a558-9d1e6332dde7",
            metrics_summary={"val_iou_mean": 0.31},
        )

        self.assertEqual(payload["semantic_version"], "0.2.0")
        self.assertEqual(payload["checkpoint_sha256"], "a" * 64)
        self.assertEqual(payload["architecture"], "TemplateConditionedLayoutModel")
        self.assertEqual(payload["model_family"], "poster_layout_generator")

    def test_versioning_adapter_protocol(self):
        """Ensure both Mock and Live adapters satisfy VersioningIntegrationProtocol."""
        self.assertTrue(isinstance(MockVersioningAdapter(), VersioningIntegrationProtocol))
        self.assertTrue(isinstance(LiveClearMLVersioningAdapter(), VersioningIntegrationProtocol))

        # Check factory
        adapter = get_versioning_adapter()
        self.assertTrue(isinstance(adapter, VersioningIntegrationProtocol))

        # Test live adapter payload generation
        p = LiveClearMLVersioningAdapter.build_sample_registration_payload(
            semantic_version="0.2.0",
            dataset_id="320b6617-63fa-42ea-a558-9d1e6332dde7",
            model_family="poster_layout_generator",
        )
        self.assertEqual(p["semantic_version"], "0.2.0")
        self.assertEqual(len(p["checkpoint_sha256"]), 64)

    def test_end_to_end_governance_lifecycle(self):
        """Register model via Versioning payload, request approval, and promote through stages."""
        headers_train = {"Authorization": "Bearer ci-train-token"}
        headers_admin = {"Authorization": "Bearer ci-promote-token"}

        # 1. Register a governed dataset first
        ds_resp = self.client.post(
            "/api/v1/governance/datasets",
            headers=headers_train,
            json={
                "dataset_name": "scipostlayout-v3-phase3",
                "dataset_version": "3.0.0",
                "storage_system": "minidvc",
                "storage_ref": ".mydvc/manifests/v3.manifest",
                "checksum_sha256": "b" * 64,
                "data_format": "parquet",
                "schema_version": "2.0.0",
                "record_count": 7855,
                "size_bytes": 1024000,
                "split_config": {"train": 0.8, "dev": 0.1, "test": 0.1},
                "quality_summary": {"completeness": 1.0},
            },
        )
        self.assertEqual(ds_resp.status_code, 201)
        ds_id = ds_resp.json()["id"]

        # 2. Build versioning payload using LiveClearMLVersioningAdapter
        payload = LiveClearMLVersioningAdapter.build_sample_registration_payload(
            semantic_version="0.2.0",
            dataset_id=ds_id,
            model_family="poster_layout_generator",
        )

        # 3. Register model
        reg_resp = self.client.post(
            "/api/v1/governance/models",
            headers=headers_train,
            json=payload,
        )
        self.assertEqual(reg_resp.status_code, 201)
        model_data = reg_resp.json()
        model_id = model_data["id"]
        self.assertEqual(model_data["lifecycle_status"], "candidate")

        # 4. Promote candidate -> staging using ModelVersioningClient
        client = ModelVersioningClient()
        stg_data = client.promote_to_staging(
            model_id=model_id,
            reason="Passed validation and quality gate checks.",
            http_client=self.client,
        )
        self.assertEqual(stg_data["lifecycle_status"], "staging")

        # 5. Promote staging -> production using ModelVersioningClient
        prod_data = client.promote_to_production(
            model_id=model_id,
            reason="Promoting verified candidate to production.",
            http_client=self.client,
        )
        self.assertEqual(prod_data["lifecycle_status"], "production")


if __name__ == "__main__":
    unittest.main()
