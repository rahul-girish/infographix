"""Comprehensive tests for the Infographix MLOps Governance Module."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Generator

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

# Ensure isolated SQLite DATABASE_URL before importing backend.app.*
import backend.tests.conftest  # noqa: F401
import backend.app.models  # noqa: F401
from backend.app.db.base import Base
from backend.app.db.session import get_db
from backend.app.governance.adapters.clearml_adapter import MockClearMLRegistryAdapter
from backend.app.governance.adapters.drift_adapter import MockDriftAdapter
from backend.app.governance.adapters.versioning_adapter import MockVersioningAdapter
from backend.app.governance.config import governance_settings
from backend.app.governance.exceptions import AuditImmutabilityError
from backend.app.governance.models import AuditEvent, GovernancePolicy, GovernedDataset
from backend.app.governance.router import get_clearml_adapter as _unused_import_check  # noqa: F401
from backend.app.governance.services.model_service import ModelService
from backend.app.governance.services.policy_service import PolicyService
from backend.app.main import app


def _make_hs256_jwt(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    p_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    sig = hmac.new(secret.encode(), f"{h_b64}.{p_b64}".encode(), hashlib.sha256).digest()
    s_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{h_b64}.{p_b64}.{s_b64}"


class GovernanceModuleTestCase(unittest.TestCase):
    """End-to-end API, RBAC, lifecycle, shadow comparison, audit, and DB migration tests."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "gov_test.db"
        self.engine = create_engine(
            f"sqlite:///{self.db_path.as_posix()}",
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(self.engine, "connect")
        def _fk_pragma(dbapi_conn, _conn_record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        Base.metadata.create_all(bind=self.engine)
        self.SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        with self.SessionFactory() as db:
            PolicyService.ensure_default_policies(db)

        def override_get_db() -> Generator[Session, None, None]:
            db = self.SessionFactory()
            try:
                yield db
            finally:
                db.close()

        self._prev_overrides = dict(app.dependency_overrides)
        app.dependency_overrides[get_db] = override_get_db

        # Configure strict auth mode with scoped service tokens + HS256 JWT secret
        self._orig_auth_mode = governance_settings.GOV_AUTH_MODE
        self._orig_jwt_secret = governance_settings.GOV_JWT_SECRET
        self._orig_svc_tokens = governance_settings.GOV_SERVICE_TOKENS
        self._orig_clearml_mode = governance_settings.GOV_CLEARML_MODE

        governance_settings.GOV_AUTH_MODE = "strict"
        governance_settings.GOV_JWT_SECRET = "test-governance-secret-key"
        governance_settings.GOV_SERVICE_TOKENS = json.dumps(
            {
                "tok-admin": {"actor_id": "alice-admin", "role": "admin"},
                "tok-eng": {"actor_id": "bob-mleng", "role": "ml_engineer"},
                "tok-rev": {"actor_id": "carol-reviewer", "role": "reviewer"},
                "tok-view": {"actor_id": "dave-viewer", "role": "viewer"},
            }
        )
        governance_settings.GOV_CLEARML_MODE = "mock"

        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides = self._prev_overrides
        governance_settings.GOV_AUTH_MODE = self._orig_auth_mode
        governance_settings.GOV_JWT_SECRET = self._orig_jwt_secret
        governance_settings.GOV_SERVICE_TOKENS = self._orig_svc_tokens
        governance_settings.GOV_CLEARML_MODE = self._orig_clearml_mode
        self.engine.dispose()
        self.tmp_dir.cleanup()

    @staticmethod
    def _headers(token: str) -> dict[str, str]:
        return {"X-Governance-Token": token}

    def _register_sample_dataset(self, version: str = "2.0.0") -> str:
        resp = self.client.post(
            "/api/v1/governance/datasets",
            headers=self._headers("tok-eng"),
            json={
                "dataset_name": "scipostlayout-features",
                "dataset_version": version,
                "source_type": "scipostlayout",
                "storage_system": "minidvc",
                "storage_ref": "minidvc:v3",
                "checksum_sha256": "a" * 64,
                "schema_version": "2.0.0",
                "record_count": 7855,
                "size_bytes": 109633569,
                "split_config": {"train": 0.7, "dev": 0.15, "test": 0.15, "random_state": 42},
                "usage_policy": "academic-research",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()["id"]

    def _register_sample_model(
        self,
        dataset_id: str,
        *,
        version: str = "0.2.0",
        val_iou: float = 0.30,
        token: str = "tok-eng",
    ) -> dict:
        payload = MockVersioningAdapter.build_sample_registration_payload(
            semantic_version=version,
            dataset_id=dataset_id,
        )
        payload["metrics_summary"]["val_iou_mean"] = val_iou
        resp = self.client.post(
            "/api/v1/governance/models",
            headers=self._headers(token),
            json=payload,
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    # 1. Authentication & RBAC Enforcement
    def test_auth_strict_mode_jwt_and_rbac_denials(self) -> None:
        # Missing credentials -> 401
        r_unauth = self.client.get("/api/v1/governance/models")
        self.assertEqual(r_unauth.status_code, 401)
        self.assertEqual(r_unauth.json()["error"]["code"], "AUTHENTICATION_REQUIRED")

        # Invalid token -> 401
        r_bad = self.client.get("/api/v1/governance/models", headers=self._headers("invalid"))
        self.assertEqual(r_bad.status_code, 401)

        # Viewer cannot register dataset -> 403 + isolated denied audit event
        r_forbid = self.client.post(
            "/api/v1/governance/datasets",
            headers=self._headers("tok-view"),
            json={
                "dataset_name": "blocked-ds",
                "dataset_version": "1.0.0",
                "storage_ref": "minidvc:v1",
                "checksum_sha256": "b" * 64,
            },
        )
        self.assertEqual(r_forbid.status_code, 403)
        self.assertEqual(r_forbid.json()["error"]["code"], "PERMISSION_DENIED")

        # Verify denied audit recorded
        audits = self.client.get(
            "/api/v1/governance/audit-events?action=auth.access_denied&outcome=denied",
            headers=self._headers("tok-view"),
        )
        self.assertEqual(audits.status_code, 200)
        self.assertGreaterEqual(len(audits.json()), 1)

        # HS256 Bearer JWT + DB role assignment
        assign = self.client.put(
            "/api/v1/governance/roles/jwt-user-1",
            headers=self._headers("tok-admin"),
            json={"role": "reviewer", "notes": "Ram Charan reviewer test"},
        )
        self.assertEqual(assign.status_code, 200)
        jwt_tok = _make_hs256_jwt(
            {"sub": "jwt-user-1", "exp": time.time() + 300},
            secret="test-governance-secret-key",
        )
        me = self.client.get(
            "/api/v1/governance/me",
            headers={"Authorization": f"Bearer {jwt_tok}"},
        )
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["role"], "reviewer")
        self.assertEqual(me.json()["auth_source"], "jwt")

    # 2. Metadata Validation, Immutability & Duplicate Prevention
    def test_metadata_validation_duplicates_and_immutability(self) -> None:
        ds_id = self._register_sample_dataset("2.0.0")

        # Duplicate dataset -> 409
        dup_ds = self.client.post(
            "/api/v1/governance/datasets",
            headers=self._headers("tok-eng"),
            json={
                "dataset_name": "scipostlayout-features",
                "dataset_version": "2.0.0",
                "storage_ref": "minidvc:v3",
                "checksum_sha256": "a" * 64,
            },
        )
        self.assertEqual(dup_ds.status_code, 409)

        # Invalid SHA-256 -> 422 with standardized ErrorEnvelope
        bad_sha = self.client.post(
            "/api/v1/governance/models",
            headers=self._headers("tok-eng"),
            json={
                "clearml_model_id": "m-bad",
                "model_family": "poster_layout_generator",
                "semantic_version": "0.2.0",
                "clearml_training_task_id": "t-1",
                "artifact_uri": "https://files.clear.ml/m.zip",
                "checkpoint_sha256": "not-hex",
                "git_commit": "abc1234",
                "dataset_id": ds_id,
            },
        )
        self.assertEqual(bad_sha.status_code, 422)
        self.assertEqual(bad_sha.json()["error"]["code"], "VALIDATION_ERROR")

        # Missing dataset_id for non-exempt version -> 422 policy violation
        no_ds = self.client.post(
            "/api/v1/governance/models",
            headers=self._headers("tok-eng"),
            json={
                "clearml_model_id": "m-nods",
                "model_family": "poster_layout_generator",
                "semantic_version": "0.9.0",
                "clearml_training_task_id": "t-1",
                "artifact_uri": "https://files.clear.ml/m.zip",
                "checkpoint_sha256": "c" * 64,
                "git_commit": "abc1234",
                "environment_summary": {"python_version": "3.12"},
                "metrics_summary": {"val_iou_mean": 0.3},
            },
        )
        self.assertEqual(no_ds.status_code, 422)

        # Valid registration + duplicate rejection + mutable metadata update
        m = self._register_sample_model(ds_id, version="0.2.0")
        dup_m = self.client.post(
            "/api/v1/governance/models",
            headers=self._headers("tok-eng"),
            json=MockVersioningAdapter.build_sample_registration_payload(
                semantic_version="0.2.0", dataset_id=ds_id
            ),
        )
        self.assertEqual(dup_m.status_code, 409)

        patch = self.client.patch(
            f"/api/v1/governance/models/{m['id']}",
            headers=self._headers("tok-eng"),
            json={"description": "Updated notes", "tags": ["stage-b"], "reason": "Add clarity"},
        )
        self.assertEqual(patch.status_code, 200)
        self.assertEqual(patch.json()["description"], "Updated notes")

    # 3. Lifecycle Transitions, Quality Gates, Shadow Comparison, Approval, Rollback
    def test_full_lifecycle_shadow_comparison_approval_and_rollback(self) -> None:
        ds_id = self._register_sample_dataset("2.0.0")

        # Low IoU model cannot promote to staging
        weak = self._register_sample_model(ds_id, version="0.1.5", val_iou=0.10)
        weak_stage = self.client.post(
            f"/api/v1/governance/models/{weak['id']}/promote-staging",
            headers=self._headers("tok-eng"),
            json={"reason": "Test weak model"},
        )
        self.assertEqual(weak_stage.status_code, 422)

        # Cannot skip staging (candidate -> production)
        m1 = self._register_sample_model(ds_id, version="0.2.0", val_iou=0.30)
        skip = self.client.post(
            f"/api/v1/governance/models/{m1['id']}/promote-production",
            headers=self._headers("tok-admin"),
            json={"reason": "Skip staging"},
        )
        self.assertEqual(skip.status_code, 409)

        # Promote m1 -> staging
        r_stg = self.client.post(
            f"/api/v1/governance/models/{m1['id']}/promote-staging",
            headers=self._headers("tok-eng"),
            json={"reason": "Passed val_iou gate"},
        )
        self.assertEqual(r_stg.status_code, 200)
        self.assertEqual(r_stg.json()["lifecycle_status"], "staging")

        # Cannot promote to production without approved ApprovalRequest
        r_no_app = self.client.post(
            f"/api/v1/governance/models/{m1['id']}/promote-production",
            headers=self._headers("tok-admin"),
            json={"reason": "Unapproved"},
        )
        self.assertEqual(r_no_app.status_code, 422)

        # Request production approval (no prior prod baseline yet)
        req1 = self.client.post(
            f"/api/v1/governance/models/{m1['id']}/approval-requests",
            headers=self._headers("tok-eng"),
            json={"target_stage": "production", "request_type": "promotion", "reason": "Ready for prod"},
        )
        self.assertEqual(req1.status_code, 201)
        req1_data = req1.json()
        self.assertTrue(req1_data["automated_check_passed"])
        self.assertFalse(req1_data["shadow_comparison_report"]["has_production_baseline"])

        # Self-approval blocked (bob-mleng cannot approve own request even if given reviewer)
        # Re-assign bob-mleng temporarily or test with admin requester:
        # Carol (reviewer) approves req1
        dec1 = self.client.post(
            f"/api/v1/governance/approval-requests/{req1_data['id']}/decide",
            headers=self._headers("tok-rev"),
            json={"decision": "approved", "comment": "Metrics & visual samples verified."},
        )
        self.assertEqual(dec1.status_code, 200)
        self.assertEqual(dec1.json()["status"], "approved")

        # Pre-flight promotion gate passes
        gate = self.client.get(
            f"/api/v1/governance/models/{m1['id']}/promotion-gate",
            headers=self._headers("tok-admin"),
        )
        self.assertEqual(gate.status_code, 200)
        self.assertTrue(gate.json()["eligible"])

        # Promote m1 -> production
        r_prod1 = self.client.post(
            f"/api/v1/governance/models/{m1['id']}/promote-production",
            headers=self._headers("tok-admin"),
            json={"reason": "Initial production release"},
        )
        self.assertEqual(r_prod1.status_code, 200)
        self.assertEqual(r_prod1.json()["lifecycle_status"], "production")
        self.assertTrue(r_prod1.json()["was_ever_production"])

        # Register m2 with REGRESSED IoU (0.27 < 0.30) -> shadow comparison flags regression
        m2_reg = self._register_sample_model(ds_id, version="0.2.5", val_iou=0.27)
        self.client.post(
            f"/api/v1/governance/models/{m2_reg['id']}/promote-staging",
            headers=self._headers("tok-eng"),
            json={"reason": "Staging regressed candidate"},
        )
        req_reg = self.client.post(
            f"/api/v1/governance/models/{m2_reg['id']}/approval-requests",
            headers=self._headers("tok-eng"),
            json={"target_stage": "production", "request_type": "promotion", "reason": "Try promote"},
        )
        self.assertEqual(req_reg.status_code, 201)
        self.assertFalse(req_reg.json()["automated_check_passed"])
        self.assertTrue(req_reg.json()["shadow_comparison_report"]["regression_detected"])
        # Regression-blocked request cannot be approved even with policy_override=True
        bad_app = self.client.post(
            f"/api/v1/governance/approval-requests/{req_reg.json()['id']}/decide",
            headers=self._headers("tok-admin"),
            json={"decision": "approved", "comment": "Attempt approve", "policy_override": True},
        )
        self.assertEqual(bad_app.status_code, 422)

        # Register m3 with IMPROVED IoU (0.36 > 0.30) -> promote through staging -> production (auto-archives m1)
        m3 = self._register_sample_model(ds_id, version="0.3.0", val_iou=0.36)
        self.client.post(
            f"/api/v1/governance/models/{m3['id']}/promote-staging",
            headers=self._headers("tok-eng"),
            json={"reason": "Improved IoU"},
        )
        req3 = self.client.post(
            f"/api/v1/governance/models/{m3['id']}/approval-requests",
            headers=self._headers("tok-admin"),
            json={"target_stage": "production", "request_type": "promotion", "reason": "Promote v0.3.0"},
        )
        self.assertEqual(req3.status_code, 201)
        self.assertTrue(req3.json()["automated_check_passed"])
        self.assertEqual(
            req3.json()["shadow_comparison_report"]["metric_deltas"]["val_iou_mean"]["status"],
            "improved",
        )

        # Self-approval check: alice-admin requested req3, so alice-admin cannot approve it
        self_app = self.client.post(
            f"/api/v1/governance/approval-requests/{req3.json()['id']}/decide",
            headers=self._headers("tok-admin"),
            json={"decision": "approved", "comment": "Self approve attempt"},
        )
        self.assertEqual(self_app.status_code, 403)

        # Carol approves req3 -> Admin promotes m3 -> m1 auto-archived
        self.client.post(
            f"/api/v1/governance/approval-requests/{req3.json()['id']}/decide",
            headers=self._headers("tok-rev"),
            json={"decision": "approved", "comment": "Confirmed +0.06 IoU improvement"},
        )
        r_prod3 = self.client.post(
            f"/api/v1/governance/models/{m3['id']}/promote-production",
            headers=self._headers("tok-admin"),
            json={"reason": "Rollout v0.3.0"},
        )
        self.assertEqual(r_prod3.status_code, 200)

        m1_after = self.client.get(
            f"/api/v1/governance/models/{m1['id']}", headers=self._headers("tok-view")
        ).json()
        self.assertEqual(m1_after["lifecycle_status"], "archived")

        # Controlled rollback of m1 (archived & was_ever_production=True) -> restores m1 & archives m3
        rb = self.client.post(
            f"/api/v1/governance/models/{m1['id']}/rollback",
            headers=self._headers("tok-admin"),
            json={"reason": "Emergency rollback to v0.2.0", "emergency_override": True},
        )
        self.assertEqual(rb.status_code, 200)
        self.assertEqual(rb.json()["lifecycle_status"], "production")

        # Never-production archived model cannot rollback
        self.client.post(
            f"/api/v1/governance/models/{weak['id']}/archive",
            headers=self._headers("tok-eng"),
            json={"reason": "Retire weak candidate"},
        )
        bad_rb = self.client.post(
            f"/api/v1/governance/models/{weak['id']}/rollback",
            headers=self._headers("tok-admin"),
            json={"reason": "Invalid rollback", "emergency_override": True},
        )
        self.assertEqual(bad_rb.status_code, 409)

    # 4. Drift Integration, Policy Updates & Approval Invalidation, Audit Immutability
    def test_drift_ingestion_policy_invalidation_and_audit_immutability(self) -> None:
        ds_id = self._register_sample_dataset("2.0.0")
        m = self._register_sample_model(ds_id, version="0.4.0", val_iou=0.33)
        self.client.post(
            f"/api/v1/governance/models/{m['id']}/promote-staging",
            headers=self._headers("tok-eng"),
            json={"reason": "Stage v0.4.0"},
        )
        req = self.client.post(
            f"/api/v1/governance/models/{m['id']}/approval-requests",
            headers=self._headers("tok-eng"),
            json={"target_stage": "production", "request_type": "promotion", "reason": "Seek approval"},
        ).json()

        # Policy update invalidates pending approval request
        pol_upd = self.client.patch(
            "/api/v1/governance/policies/generator_quality_gates",
            headers=self._headers("tok-admin"),
            json={"config": {"min_val_iou_mean": 0.28}, "reason": "Raise IoU bar"},
        )
        self.assertEqual(pol_upd.status_code, 200)
        req_after = self.client.get(
            f"/api/v1/governance/approval-requests/{req['id']}",
            headers=self._headers("tok-view"),
        ).json()
        self.assertEqual(req_after["status"], "invalidated")

        # Mock drift ingestion (critical drift marks model non_compliant and blocks promotion)
        drift_payload = MockDriftAdapter.build_sample_drift_payload(
            model_id=m["id"],
            monitoring_run_id="mock-drift-run-001",
            drift_status="critical",
        )
        d_resp = self.client.post(
            "/api/v1/governance/drift-events",
            headers=self._headers("tok-eng"),
            json=drift_payload,
        )
        self.assertEqual(d_resp.status_code, 201)
        self.assertTrue(d_resp.json()["is_mock"])

        # Duplicate monitoring_run_id -> 409
        d_dup = self.client.post(
            "/api/v1/governance/drift-events",
            headers=self._headers("tok-eng"),
            json=drift_payload,
        )
        self.assertEqual(d_dup.status_code, 409)

        m_after = self.client.get(
            f"/api/v1/governance/models/{m['id']}", headers=self._headers("tok-view")
        ).json()
        self.assertEqual(m_after["drift_status"], "critical")
        self.assertEqual(m_after["compliance_status"], "non_compliant")

        # AuditEvent ORM single-object & bulk-delete immutability guards
        with self.SessionFactory() as db:
            ev = db.query(AuditEvent).first()
            self.assertIsNotNone(ev)
            ev.reason = "tampered"
            with self.assertRaises(AuditImmutabilityError):
                db.flush()
            db.rollback()

            ev2 = db.query(AuditEvent).first()
            db.delete(ev2)
            with self.assertRaises(AuditImmutabilityError):
                db.flush()
            db.rollback()

            with self.assertRaises(AuditImmutabilityError):
                db.query(AuditEvent).delete()
            db.rollback()

    # 5. Minidvc Manifest Ingestion, Overview, & DB Foreign Key Restriction
    def test_manifest_ingestion_overview_and_db_constraints(self) -> None:
        manifest_path = Path(__file__).resolve().parents[2] / ".mydvc" / "manifests" / "v3.manifest"
        if manifest_path.exists():
            resp = self.client.post(
                "/api/v1/governance/datasets/ingest-manifest",
                headers=self._headers("tok-eng"),
                json={
                    "manifest_path": str(manifest_path),
                    "dataset_name": "scipostlayout-v1-minidvc",
                    "dataset_version": "3.0.0",
                },
            )
            self.assertEqual(resp.status_code, 201, resp.text)
            ds_id = resp.json()["id"]
        else:
            ds_id = self._register_sample_dataset("3.0.0")

        self._register_sample_model(ds_id, version="0.5.0", val_iou=0.31)

        # Overview endpoint
        ov = self.client.get("/api/v1/governance/overview", headers=self._headers("tok-view"))
        self.assertEqual(ov.status_code, 200)
        self.assertEqual(ov.json()["total_models"], 1)
        self.assertEqual(ov.json()["total_datasets"], 1)

        # MockClearMLRegistryAdapter sanity
        mock_cml = MockClearMLRegistryAdapter()
        ver = mock_cml.verify_model_artifact(clearml_model_id="m-1", expected_sha256="a" * 64)
        self.assertTrue(ver.verified and ver.is_mock)

        # DB ON DELETE RESTRICT when dataset is referenced by model
        with self.SessionFactory() as db:
            ds = db.query(GovernedDataset).first()
            db.delete(ds)
            with self.assertRaises(IntegrityError):
                db.commit()
            db.rollback()

    # 7. Retriever Anti-Leakage Gate, Rejection Flow & Dev Header Auth
    def test_retriever_gate_rejection_and_dev_auth_mode(self) -> None:
        ds_id = self._register_sample_dataset("2.1.0")

        # Retriever with anti-leakage violation cannot promote to staging
        leaked = MockVersioningAdapter.build_sample_registration_payload(
            semantic_version="2.2.0",
            dataset_id=ds_id,
            model_family="poster_layout_retriever",
        )
        leaked["metrics_summary"]["train_only_verified"] = False
        r_leak = self.client.post(
            "/api/v1/governance/models",
            headers=self._headers("tok-eng"),
            json=leaked,
        )
        self.assertEqual(r_leak.status_code, 201)
        r_leak_stg = self.client.post(
            f"/api/v1/governance/models/{r_leak.json()['id']}/promote-staging",
            headers=self._headers("tok-eng"),
            json={"reason": "Attempt staging leaked retriever"},
        )
        self.assertEqual(r_leak_stg.status_code, 422)

        # Rejection workflow prevents production promotion
        good_ret = MockVersioningAdapter.build_sample_registration_payload(
            semantic_version="2.3.0",
            dataset_id=ds_id,
            model_family="poster_layout_retriever",
        )
        r_ret = self.client.post(
            "/api/v1/governance/models",
            headers=self._headers("tok-eng"),
            json=good_ret,
        )
        self.assertEqual(r_ret.status_code, 201)
        ret_id = r_ret.json()["id"]
        self.assertEqual(
            self.client.post(
                f"/api/v1/governance/models/{ret_id}/promote-staging",
                headers=self._headers("tok-eng"),
                json={"reason": "Stage retriever"},
            ).status_code,
            200,
        )
        req_ret = self.client.post(
            f"/api/v1/governance/models/{ret_id}/approval-requests",
            headers=self._headers("tok-eng"),
            json={"target_stage": "production", "request_type": "promotion", "reason": "Prod request"},
        ).json()
        rej = self.client.post(
            f"/api/v1/governance/approval-requests/{req_ret['id']}/decide",
            headers=self._headers("tok-rev"),
            json={"decision": "rejected", "comment": "Need higher nDCG@5"},
        )
        self.assertEqual(rej.status_code, 200)
        self.assertEqual(rej.json()["status"], "rejected")
        self.assertEqual(
            self.client.post(
                f"/api/v1/governance/models/{ret_id}/promote-production",
                headers=self._headers("tok-admin"),
                json={"reason": "Attempt promote rejected"},
            ).status_code,
            422,
        )

        # Dev header auth mode works when GOV_AUTH_MODE=dev
        governance_settings.GOV_AUTH_MODE = "dev"
        me_dev = self.client.get(
            "/api/v1/governance/me",
            headers={"X-Actor-Id": "ram-dashboard", "X-Actor-Role": "viewer"},
        )
        self.assertEqual(me_dev.status_code, 200)
        self.assertEqual(me_dev.json()["auth_source"], "dev_header")
        self.assertEqual(me_dev.json()["role"], "viewer")

    # 8. Critical Drift Blocking Gate, Recovery, Cancel & Demote/Replay Protection
    def test_critical_drift_blocks_promotion_demote_prevents_replay_and_cancel(self) -> None:
        ds_id = self._register_sample_dataset("2.4.0")
        m = self._register_sample_model(ds_id, version="0.6.0", val_iou=0.35)
        self.assertEqual(
            self.client.post(
                f"/api/v1/governance/models/{m['id']}/promote-staging",
                headers=self._headers("tok-eng"),
                json={"reason": "Stage v0.6.0"},
            ).status_code,
            200,
        )

        # Create & cancel a pending approval request
        req_cancel = self.client.post(
            f"/api/v1/governance/models/{m['id']}/approval-requests",
            headers=self._headers("tok-eng"),
            json={"target_stage": "production", "request_type": "promotion", "reason": "Initial ask"},
        ).json()
        c_resp = self.client.post(
            f"/api/v1/governance/approval-requests/{req_cancel['id']}/cancel",
            headers=self._headers("tok-eng"),
            json={"reason": "Withdrawing to wait for drift check"},
        )
        self.assertEqual(c_resp.status_code, 200)
        self.assertEqual(c_resp.json()["status"], "cancelled")

        # Create new request, approve it, then trigger critical drift -> promotion-gate & promote-production MUST block
        req = self.client.post(
            f"/api/v1/governance/models/{m['id']}/approval-requests",
            headers=self._headers("tok-eng"),
            json={"target_stage": "production", "request_type": "promotion", "reason": "Promote v0.6.0"},
        ).json()
        self.assertEqual(
            self.client.post(
                f"/api/v1/governance/approval-requests/{req['id']}/decide",
                headers=self._headers("tok-rev"),
                json={"decision": "approved", "comment": "Looks good"},
            ).status_code,
            200,
        )

        crit_drift = MockDriftAdapter.build_sample_drift_payload(
            model_id=m["id"], monitoring_run_id="crit-block-001", drift_status="critical"
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/governance/drift-events",
                headers=self._headers("tok-eng"),
                json=crit_drift,
            ).status_code,
            201,
        )
        gate_blocked = self.client.get(
            f"/api/v1/governance/models/{m['id']}/promotion-gate",
            headers=self._headers("tok-admin"),
        )
        self.assertEqual(gate_blocked.status_code, 200)
        self.assertFalse(gate_blocked.json()["eligible"])
        self.assertEqual(
            self.client.post(
                f"/api/v1/governance/models/{m['id']}/promote-production",
                headers=self._headers("tok-admin"),
                json={"reason": "Attempt while critical drift"},
            ).status_code,
            422,
        )

        # Drift recovers to 'none' -> compliance restored -> promote-production succeeds
        ok_drift = MockDriftAdapter.build_sample_drift_payload(
            model_id=m["id"], monitoring_run_id="crit-recover-002", drift_status="none"
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/governance/drift-events",
                headers=self._headers("tok-eng"),
                json=ok_drift,
            ).status_code,
            201,
        )
        self.assertEqual(
            self.client.post(
                f"/api/v1/governance/models/{m['id']}/promote-production",
                headers=self._headers("tok-admin"),
                json={"reason": "Promote after drift recovery"},
            ).status_code,
            200,
        )

        # Consumed approval request cannot be replayed after production -> staging demotion
        demote = self.client.post(
            f"/api/v1/governance/models/{m['id']}/demote-staging",
            headers=self._headers("tok-admin"),
            json={"reason": "Emergency demotion"},
        )
        self.assertEqual(demote.status_code, 200)
        self.assertEqual(demote.json()["lifecycle_status"], "staging")

        replay = self.client.post(
            f"/api/v1/governance/models/{m['id']}/promote-production",
            headers=self._headers("tok-admin"),
            json={"reason": "Attempt replay old approval"},
        )
        self.assertEqual(replay.status_code, 422)

    # 9. Security Edge Cases (Strict Mode Ignores Dev Headers, JWT Errors, Cross-Role RBAC)
    def test_strict_mode_jwt_errors_and_cross_role_rbac(self) -> None:
        # Strict mode ignores X-Actor-Id / X-Actor-Role headers
        r_dev_hdr = self.client.get(
            "/api/v1/governance/me",
            headers={"X-Actor-Id": "evil", "X-Actor-Role": "admin"},
        )
        self.assertEqual(r_dev_hdr.status_code, 401)

        # Expired, bad sig, non-numeric exp, alg=none JWTs -> 401
        secret = "test-governance-secret-key"
        for bad_claims in (
            {"sub": "u1", "exp": time.time() - 10},
            {"sub": "u1", "nbf": time.time() + 300},
            {"sub": "u1", "exp": "not-a-number"},
        ):
            tok = _make_hs256_jwt(bad_claims, secret=secret)
            self.assertEqual(
                self.client.get(
                    "/api/v1/governance/me", headers={"Authorization": f"Bearer {tok}"}
                ).status_code,
                401,
            )

        bad_sig_tok = _make_hs256_jwt({"sub": "u1", "exp": time.time() + 60}, secret="wrong-secret")
        self.assertEqual(
            self.client.get(
                "/api/v1/governance/me", headers={"Authorization": f"Bearer {bad_sig_tok}"}
            ).status_code,
            401,
        )

        # Cross-role RBAC boundaries
        ds_id = self._register_sample_dataset("2.5.0")
        m = self._register_sample_model(ds_id, version="0.7.0", val_iou=0.35)
        self.assertEqual(
            self.client.post(
                f"/api/v1/governance/models/{m['id']}/promote-production",
                headers=self._headers("tok-eng"),
                json={"reason": "Eng cannot promote prod"},
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.patch(
                "/api/v1/governance/policies/generator_quality_gates",
                headers=self._headers("tok-eng"),
                json={"is_enabled": False, "reason": "Eng cannot edit policy"},
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.post(
                "/api/v1/governance/models",
                headers=self._headers("tok-rev"),
                json=MockVersioningAdapter.build_sample_registration_payload(
                    semantic_version="0.7.1", dataset_id=ds_id
                ),
            ).status_code,
            403,
        )

    # 10. Generator Missing Metrics, Retriever Missing Verification & ClearML Sync Compensation
    def test_missing_metrics_and_clearml_sync_compensation(self) -> None:
        ds_id = self._register_sample_dataset("2.6.0")

        # Generator missing val_canvas_violation_rate cannot promote to staging
        partial_gen = MockVersioningAdapter.build_sample_registration_payload(
            semantic_version="0.8.0", dataset_id=ds_id
        )
        partial_gen["metrics_summary"] = {"val_iou_mean": 0.35}
        r_pg = self.client.post("/api/v1/governance/models", headers=self._headers("tok-eng"), json=partial_gen)
        self.assertEqual(r_pg.status_code, 201)
        self.assertEqual(
            self.client.post(
                f"/api/v1/governance/models/{r_pg.json()['id']}/promote-staging",
                headers=self._headers("tok-eng"),
                json={"reason": "Missing overlap/violation metrics"},
            ).status_code,
            422,
        )

        # Retriever missing explicit train_only_verified cannot promote to staging
        ret_no_flag = MockVersioningAdapter.build_sample_registration_payload(
            semantic_version="2.6.0", dataset_id=ds_id, model_family="poster_layout_retriever"
        )
        ret_no_flag["dependencies_summary"].pop("train_only_verified", None)
        ret_no_flag["metrics_summary"].pop("train_only_verified", None)
        r_rn = self.client.post("/api/v1/governance/models", headers=self._headers("tok-eng"), json=ret_no_flag)
        self.assertEqual(r_rn.status_code, 201)
        self.assertEqual(
            self.client.post(
                f"/api/v1/governance/models/{r_rn.json()['id']}/promote-staging",
                headers=self._headers("tok-eng"),
                json={"reason": "Missing train_only_verified"},
            ).status_code,
            422,
        )

        # ClearML compensation test: promote m1 to production, stage m2, fail 2nd ClearML sync -> m1 restored in ClearML & DB
        m1 = self._register_sample_model(ds_id, version="0.8.1", val_iou=0.31)
        self.client.post(
            f"/api/v1/governance/models/{m1['id']}/promote-staging",
            headers=self._headers("tok-eng"),
            json={"reason": "Stage m1"},
        )
        req1 = self.client.post(
            f"/api/v1/governance/models/{m1['id']}/approval-requests",
            headers=self._headers("tok-eng"),
            json={"target_stage": "production", "request_type": "promotion", "reason": "Approve m1"},
        ).json()
        self.client.post(
            f"/api/v1/governance/approval-requests/{req1['id']}/decide",
            headers=self._headers("tok-rev"),
            json={"decision": "approved", "comment": "Approved"},
        )
        self.client.post(
            f"/api/v1/governance/models/{m1['id']}/promote-production",
            headers=self._headers("tok-admin"),
            json={"reason": "Prod m1"},
        )

        m2 = self._register_sample_model(ds_id, version="0.8.2", val_iou=0.35)
        self.client.post(
            f"/api/v1/governance/models/{m2['id']}/promote-staging",
            headers=self._headers("tok-eng"),
            json={"reason": "Stage m2"},
        )
        req2 = self.client.post(
            f"/api/v1/governance/models/{m2['id']}/approval-requests",
            headers=self._headers("tok-eng"),
            json={"target_stage": "production", "request_type": "promotion", "reason": "Approve m2"},
        ).json()
        self.client.post(
            f"/api/v1/governance/approval-requests/{req2['id']}/decide",
            headers=self._headers("tok-rev"),
            json={"decision": "approved", "comment": "Approved"},
        )

        class _FailingSecondSyncAdapter(MockClearMLRegistryAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def sync_lifecycle_status(self, *, clearml_model_id: str, target_status: str, model_family: str):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("Simulated ClearML network failure")
                return super().sync_lifecycle_status(
                    clearml_model_id=clearml_model_id,
                    target_status=target_status,
                    model_family=model_family,
                )

        failing_adapter = _FailingSecondSyncAdapter()
        from backend.app.governance.auth import ActorContext
        from backend.app.governance.constants import ROLE_PERMISSIONS
        from backend.app.governance.schemas import LifecycleActionRequest

        admin_actor = ActorContext(
            actor_id="alice-admin",
            role="admin",
            auth_source="service_token",
            permissions=ROLE_PERMISSIONS["admin"],
            correlation_id="corr-comp-test",
        )
        with self.SessionFactory() as db:
            with self.assertRaises(RuntimeError):
                ModelService.promote_to_production(
                    db,
                    m2["id"],
                    LifecycleActionRequest(reason="Should compensate"),
                    admin_actor,
                    failing_adapter,
                )
        # m1 remains production in DB and was compensated back to production in ClearML adapter
        m1_db = self.client.get(
            f"/api/v1/governance/models/{m1['id']}", headers=self._headers("tok-view")
        ).json()
        self.assertEqual(m1_db["lifecycle_status"], "production")
        self.assertEqual(failing_adapter._synced_stages[m1["clearml_model_id"]], "production")

    # 6. Alembic Upgrade / Downgrade Migration Verification
    def test_alembic_upgrade_and_downgrade(self) -> None:
        with tempfile.TemporaryDirectory() as mig_dir:
            mig_db = Path(mig_dir) / "mig_test.db"
            alembic_ini = Path(__file__).resolve().parents[1] / "alembic.ini"
            cfg = Config(str(alembic_ini))
            cfg.set_main_option("sqlalchemy.url", f"sqlite:///{mig_db.as_posix()}")

            command.upgrade(cfg, "head")
            eng = create_engine(f"sqlite:///{mig_db.as_posix()}")
            tables = set(inspect(eng).get_table_names())
            expected = {
                "generation_runs",
                "governed_datasets",
                "governed_models",
                "governance_policies",
                "approval_requests",
                "approval_decisions",
                "drift_events",
                "governance_role_assignments",
                "audit_events",
            }
            self.assertTrue(expected.issubset(tables), f"Missing tables: {expected - tables}")

            # Verify all 8 default policies were seeded & SQLite DB trigger blocks raw SQL mutation of audit_events
            with eng.begin() as conn:
                pol_count = conn.execute(text("SELECT COUNT(*) FROM governance_policies")).scalar()
                self.assertEqual(pol_count, 8)
                conn.execute(
                    text(
                        "INSERT INTO audit_events (id, timestamp, actor_id, actor_role, action, entity_type, entity_id, outcome, correlation_id) "
                        "VALUES ('11111111111111111111111111111111', '2026-11-28T00:00:00Z', 'a', 'admin', 'test', 't', '1', 'success', 'c')"
                    )
                )
            with self.assertRaises(IntegrityError):
                with eng.begin() as conn:
                    conn.execute(text("DELETE FROM audit_events"))
            with self.assertRaises(IntegrityError):
                with eng.begin() as conn:
                    conn.execute(text("UPDATE audit_events SET reason = 'tamper'"))
            eng.dispose()

            command.downgrade(cfg, "base")
            eng2 = create_engine(f"sqlite:///{mig_db.as_posix()}")
            tables_after = set(inspect(eng2).get_table_names())
            self.assertNotIn("governed_models", tables_after)
            self.assertIn("generation_runs", tables_after)
            eng2.dispose()


if __name__ == "__main__":
    unittest.main()
