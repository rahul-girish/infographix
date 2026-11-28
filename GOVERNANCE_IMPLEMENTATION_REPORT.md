# Infographix MLOps Governance Module — Implementation & Team Handover Report

**Module Owner:** Rakshith (MLOps Governance)  
**Git Branch:** `feature/mlops-governance` (uncommitted working tree; not pushed or merged)  
**Date:** 2026-11-28  
**Related Docs:** [`GOVERNANCE_DECISIONS.md`](GOVERNANCE_DECISIONS.md) | [`GOVERNANCE_SPECIFICATION.md`](GOVERNANCE_SPECIFICATION.md)

---

## 5.1 Implementation Summary

### What Was Implemented & Why
An independent, modular **MLOps Governance Layer** inside `backend/app/governance/` that adds policy enforcement, lifecycle state-machine guardrails, automated shadow comparison, human approval workflows, dataset/model lineage tracking, RBAC, and append-only audit logging on top of Infographix's existing ClearML Native Model Registry (`src/registry/clearml_registry.py`), `minidvc` dataset manifests (`.mydvc/`), and FastAPI + SQLAlchemy backend (`backend/`).

### How It Works
- **Control Plane Over ClearML:** Governance does not duplicate ClearML weights/artifacts or build a competing model registry. It indexes immutable ClearML provenance (`clearml_model_id`, `checkpoint_sha256`, `clearml_training_task_id`, `artifact_uri`) in PostgreSQL/SQLite and enforces transition rules before delegating stage tag updates via `ClearMLRegistryProtocol` (`MockClearMLRegistryAdapter` offline, `LiveClearMLRegistryAdapter` when `GOV_CLEARML_MODE=clearml`).
- **Existing Poster Flows Preserved:** Existing endpoints (`/api/v1/generate`, `/api/v1/runs`, `/api/v1/export/pdf`, `/api/v1/health`) and `generation_runs` remain completely untouched.
- **Technologies Used:** Python 3.12, FastAPI, Pydantic v2 (`pydantic-settings`), SQLAlchemy 2.0 (`JSONB` on PostgreSQL / `JSON` on SQLite), Alembic migrations, stdlib HS256 JWT/HMAC verification, `unittest` + `fastapi.testclient.TestClient`.

### Files Created
- **Governance Decisions & Spec:** `GOVERNANCE_DECISIONS.md`, `GOVERNANCE_SPECIFICATION.md`, `GOVERNANCE_IMPLEMENTATION_REPORT.md`
- **Alembic Migrations:**
  - `backend/alembic.ini`
  - `backend/alembic/env.py`
  - `backend/alembic/script.py.mako`
  - `backend/alembic/versions/001_governance_foundation.py`
- **Governance Package (`backend/app/governance/`):**
  - `__init__.py`, `config.py`, `constants.py`, `exceptions.py`, `models.py`, `schemas.py`, `auth.py`, `router.py`
  - `adapters/__init__.py`, `adapters/clearml_adapter.py`, `adapters/dataset_manifest_adapter.py`, `adapters/drift_adapter.py`, `adapters/versioning_adapter.py`
  - `services/__init__.py`, `services/audit_service.py`, `services/dataset_service.py`, `services/policy_service.py`, `services/model_service.py`, `services/approval_service.py`, `services/drift_service.py`, `services/rbac_service.py`
- **Tests:**
  - `backend/tests/test_governance.py`

### Existing Files Modified
- `backend/app/models/__init__.py` — Exports all 8 governance ORM models alongside `GenerationRun` so `Base.metadata` discovers them.
- `backend/app/api/v1/router.py` — Mounts `governance_router` under `/api/v1/governance` (and `/api/governance`).
- `backend/app/main.py` — Registers `GovernanceError` JSON error envelope handler and seeds default policies on startup lifespan when `GOV_SEED_DEFAULT_POLICIES=true`.
- `backend/app/db/session.py` — Enables `PRAGMA foreign_keys=ON` on SQLite connections for referential integrity parity with PostgreSQL.
- `backend/requirements.txt` — Adds `alembic>=1.13.0` and `httpx>=0.27.0`.
- `backend/.env.example` — Documents `GOV_AUTH_MODE`, `GOV_JWT_SECRET`, `GOV_JWT_AUDIENCE`, `GOV_SERVICE_TOKENS`, `GOV_CLEARML_MODE`, `GOV_CLEARML_PROJECT_NAME`, `GOV_SEED_DEFAULT_POLICIES`.
- `.gitignore` *(pre-existing local unstaged modification preserved from workspace)*.

---

## 5.2 Governance Policies

All policies are stored in `governance_policies`, seeded via migration `001_governance_foundation` (and `PolicyService.ensure_default_policies`), and evaluated by `PolicyService.evaluate_model` & `PolicyService.compute_shadow_comparison`:

| `policy_key` | Scope | Mode | Default Config | Enforcement Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| `require_dataset_lineage` | Global | `blocking` | `{"allow_legacy_fallback": true, "legacy_versions": ["0.1.0"]}` | Blocks registration/promotion without active `dataset_id` FK (unless exempted legacy baseline). |
| `require_artifact_integrity` | Global | `blocking` | `{"require_sha256": true, "verify_clearml_bundle": true}` | Validates 64-char SHA-256 checksum & invokes `ClearMLRegistryProtocol.verify_model_artifact` on `staging`/`production`. |
| `require_reproducibility_metadata` | Global | `blocking` | `{"required_fields": ["clearml_training_task_id", "git_commit", "environment_summary"]}` | Requires training task ID, commit SHA, and runtime environment JSON. |
| `generator_quality_gates` | `poster_layout_generator` | `blocking` | `{"min_val_iou_mean": 0.25, "max_canvas_violation_rate": 0.05, "max_unintended_overlap_mean_area": 0.15}` | Enforced on `staging`/`production`; requires non-empty `metrics_summary` on `candidate`. |
| `retriever_quality_gates` | `poster_layout_retriever` | `blocking` | `{"require_train_only": true, "min_train_posters": 100}` | Enforced on `staging`/`production`; blocks anti-leakage violations (`train_only_verified=false`). |
| `production_approval_policy` | Global | `blocking` | `{"require_human_approval": true, "allow_self_approval": false, "single_active_staging": true, "single_active_production": true}` | Requires approved `ApprovalRequest` for `production`, blocks self-approval, and keeps stages unambiguous. |
| `shadow_comparison_policy` | Global | `blocking` | `{"require_comparison_when_prod_exists": true, "block_on_iou_regression": true, "max_allowed_iou_drop": 0.0}` | Compares candidate metrics vs. active `production` model; blocks approval if `val_iou_mean` regresses. |
| `drift_response_policy` | Global | `warning` | `{"block_promotion_on_critical_drift": true, "mark_model_degraded_on_critical": true}` | Marks model `non_compliant` on `critical` drift and blocks `production` promotion. |

---

## 5.3 Model Lifecycle

Stages match `src/registry/clearml_registry.py`: `("candidate", "staging", "production", "archived")`.

- **Allowed Transitions:**
  - *(New)* → `candidate`: Automated policy check on `POST /models`.
  - `candidate` → `staging`: Automated policy check (`POST /models/{id}/promote-staging`); auto-archives prior `staging` model in same family when `single_active_staging=true`.
  - `staging` → `production`: Automated policy check + approved `ApprovalRequest` (`POST /models/{id}/promote-production`); auto-archives prior `production` model in same family and sets `was_ever_production=true`.
  - `candidate` / `staging` / `production` → `archived`: Mandatory `reason` (`POST /models/{id}/archive`); archiving from `production` requires `admin`. Pending approvals are automatically `invalidated`.
  - `archived` → `production` **(Controlled Rollback):** Allowed **only** if `was_ever_production=true`, SHA-256 passes verification, `reason` is given, and either an approved `rollback` `ApprovalRequest` exists or an `admin` passes `emergency_override=true` (`POST /models/{id}/rollback`).
- **Rejected Transitions (`409 Conflict` + audited `model.transition_denied`):** `candidate` → `production`, `archived` → `candidate`, `archived` → `staging`, and same-stage no-ops.
- **Rejection & Invalidation:** Rejected `ApprovalRequest` decisions prevent promotion. Updating a `governance_policies` rule or archiving a model automatically marks `pending` requests `invalidated`.

---

## 5.4 Metadata Contracts

### Dataset Metadata (`governed_datasets`)
- **Required on `POST /datasets`:** `dataset_name` (2–128 chars), `dataset_version` (1–64 chars, unique pair), `storage_ref`, `checksum_sha256` (64-char hex).
- **Optional / Defaulted:** `source_type` (`scipostlayout` | `hitl_export` | `hybrid` | `mock`), `owner` (defaults to actor), `description`, `storage_system` (`minidvc` | `s3` | `clearml_dataset` | `local`), `data_format` (`parquet`), `schema_version`, `record_count`, `size_bytes`, `split_config`, `quality_summary`, `preprocessing_version`, `usage_policy`, `parent_dataset_id` (FK).
- **Manifest Ingestion (`POST /datasets/ingest-manifest`):** `DatasetManifestAdapter` parses `.mydvc/manifests/*.manifest` (`MINIDVC-MANIFEST 2`) and optional `feature_report.json`.

### Model Metadata (`governed_models`)
- **Immutable Provenance (set at `POST /models`):** `clearml_model_id` (unique), `model_family` (`poster_layout_generator` | `poster_layout_retriever`), `semantic_version` (`X.Y.Z`, unique per family), `architecture`, `parent_model_id` (self-FK), `clearml_training_task_id`, `artifact_uri`, `checkpoint_sha256` (64-hex), `git_commit`, `dataset_id` (FK → `governed_datasets.id`, `ON DELETE RESTRICT`), `legacy_dataset_identifier`, `retriever_model_id`, `retriever_version`, `schema_version`, `environment_summary`, `dependencies_summary`, `metrics_summary`.
- **Governance State (system/lifecycle managed):** `lifecycle_status`, `was_ever_production`, `compliance_status` (`pending` | `compliant` | `non_compliant`), `drift_status` (`unknown` | `none` | `warning` | `critical`).
- **Mutable Fields (`PATCH /models/{id}` with mandatory `reason`):** `owner`, `description`, `tags`.

---

## 5.5 Audit Logging

- **Entity:** `audit_events` (`id`, `timestamp`, `actor_id`, `actor_role`, `action`, `entity_type`, `entity_id`, `related_model_version`, `related_dataset_version`, `previous_state`, `new_state`, `reason`, `outcome` [`success` | `denied` | `failed`], `correlation_id`).
- **Supported Actions:** `dataset.registered`, `dataset.updated`, `model.registered`, `model.metadata_updated`, `model.evaluated`, `model.lifecycle_transition`, `model.promoted_to_production`, `model.rollback`, `model.archived`, `model.transition_denied`, `approval.requested`, `approval.approved`, `approval.rejected`, `approval.cancelled`, `approval.invalidated`, `policy.created`, `policy.updated`, `role.assigned`, `role.revoked`, `drift.event_recorded`, `auth.access_denied`.
- **Retrieval:** Read-only `GET /api/v1/governance/audit-events` with filters (`action`, `entity_type`, `entity_id`, `actor_id`, `outcome`, `correlation_id`, `related_model_version`, `since`, `until`, `limit`, `offset`).
- **Integrity & Access:** Zero `PUT`/`PATCH`/`DELETE` routes; SQLAlchemy `before_update` & `before_delete` hooks raise `AuditImmutabilityError`; PostgreSQL migration attaches `trg_audit_events_append_only`; denied/failed operations commit via `AuditService.record_isolated` in an independent session so transaction rollbacks never lose security/denial logs.

---

## 5.6 Roles and Permissions

Auth dependency `get_current_actor` supports **Service Tokens** (`X-Governance-Token` / `Authorization: Bearer` via `GOV_SERVICE_TOKENS`), **HS256 Bearer JWTs** (`GOV_JWT_SECRET`), and **`GOV_AUTH_MODE=dev` headers** (`X-Actor-Id`, `X-Actor-Role`, disabled in `strict` mode).

| Permission | `viewer` | `ml_engineer` | `reviewer` | `admin` |
| :--- | :---: | :---: | :---: | :---: |
| `governance:read` | ✅ | ✅ | ✅ | ✅ |
| `dataset:write`, `model:register`, `model:update_metadata`, `model:promote_staging`, `model:archive_non_prod`, `approval:request`, `drift:record` | ❌ | ✅ | ❌ | ✅ |
| `approval:decide` *(blocked if requester == decider unless `allow_self_approval=true`)* | ❌ | ❌ | ✅ | ✅ |
| `model:promote_production`, `model:rollback`, `model:archive_production`, `model:demote_production`, `policy:write`, `rbac:manage` | ❌ | ❌ | ❌ | ✅ |

---

## 5.7 Database Architecture

- **Tables Added (8):** `governed_datasets`, `governed_models`, `governance_policies`, `approval_requests`, `approval_decisions`, `drift_events`, `governance_role_assignments`, `audit_events` (coexisting alongside `generation_runs`).
- **Key Foreign Keys & Constraints:**
  - `governed_models.dataset_id` → `governed_datasets.id` (`ON DELETE RESTRICT`, `passive_deletes="all"`).
  - `approval_requests.model_id` → `governed_models.id` (`ON DELETE CASCADE`), `baseline_model_id` → `governed_models.id` (`ON DELETE SET NULL`).
  - `approval_decisions.request_id` → `approval_requests.id` (`ON DELETE CASCADE`).
  - `drift_events.model_id` → `governed_models.id` (`ON DELETE CASCADE`), `dataset_id` → `governed_datasets.id` (`ON DELETE SET NULL`), unique `monitoring_run_id`.
  - `audit_events` has **zero foreign keys** so audit history persists independently.
- **Migration:** `backend/alembic/versions/001_governance_foundation.py` (`alembic -c backend/alembic.ini upgrade head`).

---

## 5.8 API Documentation

Base prefix: `/api/v1/governance` (also aliased under `/api/governance`).  
Auth header: `X-Governance-Token: <token>` or `Authorization: Bearer <jwt-or-token>`.  
Error envelope on `401`/`403`/`404`/`409`/`422`: `{"error": {"code": "...", "message": "...", "details": {...}, "correlation_id": "..."}}`.

| Method & Route | Purpose | Permission |
| :--- | :--- | :--- |
| `GET /me` | Current actor, role, auth source & permissions | `governance:read` |
| `GET /overview` | Dashboard summary (counts, active prod/staging, pending approvals, critical drift) | `governance:read` |
| `POST /datasets` | Register dataset version | `dataset:write` |
| `POST /datasets/ingest-manifest` | Ingest `.mydvc` manifest + `feature_report.json` | `dataset:write` |
| `GET /datasets`, `GET /datasets/{dataset_id}` | List/fetch dataset metadata & linked model count | `governance:read` |
| `POST /models` | Register model version in `candidate` + evaluate policies | `model:register` |
| `GET /models`, `GET /models/{model_id}` | List/filter or inspect model governance record | `governance:read` |
| `PATCH /models/{model_id}` | Update `owner`, `description`, `tags` (with `reason`) | `model:update_metadata` |
| `POST /models/{model_id}/evaluate?target_stage=staging` | Re-run policy evaluation | `governance:read` |
| `POST /models/{model_id}/promote-staging` | Promote `candidate` → `staging` | `model:promote_staging` |
| `POST /models/{model_id}/approval-requests` | Request `production`/`rollback` approval + shadow comparison | `approval:request` |
| `GET /approval-requests`, `GET /approval-requests/{id}` | List/inspect approval requests & shadow comparison deltas | `governance:read` |
| `POST /approval-requests/{id}/decide` | Approve or reject (`approved`/`rejected`) | `approval:decide` |
| `GET /models/{model_id}/promotion-gate` | CI/CD pre-flight eligibility check | `governance:read` |
| `POST /models/{model_id}/promote-production` | Promote approved `staging` model → `production` | `model:promote_production` |
| `POST /models/{model_id}/rollback` | Controlled `archived` → `production` rollback | `model:rollback` |
| `POST /models/{model_id}/archive` | Retire model (`admin` required if `production`) | `model:archive_non_prod` |
| `POST /drift-events`, `GET /drift-events` | Ingest or list drift monitoring runs | `drift:record` / `governance:read` |
| `GET /policies`, `GET /policies/{key}`, `PATCH /policies/{key}` | View or update policy thresholds/toggles | `governance:read` / `policy:write` |
| `GET /roles`, `PUT /roles/{actor_id}`, `DELETE /roles/{actor_id}` | List, assign, or revoke actor roles | `governance:read` / `rbac:manage` |
| `GET /audit-events` | Filtered read-only audit timeline | `governance:read` |

---

## 5.9 Teammate Integration Instructions

### Rahul RVCE AI — Data Drift & Monitoring
1. **Authentication:** Use your scoped service token (e.g. `X-Governance-Token: <drift-svc-token>` mapped to `ml_engineer` or `admin` in `GOV_SERVICE_TOKENS`).
2. **Endpoint to Call:** `POST /api/v1/governance/drift-events` after each drift monitoring run.
3. **Target Identification:** Pass either `model_id` (governance UUID) OR `model_family` + `semantic_version` (e.g. `"poster_layout_generator"` + `"0.2.0"`), plus a unique `monitoring_run_id` (enforces idempotency—duplicate IDs return `409`).
4. **Governance Behavior:** Setting `drift_status: "critical"` automatically updates `model.drift_status`, triggers `drift_response_policy` (marking `compliance_status="non_compliant"` and blocking future production promotions), and logs `drift.event_recorded` in `audit_events`. Set `is_mock: false` on real monitoring runs.

### Rahul S RV AI — Model Versioning
1. **Prerequisite Dataset Reference:** Ensure the training dataset is registered via `POST /api/v1/governance/datasets` (or `POST /datasets/ingest-manifest` pointing to `.mydvc/manifests/*.manifest`) to obtain `dataset_id`.
2. **Registration Flow:** Register the immutable artifact bundle in ClearML (`src/registry/clearml_registry.py`) to obtain `clearml_model_id`, `artifact_uri`, and `checkpoint_sha256`.
3. **Sync to Governance:** Call `POST /api/v1/governance/models` with the ClearML ID, `model_family`, `semantic_version`, `clearml_training_task_id`, `git_commit`, `dataset_id`, `environment_summary`, and `metrics_summary`. Governance registers the model in `candidate` and evaluates registration policies.
4. **Lifecycle Transitions:** Never call `clearml_registry.set_model_lifecycle_status()` directly in CI/CD—call governance's `/promote-staging`, `/approval-requests`, and `/promote-production` (with `GOV_CLEARML_MODE=clearml`), which validate rules and sync ClearML automatically.

### Ram Charan RVCE AI — MLOps Dashboard
1. **Role-Aware UI:** Call `GET /api/v1/governance/me` on load to get `{actor_id, role, permissions[]}` and gate UI buttons (`Approve/Reject` only for `reviewer`/`admin` when `actor_id != requested_by`; `Policies/Roles` only for `admin`).
2. **Dashboard Views:**
   - **Overview KPIs:** `GET /api/v1/governance/overview`
   - **Models & Lineage:** `GET /api/v1/governance/models` + `GET /api/v1/governance/datasets/{dataset_id}`
   - **Approval Queue & Shadow Comparison:** `GET /api/v1/governance/approval-requests?status=pending` (render `shadow_comparison_report.metric_deltas` side-by-side before `POST /approval-requests/{id}/decide`)
   - **Drift & Policy Compliance:** `GET /api/v1/governance/drift-events` and `GET /api/v1/governance/policies`
   - **Audit Timeline:** `GET /api/v1/governance/audit-events`

---

## 5.10 Integration Contracts & Example Payloads

### 1. Model Registration (`POST /api/v1/governance/models`)
```json
{
  "clearml_model_id": "6a2c2d4897384aaf8aa59c7c66a86bfb",
  "model_family": "poster_layout_generator",
  "semantic_version": "0.2.0",
  "clearml_training_task_id": "8cd6cb755a084535a7bba3ae551e3e20",
  "artifact_uri": "https://files.clear.ml/Infographix/poster_layout_generator-v0.2.0.zip",
  "checkpoint_sha256": "510cdbba983471f1e41ecabc3270a60bc8f234b10e33824c58833197203281a7",
  "git_commit": "22ebf82",
  "dataset_id": "<governed-dataset-uuid>",
  "retriever_model_id": "5a0101a1cf6340718c7d2b85a15bd900",
  "retriever_version": "2.1.0",
  "environment_summary": {"python_version": "3.12.3", "torch_version": "2.5.0", "clearml_version": "2.2.0"},
  "metrics_summary": {
    "val_iou_mean": 0.31,
    "val_loss_total": 3.45,
    "val_canvas_violation_rate": 0.01,
    "val_unintended_overlap_mean_area": 0.04
  },
  "description": "Trained with 70/15/15 SciPostLayout v2 split"
}
```

### 2. Drift Event Ingestion (`POST /api/v1/governance/drift-events`)
```json
{
  "model_family": "poster_layout_generator",
  "semantic_version": "0.2.0",
  "monitoring_run_id": "drift-run-2026-11-28-001",
  "monitoring_window": "2026-11-27T00:00Z/24h",
  "drift_status": "critical",
  "drift_metrics": {"psi_element_count": 0.28, "ks_aspect_ratio_pvalue": 0.004},
  "threshold_config": {"psi_warning": 0.10, "psi_critical": 0.25},
  "detected_at": "2026-11-28T05:00:00Z",
  "is_mock": false
}
```

### 3. Approval Decision (`POST /api/v1/governance/approval-requests/{request_id}/decide`)
```json
{
  "decision": "approved",
  "comment": "Shadow comparison confirms +0.06 val_iou_mean without overlap regression.",
  "policy_override": false
}
```

### Mock-to-Real Replacement Instructions
- **ClearML Registry:** Set `GOV_CLEARML_MODE=clearml` and ensure `CLEARML_API_ACCESS_KEY`/`CLEARML_API_SECRET_KEY` are configured in `backend/.env`; `get_clearml_adapter()` automatically switches from `MockClearMLRegistryAdapter` to `LiveClearMLRegistryAdapter`.
- **Drift & Versioning Modules:** `MockDriftAdapter` and `MockVersioningAdapter` are isolated helper fixtures used in tests/demos—real modules simply HTTP-POST directly to `/api/v1/governance/drift-events` and `/api/v1/governance/models` using scoped `GOV_SERVICE_TOKENS`.

---

## 5.11 Completion Status

| Component | Status | Notes |
| :--- | :--- | :--- |
| Governance Policies & Evaluator | **Completed** | 8 DB-backed policies + `PolicyEvaluator` + tests |
| Lifecycle State Machine & Rollback | **Completed** | 4-stage FSM, stage uniqueness, controlled rollback |
| Automated Shadow Comparison & Approvals | **Completed** | Delta calculation, regression block, separation of duties, invalidation |
| Dataset & Model Metadata Catalog | **Completed** | Schema, validation, `.mydvc` manifest ingestion |
| Append-Only Audit Logging | **Completed** | `AuditService`, PG trigger, isolated denied-action commits |
| Pluggable Auth & 4-Role RBAC | **Completed** | Service Tokens, HS256 JWT, `dev` mode, role bindings |
| Database Schema & Alembic Migrations | **Completed** | `001_governance_foundation.py` upgrade & downgrade tested |
| ClearML Registry Integration | **Pending External Integration** | Verified via `MockClearMLRegistryAdapter`; `LiveClearMLRegistryAdapter` wired for `GOV_CLEARML_MODE=clearml` |
| Data Drift Detection (Rahul) | **Pending External Integration** | Endpoint & `MockDriftAdapter` ready; real detector out of scope |
| Model Versioning (Rahul S) | **Pending External Integration** | Contract & `MockVersioningAdapter` ready; teammate module out of scope |
| MLOps Dashboard (Ram Charan) | **Pending External Integration** | Read/write APIs ready; frontend UI out of scope |
| CI/CD Workflow Wiring (`.github/workflows/model-cd-promote.yml`) | **Deferred by Design** | Existing workflows left untouched; integration instructions provided above |

---

## 5.12 Git and Change Summary

- **Active Branch:** `feature/mlops-governance`
- **Database Migrations:** `backend/alembic/versions/001_governance_foundation.py` (`upgrade -> head` and `downgrade -> base` verified on SQLite; PostgreSQL trigger guarded by `dialect.name == "postgresql"`).
- **Tests Executed:**
  - Command: `PYDANTIC_DISABLE_PLUGINS=__all__ PYTHONPATH=/home/raksh/projects/infographix .venv/bin/python -u -c "..."` running `backend.tests.test_governance` (7 test methods) and `backend.tests.test_generation_mlops` (4 test methods).
  - Result: **`Ran 11 tests in 212.259s — OK` (11/11 passed, 0 failures, 0 errors, 0 skipped).**
- **Known Limitations:**
  - PostgreSQL `BEFORE UPDATE OR DELETE` trigger syntax in `001_governance_foundation.py` is dialect-guarded and was not executed against a live Supabase PostgreSQL instance in offline tests (SQLite ORM hooks were verified instead).
  - `LiveClearMLRegistryAdapter` requires live ClearML credentials and was not exercised against ClearML Cloud during offline tests.
- **Remaining Tasks (Post-Handover):**
  - Run `alembic -c backend/alembic.ini upgrade head` against Supabase PostgreSQL when deploying.
  - Set `GOV_SERVICE_TOKENS` and `GOV_JWT_SECRET` in production `backend/.env`.
  - Wire teammates' modules (Rahul, Rahul S, Ram Charan) and `model-cd-promote.yml` using Section 5.9–5.10 contracts.
  - Commit and push `feature/mlops-governance` once you review the diff.
