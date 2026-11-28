# Infographix MLOps Governance Module — Architecture & Implementation Specification (Phase 3)

**Status:** Draft for Approval (Phase 3)  
**Branch:** `feature/mlops-governance`  
**Owner:** Rakshith (MLOps Governance)  
**Date:** 2026-11-28  
**Decisions Reference:** [`GOVERNANCE_DECISIONS.md`](GOVERNANCE_DECISIONS.md)

---

## 1. Governance Architecture

The Infographix MLOps Governance Module acts as a **modular control, policy, approval, and audit plane** inside the existing FastAPI application (`backend/`), coexisting cleanly with `generation_runs` without duplicating ClearML (`src/registry/clearml_registry.py`), `minidvc` (`.mydvc/`), or `inference_service/`.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                              External Callers                              │
│  • Ram Charan's MLOps Dashboard (UI)                                       │
│  • Rahul S's Model Versioning Module / model-train-register.yml            │
│  • Rahul's Data Drift Monitoring Service                                   │
│  • CI/CD Promotion Pipeline (model-cd-promote.yml)                         │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │ HTTPS + Bearer JWT / Service Token
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│               FastAPI Governance Router (/api/v1/governance/*)             │
│  • Auth & RBAC Dependency (get_current_actor, require_permission)          │
│  • Consistent JSON Error & Response Envelopes                              │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌─────────────────┐     ┌────────────────────┐
│ Policy & Gate │      │   Lifecycle &   │     │   Catalog, Drift   │
│  Evaluator    │      │ Approval Engine │     │  & Audit Services  │
└───────┬───────┘      └────────┬────────┘     └─────────┬──────────┘
        │                       │                        │
        └───────────────────────┼────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌─────────────────────────────────┐     ┌────────────────────────────────────┐
│  PostgreSQL (Supabase) / SQLite │     │     External / Mock Adapters       │
│  • governed_datasets            │     │  • ClearMLRegistryAdapter (mock/   │
│  • governed_models              │     │    real src.registry.clearml_reg)  │
│  • governance_policies          │     │  • MinidvcManifestAdapter          │
│  • approval_requests/decisions  │     │  • DriftIntegrationAdapter (mock)  │
│  • drift_events                 │     │  • VersioningSyncAdapter (mock)    │
│  • governance_role_assignments  │     │                                    │
│  • audit_events (append-only)   │     │                                    │
└─────────────────────────────────┘     └────────────────────────────────────┘
```

### Proposed Module Layout (`backend/app/governance/`)
Keeping governance isolated inside `backend/app/governance/` ensures zero interference with existing poster generation routes (`backend/app/api/v1/generation.py`, `backend/app/services/layout_pipeline.py`) and clean Alembic integration:

```text
backend/
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/
│       └── 001_add_governance_tables.py
└── app/
    ├── api/v1/router.py                    # Mounts governance_router under /governance
    └── governance/
        ├── __init__.py
        ├── config.py                       # Governance-specific env settings (auth mode, tokens, ClearML sync)
        ├── constants.py                    # Lifecycle stages, transitions, roles, permissions, action enums
        ├── auth.py                         # get_current_actor, require_permission (JWT / Service Token / Dev)
        ├── exceptions.py                   # Structured domain exceptions -> HTTP error handlers
        ├── models.py                       # 8 SQLAlchemy ORM models sharing backend.app.db.base.Base
        ├── schemas.py                      # Versioned Pydantic request/response contracts & error envelope
        ├── adapters/
        │   ├── __init__.py
        │   ├── clearml_adapter.py          # Protocol + MockClearMLAdapter + ClearMLRegistryAdapter
        │   ├── dataset_manifest_adapter.py # Reads .mydvc/manifests/*.manifest & feature_report.json
        │   ├── drift_adapter.py            # Mock/interface boundary for Rahul's drift module
        │   └── versioning_adapter.py       # Mock/interface boundary for Rahul S's versioning module
        ├── services/
        │   ├── __init__.py
        │   ├── audit_service.py            # Append-only AuditService (independent denied-event commit)
        │   ├── dataset_service.py          # Dataset catalog registration, manifest ingestion, lookup
        │   ├── policy_service.py           # Policy CRUD, seeding, PolicyEvaluator, shadow comparison
        │   ├── model_service.py            # Model registration, metadata updates, lifecycle transitions
        │   ├── approval_service.py         # Approval requests, decisions, invalidation, separation of duties
        │   ├── drift_service.py            # Drift event ingestion & policy reaction evaluation
        │   └── rbac_service.py             # Role assignment CRUD & /me inspection
        └── router.py                       # /api/v1/governance/* FastAPI routes
```

---

## 2. Functional Requirements

| ID | Requirement |
| :--- | :--- |
| **FR-01** | **Dataset Catalog:** Register, inspect, and deactivate governed dataset versions with storage references (`minidvc`, `s3`, `clearml_dataset`), SHA-256 checksums, split/quality summaries, and lineage (`parent_dataset_id`). Support parsing `.mydvc/manifests/*.manifest` + `feature_report.json`. |
| **FR-02** | **Model Governance Registration:** Register models from ClearML/Rahul S into `candidate` with immutable provenance (`clearml_model_id`, `(model_family, semantic_version)`, `checkpoint_sha256`, `clearml_training_task_id`, `git_commit`, `dataset_id` FK). |
| **FR-03** | **Policy Evaluation Engine:** Maintain DB-backed configurable policies (`governance_policies`) and evaluate models against family-specific and global policies on registration, staging promotion, and production approval. |
| **FR-04** | **Lifecycle State Machine:** Enforce strict transitions across `("candidate", "staging", "production", "archived")`, blocking skipped stages (`candidate` → `production`), enforcing single active `production` (and configurable single active `staging`) per family, and supporting controlled `archived` → `production` rollback. |
| **FR-05** | **Automated Shadow Comparison:** Automatically generate/validate a comparison report comparing a candidate/staging model's metrics (and optional shadow evaluation payload) against the active `production` model in the same family prior to `production` approval. |
| **FR-06** | **Human Approval Workflow:** Support `ApprovalRequest` creation, automated pre-check attachment, reviewer `approved`/`rejected` decisions, self-approval prevention (`allow_self_approval=False` default), and automatic invalidation of stale pending requests. |
| **FR-07** | **Promotion vs. Deployment Separation:** Provide a dedicated `can-promote` / `promote` gate API for `model-cd-promote.yml` so governance approval & lifecycle transition remain decoupled from Render infrastructure deployment. |
| **FR-08** | **Append-Only Audit Trail:** Record every governance mutation, lifecycle transition, policy change, role assignment, drift event, and **denied/failed attempt** in `audit_events` with `correlation_id`. |
| **FR-09** | **RBAC & Pluggable Auth:** Enforce 4 roles (`admin`, `ml_engineer`, `reviewer`, `viewer`) across `/api/v1/governance/*` using Bearer JWT, Service API Tokens, or dev/test mode without altering public poster endpoints. |
| **FR-10** | **Teammate Integration & Mocks:** Expose drift ingestion (`POST /drift-events`), model sync/registration contracts, and dashboard summary/list endpoints backed by identifiable mock adapters where external modules are pending. |

---

## 3. Non-Functional Requirements

| ID | Requirement | Target / Implementation |
| :--- | :--- | :--- |
| **NFR-01** | **Database Portability** | Full compatibility with PostgreSQL (Supabase `JSONB`, PG triggers) and SQLite (`JSON`, Python-level append-only guard) for offline CI/tests. |
| **NFR-02** | **Migration-Driven Schema** | All governance schema and seed changes managed via Alembic migrations; safe coexistence with existing `generation_runs`. |
| **NFR-03** | **Audit Integrity** | Zero `PUT`/`PATCH`/`DELETE` endpoints or ORM mutation helpers on `audit_events`; PostgreSQL `BEFORE UPDATE OR DELETE` trigger; failed/denied events committed independently of rolled-back transactions. |
| **NFR-04** | **Fail-Safe Enforcement** | Missing credentials, unknown tokens, invalid transitions, missing mandatory metadata, or failed policy gates fail closed with explicit `401`/`403`/`409`/`422` errors. |
| **NFR-05** | **ClearML Independence in Tests** | All governance logic and tests execute cleanly offline without ClearML network credentials via `ClearMLRegistryAdapter` protocol + mock adapter. |
| **NFR-06** | **Secret Hygiene** | JWT secrets, service tokens, and ClearML credentials loaded strictly from environment variables (`backend/.env`), never committed to source control. |
| **NFR-07** | **Traceability** | Every API response error and audit event includes a `correlation_id` (`X-Correlation-ID` header or generated UUID). |

---

## 4. Governance Policies (Seeded Defaults)

Seeded into `governance_policies` via migration/bootstrap and editable only by `admin`:

| `policy_key` | Scope (`model_family`) | Enforcement | Default `config` | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `require_dataset_lineage` | `null` (global) | `blocking` | `{"allow_legacy_fallback": true, "legacy_versions": ["0.1.0"]}` | Requires valid `dataset_id` FK to active `governed_datasets` record. |
| `require_artifact_integrity` | `null` (global) | `blocking` | `{"require_sha256": true, "verify_clearml_bundle": true}` | Requires `checkpoint_sha256` and ClearML bundle verification before promotion. |
| `require_reproducibility_metadata` | `null` (global) | `blocking` | `{"required_fields": ["clearml_training_task_id", "git_commit", "environment_summary"]}` | Ensures experiment provenance is complete. |
| `generator_quality_gates` | `poster_layout_generator` | `blocking` | `{"min_val_iou_mean": 0.25, "max_canvas_violation_rate": 0.05, "max_unintended_overlap_mean_area": 0.15}` | Enforces layout quality thresholds aligned with `model-train-register.yml`. |
| `retriever_quality_gates` | `poster_layout_retriever` | `blocking` | `{"require_train_only": true, "min_train_posters": 100}` | Prevents train/dev/test leakage and empty retrievers. |
| `production_approval_policy` | `null` (global) | `blocking` | `{"require_human_approval": true, "allow_self_approval": false, "single_active_staging": true, "single_active_production": true}` | Controls human sign-off, separation of duties, and stage uniqueness. |
| `shadow_comparison_policy` | `null` (global) | `blocking` | `{"require_comparison_when_prod_exists": true, "block_on_iou_regression": true, "max_allowed_iou_drop": 0.0}` | Requires automated shadow comparison vs. active production model before approval. |
| `drift_response_policy` | `null` (global) | `warning` | `{"block_promotion_on_critical_drift": true, "mark_model_degraded_on_critical": true}` | Controls how incoming drift alerts affect compliance/promotion eligibility. |

---

## 5. Model Lifecycle State-Transition Rules

Authoritative stages match `src/registry/clearml_registry.py`: `("candidate", "staging", "production", "archived")`.

| From Stage | To Stage | Allowed | Preconditions | Auto Side Effects |
| :--- | :--- | :---: | :--- | :--- |
| *(New)* | `candidate` | ✅ | Provenance + dataset lineage + SHA-256 + metrics present | Creates `governed_models` row; audits `model.registered` |
| `candidate` | `staging` | ✅ | Automated policies pass (`PolicyEvaluator`); `reason` provided | If `single_active_staging=true`, archives prior `staging` model in same family; syncs ClearML tag; audits |
| `staging` | `production` | ✅ | Automated policies pass + `ApprovalRequest` is `approved` + no unresolved critical drift block + `reason` | Archives prior `production` model in same family; sets `was_ever_production=true`; syncs ClearML tag; marks approval consumed |
| `production` | `staging` | ✅ | Emergency demotion; `admin` role + mandatory `reason` | Syncs ClearML tag; audits `model.lifecycle_transition` |
| `candidate` / `staging` | `archived` | ✅ | Mandatory `reason` | Invalidates any pending approval requests; syncs ClearML tag; audits `model.archived` |
| `production` | `archived` | ✅ | `admin` role + mandatory `reason` | Syncs ClearML tag; audits `model.archived` |
| `archived` | `production` | ✅ | **Rollback only:** `was_ever_production == true` + SHA-256 valid + mandatory `reason` + (approved rollback request OR `admin` emergency override) | Archives current `production` model in same family; syncs ClearML tag; audits `model.rollback` |
| `candidate` | `production` | ❌ | Rejected (`409 Conflict`: cannot skip `staging`) | Audits `model.transition_denied` |
| `archived` | `candidate` / `staging` | ❌ | Rejected (`409 Conflict`: retired versions cannot return to candidate/staging) | Audits `model.transition_denied` |
| Same → Same | Same | ❌ | Rejected (`409 Conflict`: no-op transition) | — |

---

## 6. Approval and Promotion Criteria

### 6.1 Four Decoupled Steps
1. **Automated Validation (`POST /models/{id}/evaluate`):** Runs `PolicyEvaluator` rules + SHA-256 adapter verification; updates `compliance_status` (`compliant` | `non_compliant`).
2. **Human Approval (`POST /models/{id}/approval-requests` → `POST /approval-requests/{req_id}/decide`):**
   * On request creation for `target_stage="production"` (requires model in `staging`, or `archived` with `was_ever_production=true` for rollback):
     * Runs automated policy checks.
     * Runs **Automated Shadow Comparison** if an active `production` model exists in the same `model_family`:
       * Compares `metrics_summary` deltas (e.g., `val_iou_mean`, `val_loss_total`, `val_canvas_violation_rate`, `val_unintended_overlap_mean_area` for generator; retrieval metrics for retriever) and optionally incorporates supplied offline/shadow evaluation summary (`shadow_eval_payload`).
       * Produces `shadow_comparison_report`: `{baseline_model_id, baseline_version, metric_deltas, regression_detected, regression_reasons, summary}`.
     * Blocks request creation (or marks `automated_check_passed=false`) if blocking policies or `block_on_iou_regression` fail (unless an explicit rollback/override request).
   * Reviewer (`reviewer` or `admin`, where `decided_by != requested_by` unless `allow_self_approval=true`) records `approved` or `rejected` with `comment`.
   * **Invalidation:** Any metadata change affecting metrics/checksum, policy threshold edit, or model archival automatically transitions `pending` requests to `invalidated` (audited as `approval.invalidated`).
3. **Model Promotion (`POST /models/{id}/promote`):** Verifies approved `ApprovalRequest` exists for the target model, transitions lifecycle stage in Governance DB, and calls `ClearMLRegistryAdapter.set_lifecycle_status()`.
4. **Production Deployment (CI/CD):** `model-cd-promote.yml` calls `GET /models/{id}/promotion-gate` (or `POST /models/{id}/promote`), then updates Render `NEURAL_MODEL_ID` / `RETRIEVER_MODEL_ID` and polls `/health/ready`.

---

## 7. Dataset Metadata Specification

Stored in `governed_datasets` without duplicating raw Parquet files or full `minidvc` file lists:

| Field | Type | Required | Validation / Notes |
| :--- | :--- | :---: | :--- |
| `id` | `UUID` | Auto | Primary key |
| `dataset_name` | `str` (`<=128`) | Yes | Slug/identifier (e.g. `scipostlayout-features`) |
| `dataset_version` | `str` (`<=64`) | Yes | Version string (e.g. `2.0.0`); unique `(dataset_name, dataset_version)` |
| `source_type` | `enum` | Yes | `scipostlayout` \| `hitl_export` \| `hybrid` \| `mock` |
| `owner` | `str` (`<=128`) | Yes | Team/owner identifier |
| `description` | `text` | No | Human summary |
| `storage_system` | `enum` | Yes | `minidvc` \| `s3` \| `clearml_dataset` \| `local` |
| `storage_ref` | `str` (`<=512`) | Yes | e.g. `minidvc:v3` or `s3://bucket/key` |
| `checksum_sha256` | `str` (`64` hex) | Yes | Manifest or primary dataset SHA-256 |
| `data_format` | `str` (`<=32`) | Yes | Default `parquet` |
| `schema_version` | `str` (`<=32`) | Yes | Matches `feature_schema.json` (e.g. `2.0.0`) |
| `record_count` | `int` (`>=0`) | Yes | Poster/annotation count |
| `size_bytes` | `int` (`>=0`) | Yes | Total byte size from manifest |
| `split_config` | `dict` | Yes | e.g. `{"train": 0.7, "dev": 0.15, "test": 0.15, "random_state": 42, "stratify_by_orientation": true}` |
| `quality_summary` | `dict` | No | Compact snapshot from `feature_report.json` |
| `preprocessing_version` | `str` (`<=32`) | Yes | Pipeline version from `configs/pipeline.yaml` |
| `usage_policy` | `str` (`<=255`) | Yes | Licensing / internal usage terms |
| `parent_dataset_id` | `UUID` | No | Optional FK to parent dataset (for HITL-augmented versions) |
| `is_active` | `bool` | Yes | Default `true` |

**Manifest Ingestion Helper:** `DatasetManifestAdapter.parse_minidvc_and_report(manifest_path, report_path)` parses `.mydvc/manifests/v*.manifest` (`MINIDVC-MANIFEST 2` header, file count, total bytes, deterministic SHA-256 of manifest entries) and `feature_report.json`.

---

## 8. Model Metadata Specification

Stored in `governed_models`, indexing ClearML metadata while holding governance-exclusive fields:

| Field | Type | Mutable Post-Reg? | Notes |
| :--- | :--- | :---: | :--- |
| `id` | `UUID` | No | Governance PK |
| `clearml_model_id` | `str` (`<=128`) | No | Unique ClearML model ID |
| `model_name` | `str` (`<=192`) | No | Canonical `{model_family}-v{semantic_version}` |
| `model_family` | `enum` | No | `poster_layout_generator` \| `poster_layout_retriever` |
| `semantic_version` | `str` (`<=32`) | No | Semver `X.Y.Z`; unique `(model_family, semantic_version)` |
| `architecture` | `str` (`<=128`) | No | `TemplateConditionedLayoutModel` \| `LayoutRetriever` |
| `parent_model_id` | `UUID` | No | Optional FK to prior `governed_models.id` |
| `clearml_training_task_id` | `str` (`<=128`) | No | ClearML Task ID |
| `artifact_uri` | `str` (`<=512`) | No | ClearML Cloud zip bundle URI |
| `checkpoint_sha256` | `str` (`64` hex) | No | SHA-256 of `model_weights.pt` or `retriever.joblib` |
| `git_commit` | `str` (`<=64`) | No | Commit SHA (7–40 chars) |
| `dataset_id` | `UUID` | No* | FK → `governed_datasets.id` (*nullable only if legacy fallback enabled) |
| `legacy_dataset_identifier` | `str` (`<=255`) | No | Fallback string (e.g. `"SciPostLayout v2.0.0"`) |
| `retriever_model_id` | `str` (`<=128`) | No | Conditioning retriever ClearML ID (for `poster_layout_generator`) |
| `retriever_version` | `str` (`<=32`) | No | Conditioning retriever semver |
| `schema_version` | `str` (`<=32`) | No | `v1.0.0` / `v2.0.0` |
| `environment_summary` | `dict` | No | Python, torch/sklearn, clearml, numpy versions |
| `dependencies_summary` | `dict` | No | Compact dependency metadata (e.g., `n_train_posters`, `train_poster_ids_sha256`) |
| `metrics_summary` | `dict` | No* | Key evaluation scalars (*immutable once approved/in staging+) |
| `lifecycle_status` | `enum` | Transition API only | `candidate` \| `staging` \| `production` \| `archived` |
| `was_ever_production` | `bool` | System-managed | Set `true` upon entering `production` |
| `compliance_status` | `enum` | System-managed | `pending` \| `compliant` \| `non_compliant` |
| `drift_status` | `enum` | Drift API only | `unknown` \| `none` \| `warning` \| `critical` |
| `owner` | `str` (`<=128`) | Yes (`admin`/`ml_engineer`) | Registered owner/author |
| `description` | `text` | Yes (`admin`/`ml_engineer`) | Notes/changelog (edits audited) |
| `tags` | `list[str]` | Yes (`admin`/`ml_engineer`) | Governance tags (edits audited) |
| `registered_at` / `updated_at` | `datetime` | System-managed | UTC timestamps |

---

## 9. Audit Event Specification

* **Table:** `audit_events` (append-only, zero FKs, PG `BEFORE UPDATE OR DELETE` trigger).
* **Supported Actions:**
  `dataset.registered`, `dataset.updated`, `model.registered`, `model.metadata_updated`, `model.lifecycle_transition`, `model.promoted_to_production`, `model.rollback`, `model.archived`, `model.transition_denied`, `approval.requested`, `approval.approved`, `approval.rejected`, `approval.cancelled`, `approval.invalidated`, `policy.created`, `policy.updated`, `role.assigned`, `role.revoked`, `drift.event_recorded`.
* **Fields:** `id` (`UUID`), `timestamp` (`UTC`), `actor_id`, `actor_role`, `action`, `entity_type`, `entity_id`, `related_model_version`, `related_dataset_version`, `previous_state` (`JSON`), `new_state` (`JSON`), `reason`, `outcome` (`success` | `denied` | `failed`), `correlation_id`.
* **Denied-Action Guarantee:** RBAC denials (`403`) and illegal transition/policy rejections (`409`/`422`) commit an audit entry via an isolated session/transaction so rollback of the main request never erases the audit log.

---

## 10. Role-Based Access Control (RBAC) Matrix

### 10.1 Roles & Permissions

| Permission / Endpoint Action | `viewer` | `ml_engineer` | `reviewer` | `admin` |
| :--- | :---: | :---: | :---: | :---: |
| `governance:read` (list/get models, datasets, policies, approvals, drift, audit, `/me`) | ✅ | ✅ | ✅ | ✅ |
| `dataset:write` (register / update governed datasets) | ❌ | ✅ | ❌ | ✅ |
| `model:register` (register model into `candidate`) | ❌ | ✅ | ❌ | ✅ |
| `model:update_metadata` (update `description`, `tags`, `owner`) | ❌ | ✅ | ❌ | ✅ |
| `model:promote_staging` (`candidate` → `staging`) | ❌ | ✅ | ❌ | ✅ |
| `model:archive_non_prod` (`candidate`/`staging` → `archived`) | ❌ | ✅ | ❌ | ✅ |
| `approval:request` (submit approval request for `production` / `rollback`) | ❌ | ✅ | ❌ | ✅ |
| `approval:decide` (approve/reject request; blocked if `actor_id == requested_by` unless self-approval enabled) | ❌ | ❌ | ✅ | ✅ |
| `model:promote_production` (`staging` → `production` for approved models) | ❌ | ❌ | ❌ | ✅ |
| `model:rollback` / `model:archive_production` / `model:demote_production` | ❌ | ❌ | ❌ | ✅ |
| `drift:record` (ingest drift events from monitoring service) | ❌ | ✅ | ❌ | ✅ |
| `policy:write` & `rbac:manage` (create/update policies, assign/revoke roles) | ❌ | ❌ | ❌ | ✅ |

### 10.2 Authentication Resolution (`get_current_actor`)
Applied **only** to `/api/v1/governance/*`:
1. **Service Token (`Authorization: Bearer <token>` or `X-Governance-Token: <token>`):** Checked against env-configured JSON/mapping `GOV_SERVICE_TOKENS` (e.g. `ci-promote:admin`, `ci-train:ml_engineer`, `drift-svc:ml_engineer`, `dashboard-viewer:viewer`).
2. **Bearer JWT:** Validated with `GOV_JWT_SECRET` (HS256, Supabase-compatible `sub` + optional `role` claim); effective role resolved from `governance_role_assignments` table (falling back to JWT claim or `viewer` if configured).
3. **Dev/Test Mode (`GOV_AUTH_MODE=dev` only; rejected when `GOV_AUTH_MODE=strict`):** Reads `X-Actor-Id` and `X-Actor-Role` headers (or `governance_role_assignments`) for local Swagger & pytest fixtures.
4. **Missing/Invalid Credentials:** Always returns `401 Unauthorized` (fails closed).

---

## 11. Database Schema and Relationships

All 8 tables inherit from `backend.app.db.base.Base` and use `JSONVariant = JSON().with_variant(JSONB, "postgresql")`:

1. **`governed_datasets`**: PK `id` (`Uuid`), FK `parent_dataset_id` → `governed_datasets.id` (`SET NULL`), Unique `(dataset_name, dataset_version)`.
2. **`governed_models`**: PK `id` (`Uuid`), FK `dataset_id` → `governed_datasets.id` (`RESTRICT`, nullable for legacy backfill), FK `parent_model_id` → `governed_models.id` (`SET NULL`), Unique `clearml_model_id`, Unique `(model_family, semantic_version)`, Index `(model_family, lifecycle_status)`.
3. **`governance_policies`**: PK `id` (`Uuid`), Unique `policy_key`, Index `model_family`.
4. **`approval_requests`**: PK `id` (`Uuid`), FK `model_id` → `governed_models.id` (`CASCADE`), FK `baseline_model_id` → `governed_models.id` (`SET NULL`), Index `(model_id, status)`.
5. **`approval_decisions`**: PK `id` (`Uuid`), FK `request_id` → `approval_requests.id` (`CASCADE`), Index `request_id`.
6. **`drift_events`**: PK `id` (`Uuid`), FK `model_id` → `governed_models.id` (`CASCADE`), FK `dataset_id` → `governed_datasets.id` (`SET NULL`), Unique `monitoring_run_id`, Index `(model_id, detected_at)`.
7. **`governance_role_assignments`**: PK `id` (`Uuid`), Unique `actor_id`.
8. **`audit_events`**: PK `id` (`Uuid`), no FKs, Indexes on `timestamp`, `action`, `(entity_type, entity_id)`, `correlation_id`; PostgreSQL trigger `trg_audit_events_append_only` blocking `UPDATE` and `DELETE`.

**Migration Strategy (Alembic):**
* `backend/alembic/env.py` imports `Base.metadata` (covering both `GenerationRun` and governance models) and reads `settings.DATABASE_URL`.
* Migration `001_governance_foundation.py` creates the 8 governance tables (checking/creating `generation_runs` only if absent on fresh DBs so existing Supabase deployments do not conflict), attaches the PG trigger conditionally (`dialect.name == "postgresql"`), and seeds the 8 default `governance_policies`.

---

## 12. API Contracts (`/api/v1/governance`)

All routes also mount under `/api/governance` (matching `backend/app/main.py` prefix mounting) and return a consistent error envelope:

```json
{
  "error": {
    "code": "POLICY_VIOLATION",
    "message": "Model val_iou_mean (0.18) is below required threshold (0.25).",
    "details": {"policy_key": "generator_quality_gates"},
    "correlation_id": "9f8c2a1e-..."
  }
}
```

### Endpoints Summary

| Method & Route | Purpose | Permission | Key Request / Response & Validation |
| :--- | :--- | :--- | :--- |
| `GET /me` | Current actor, role & effective permissions | `governance:read` | Returns `{actor_id, role, auth_source, permissions[]}` |
| `GET /overview` | Dashboard summary counts & compliance snapshot | `governance:read` | Returns active prod/staging per family, pending approvals, compliance & drift summary |
| `POST /datasets` | Register governed dataset version | `dataset:write` | `DatasetCreateRequest` → `201 DatasetOut`; rejects duplicate `(name, version)` (`409`) |
| `POST /datasets/ingest-manifest` | Parse `.mydvc` manifest + `feature_report.json` | `dataset:write` | Validates paths/checksums → `201 DatasetOut` |
| `GET /datasets` / `GET /datasets/{id}` | List or fetch dataset + linked model count | `governance:read` | Paginated list / detail `DatasetOut` |
| `POST /models` | Register model in `candidate` (with optional ClearML/mock sync) | `model:register` | `ModelRegisterRequest` → evaluates `candidate` policies → `201 GovernedModelOut`; rejects duplicate semver/clearml_id (`409`) or missing lineage (`422`) |
| `GET /models` / `GET /models/{id}` | List/filter models or fetch full detail + lineage | `governance:read` | Filter by `model_family`, `lifecycle_status`, `compliance_status`, `drift_status` |
| `PATCH /models/{id}` | Update mutable fields (`description`, `tags`, `owner`) | `model:update_metadata` | Rejects immutable field edits (`422`); audits diff |
| `POST /models/{id}/evaluate` | Re-run `PolicyEvaluator` + SHA-256 verification | `governance:read` / `model:register` | Returns `PolicyEvaluationOut` (`passed`, `results[]`) |
| `POST /models/{id}/promote-staging` | Transition `candidate` → `staging` | `model:promote_staging` | Validates policies + ClearML adapter; auto-archives prior staging if policy set |
| `POST /models/{id}/approval-requests` | Request `production` promotion or `rollback` approval | `approval:request` | Runs policies + **automated shadow comparison vs current prod** → `201 ApprovalRequestOut` |
| `GET /approval-requests` / `GET /approval-requests/{req_id}` | List/inspect approval requests + shadow comparison | `governance:read` | Filter by `status`, `model_id` |
| `POST /approval-requests/{req_id}/decide` | Submit `approved` or `rejected` decision | `approval:decide` | Blocks self-approval (`403`), requires `pending` (`409`) + `comment` (`422`) |
| `GET /models/{id}/promotion-gate` | Read-only CI/CD pre-flight check before deploy | `governance:read` | Returns `{eligible: bool, checks[], approved_request_id}` |
| `POST /models/{id}/promote-production` | Execute approved `staging` → `production` transition | `model:promote_production` | Verifies active approval + policies + ClearML sync; archives prior prod |
| `POST /models/{id}/rollback` | Controlled `archived` → `production` rollback | `model:rollback` | Requires `was_ever_production=true`, SHA-256 valid, `reason`, approval or `admin` override |
| `POST /models/{id}/archive` | Retire `candidate`/`staging`/`production` model | `model:archive_non_prod` (`admin` for prod) | Requires `reason`; invalidates pending approvals |
| `POST /drift-events` | Ingest drift detection run (from Rahul / mock) | `drift:record` | Idempotent on `monitoring_run_id`; updates `model.drift_status`; applies `drift_response_policy`; audits |
| `GET /drift-events` | Query drift history by model/status | `governance:read` | Paginated `DriftEventOut[]` |
| `GET /policies` / `PATCH /policies/{key}` | List policies or update thresholds/toggles | `governance:read` / `policy:write` | Validates config schema; invalidates impacted pending approvals; audits |
| `GET /roles` / `PUT /roles/{actor_id}` / `DELETE /roles/{actor_id}` | Manage actor role assignments | `governance:read` / `rbac:manage` | Audits `role.assigned` / `role.revoked` |
| `GET /audit-events` | Read-only filtered & paginated audit timeline | `governance:read` | Filter by `action`, `entity_type`, `entity_id`, `actor_id`, `outcome`, `correlation_id`, date range |

---

## 13. Integration Boundaries (Teammates & External Systems)

1. **Rahul RVCE AI — Data Drift & Monitoring:**
   * **Contract:** Calls `POST /api/v1/governance/drift-events` with `{model_id (or model_family + semantic_version), dataset_id (optional), monitoring_run_id, monitoring_window, drift_status, drift_metrics, threshold_config, detected_at}`.
   * **Mock in Phase 4:** `MockDriftAdapter` + mock sample payloads flagged `"is_mock": true` for testing drift ingestion & policy reactions.
2. **Rahul S RV AI — Model Versioning:**
   * **Contract:** After registering an immutable bundle in ClearML (`src/registry/clearml_registry.py`), calls `POST /api/v1/governance/models` with `{clearml_model_id, model_family, semantic_version, architecture, clearml_training_task_id, artifact_uri, checkpoint_sha256, git_commit, dataset_id, retriever_model_id, environment_summary, metrics_summary, owner}`.
   * **Mock in Phase 4:** `MockVersioningAdapter` & `MockClearMLAdapter` implementing the `ClearMLRegistryProtocol` so tests run without ClearML Cloud credentials.
3. **Ram Charan RVCE AI — MLOps Dashboard:**
   * **Contract:** Consumes `GET /me`, `GET /overview`, `GET /models`, `GET /datasets`, `GET /approval-requests`, `GET /policies`, `GET /drift-events`, `GET /audit-events`, and posts decisions to `POST /approval-requests/{id}/decide`.
   * **Mock in Phase 4:** Seeded demo/mock fixtures via a clearly labeled test/dev script or adapter (never returned as real production monitoring data).
4. **CI/CD (`model-train-register.yml` & `model-cd-promote.yml`):**
   * Documented integration hooks (we will not break existing workflows; optionally provide a helper CLI/script or documented curl steps for Phase 5 handover).

---

## 14. Validation and Testing Strategy

Automated `pytest` suite under `backend/tests/governance/` using isolated SQLite temp DBs + mocked ClearML/Drift/Versioning adapters:
1. **Lifecycle State Machine:** Valid transitions (`candidate` → `staging` → `production` → `archived`), invalid skips (`candidate` → `production`), `archived` → `staging` blocked, controlled rollback (`was_ever_production=true` vs `false`), stage uniqueness auto-archival.
2. **Approval & Shadow Comparison:** Approval request creation, automated shadow comparison delta & regression detection against active `production` model, approve/reject flows, self-approval denial, approval invalidation on policy/metadata change, `promotion-gate` check.
3. **Metadata & Dataset Validation:** Mandatory fields per model family, immutable provenance enforcement, duplicate `(model_family, semantic_version)` & `clearml_model_id` rejection (`409`), `.mydvc` manifest + `feature_report.json` ingestion.
4. **Audit Logging Integrity:** Audit rows created for all mutations + denied/failed attempts (`outcome="denied"`), correlation IDs across multi-step promotions, verification that no update/delete API or service method mutates audit rows.
5. **RBAC & Auth Enforcement:** `401` on missing/invalid credentials, `403` across `viewer` / `ml_engineer` / `reviewer` / `admin` boundaries, strict vs dev auth mode checks.
6. **Database Constraints & Migrations:** Alembic upgrade/downgrade verification, `ON DELETE RESTRICT` on datasets in use, idempotent `monitoring_run_id` on `drift_events`.
7. **Mock Adapters:** Verify mock ClearML/Drift/Versioning responses carry explicit mock markers and never masquerade as live telemetry.
