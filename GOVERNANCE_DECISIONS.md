# Infographix MLOps Governance — Decisions Log

> Living record of approved, rejected, and deferred governance requirements and architectural decisions.
> Updated collaboratively during Phase 2 brainstorming. Do not silently alter previously approved decisions.

**Active Branch:** `feature/mlops-governance`  
**Last Updated:** 2026-11-28

---

## Phase 1 — Discovery Baseline (Verified)

- **Authoritative Model Registry:** ClearML Native Model Registry (`src/registry/clearml_registry.py`) supporting `poster_layout_generator` and `poster_layout_retriever`, immutable semantic versions (`X.Y.Z`), SHA-256 artifact bundles, and stages `("candidate", "staging", "production", "archived")`.
- **Legacy Registry:** File-based `src/registry/model_registry.py` (`models/registry/`) is deprecated for registration; still used locally by `scripts/train_neural_layout_local.py` for retriever loading.
- **Dataset Versioning:** `minidvc` (`.mydvc/`) with SHA-256 manifests (`v0`–`v3`), plus `feature_schema.json` and `feature_report.json` from `src/features/pipeline.py`.
- **Backend & DB:** FastAPI (`backend/`) + SQLAlchemy (`generation_runs` table) targeting Supabase PostgreSQL (SQLite fallback in local dev/tests). No Alembic migrations or authentication/RBAC currently exist.
- **Inference Service:** Standalone FastAPI (`inference_service/`) loading ClearML models by ID, semver, or unambiguous lifecycle stage.
- **CI/CD:** `.github/workflows/model-ci.yml`, `model-train-register.yml`, `model-cd-promote.yml` (Render deployment).
- **Out of Scope (Teammate Modules):** Data drift detection (Rahul RVCE AI), model versioning implementation (Rahul S RV AI), MLOps dashboard UI (Ram Charan RVCE AI). Governance will integrate via documented contracts and use mock adapters until external modules are ready.

---

## Area 2.1 — Governance Requirements and Policies

**Status:** Approved

### Approved Requirements
1. **Policy Storage & Evaluation Architecture (Option A):** Database-backed configurable governance policies with seeded defaults, evaluated by a `PolicyEvaluator` service during registration, approval, and promotion gates.
2. **Essential Policy Domains (MVP Scope):**
   - **Dataset Governance:** Require structured dataset lineage linkage (`minidvc` manifest / dataset record, split policy, quality summary) for registered models.
   - **Experiment & Reproducibility Governance:** Enforce presence of ClearML `training_task_id`, `git_commit`, hyperparameters, and runtime environment metadata at governance registration.
   - **Model Artifact Integrity:** Require SHA-256 checksum verification (`bundle_sha256s` / `checkpoint_sha256`, reusing `clearml_registry.verify_bundle_checksums`).
   - **Model Approval & Lifecycle Policies:** Enforce ordered lifecycle transitions, configurable validation metric gates (e.g., `min_val_iou`), and human approval prior to `production` promotion.
   - **Deployment Governance:** Separate governance *approval* from CI/CD *deployment*; expose a policy/compliance check gate callable by `model-cd-promote.yml` prior to Render rollout.
   - **Traceability & Auditability:** Maintain an append-only audit log for governance actions, lifecycle transitions, policy edits, and incoming drift events.
   - **Data Access & Usage Policies (Lightweight):** Enforce role-based access control on governance APIs without disrupting public/anonymous poster generation endpoints.

### Configurable Policy Behaviour
- **Self-Approval Policy:** Configurable boolean policy flag (defaulting to blocked / separation of duties enforced unless explicitly toggled in policy configuration; exact RBAC rules to be finalized in Areas 2.3 & 2.7).

### Rejected / Deferred as Unnecessary Overhead
- **Rejected (Unnecessary Overhead):** Multi-approver quorum workflows, PII/GDPR erasure workflows for SciPostLayout public research posters, and automated copyright/PII scanning pipelines.
- **Deferred / External:** Real data drift detection (Rahul), teammate-owned model versioning internals (Rahul S), MLOps dashboard UI (Ram Charan).

### Dependencies
- **Rahul S (Model Versioning):** Must supply ClearML/version registration identifiers and manifest references to governance.
- **Rahul (Data Drift):** Will post drift events to governance so drift-related policies can be evaluated and audited.
- **Ram Charan (Dashboard):** Will read policies and compliance evaluation results via governance APIs.
- **CI/CD (`model-cd-promote.yml`):** Will query the governance promotion gate before promoting/deploying to Render.

---

## Area 2.2 — Model Lifecycle Rules

**Status:** Approved

### Approved Requirements
1. **Single Lifecycle Mechanism (ClearML 4-Stage Model):** Standardize exclusively on `("candidate", "staging", "production", "archived")` from `src/registry/clearml_registry.py`. Do not re-introduce legacy `development` or create a competing lifecycle system.
2. **Governance-as-Control-Plane:** Governance DB is the authoritative enforcer of lifecycle transition rules and history, synchronizing stage tags/metadata to ClearML via an adapter (`ClearMLRegistryAdapter`).
3. **State-Transition Rules:**
   - *(New)* → `candidate`: Automated gate upon valid ClearML task, dataset lineage, and SHA-256 bundle manifest.
   - `candidate` → `staging`: Automated policy validation passes (no human approval required for staging).
   - `staging` → `production`: Automated policy validation + mandatory human approval; atomically archives the previous `production` model in the same `model_family`.
   - `production` → `staging`: Allowed as an emergency demotion with mandatory justification.
   - `candidate` / `staging` / `production` → `archived`: Allowed for retirement or supersession; mandatory reason when retiring from `production`.
   - `archived` → `production` *(Controlled Rollback)*: Allowed **only** if the model previously held `production` status (verified via governance history/audit), passes SHA-256 integrity check, and includes a mandatory rollback reason (with approval/policy check).
   - **Rejected Transitions:** `candidate` → `production` (no skipping `staging`), `archived` → `candidate`, and `archived` → `staging`.
4. **Stage Uniqueness:** Enforce at most **1 active `production` model per `model_family`** (and default to at most 1 active `staging` model per family—configurable via policy—to prevent `_resolve_model_selector` ambiguity in `inference_service`).
5. **Mandatory Justification:** Every manual/promoted transition and rollback requires a non-empty `reason` recorded in the audit trail.

### Rejected / Deferred
- **Rejected:** Canary/shadow traffic splitting and `deprecated` sub-stage (unnecessary for single-instance Render inference).

### Dependencies
- **Rahul S (Model Versioning):** Registers models into `candidate` through governance contract.
- **Rahul (Data Drift):** May flag `production` models as degraded / recommend rollback; does not bypass transition rules.
- **Ram Charan (Dashboard):** Displays lifecycle stage and valid transition actions per model.

---

## Area 2.3 — Approval and Promotion Criteria

**Status:** Approved

### Approved Requirements
1. **Option 1 — Governance API Approval Flow:** Separate four distinct stages:
   - **Automated Validation** (`PolicyEvaluator` checks)
   - **Human Approval** (`ApprovalRequest` + `ApprovalDecision` in Governance DB)
   - **Model Promotion** (Governance + ClearML lifecycle transition)
   - **Production Deployment** (CI/CD Render rollout & `/health/ready` verification)
2. **Stage Gates:**
   - `candidate` (registration): Automated provenance, metadata, SHA-256, dataset lineage, and required evaluation metrics checks.
   - `staging`: Automated policy & family-specific validation thresholds pass (no human approval required).
   - `production`: All automated checks re-verified + **Automated Shadow Comparison against current `production` model** + **Mandatory Human Approval**.
3. **Automated Shadow Comparison (Before Approval):**
   - When a model requests approval for `production` (and an active `production` model exists for that `model_family`), governance automatically computes/validates a comparison report against the active `production` model (comparing validation metrics such as `val_iou_mean`, `val_loss_total`, `val_canvas_violation_rate`, `val_unintended_overlap_mean_area` for generator, or retrieval metrics for retriever, plus optional offline/shadow evaluation outputs).
   - The comparison summary and regression verdict are attached to the `ApprovalRequest` so reviewers and the dashboard see side-by-side deltas prior to approval (with configurable policy handling for first-time `production` models or intentional trade-offs).
4. **Separation of Duties:** `allow_self_approval = False` by default (configurable in DB policy for emergency/small-team testing).
5. **Controlled Rollback:** `archived` → `production` requires the model to have previously held `production` status, valid SHA-256 integrity, and either an explicit approval OR an authorized Admin/Governance Admin emergency rollback override with mandatory justification.
6. **Approval Invalidation:** Pending/stale approval requests are invalidated if model metadata, policy thresholds, or stage eligibility changes.

### Rejected / Deferred
- **Rejected:** Multi-approver quorum and Slack/external chatops integrations for MVP.

### Dependencies
- **Rahul S (Model Versioning):** Supplies evaluation metrics and ClearML references on registration.
- **Ram Charan (Dashboard):** Displays pending approval requests, shadow comparison deltas vs. current production, and approve/reject controls.
- **CI/CD (`model-cd-promote.yml`):** Queries governance promotion gate before deploying to Render.

---

## Area 2.4 — Dataset Metadata Requirements

**Status:** Approved

### Approved Requirements
1. **Option 1 — Lightweight Governance Dataset Catalog:** Maintain a `governed_datasets` table storing dataset identity (`dataset_name`, `dataset_version` unique pair), `source_type` (`scipostlayout`, `hitl_export`, `hybrid`), `owner`, `description`, `storage_system` (`minidvc`, `s3`, `clearml_dataset`), `storage_ref`, `checksum_sha256`, `data_format` (`parquet`), `schema_version`, `record_count`, `size_bytes`, `split_config` (JSON), `quality_summary` (JSON), `preprocessing_version`, `usage_policy` / `license_info`, `parent_dataset_id` (for derived/HITL datasets), and `is_active`.
2. **Zero Duplication of Raw Files/Manifests:** Do not copy raw `.parquet` data or full per-file `minidvc` file lists or full column schemas into PostgreSQL; reference `.mydvc/manifests/*.manifest` and `feature_schema.json` / `feature_report.json`.
3. **Manifest Ingestion Support:** Provide an adapter/helper to parse `.mydvc/manifests/*.manifest` + `feature_report.json` (with mock dataset fixtures for testing).
4. **Model-Dataset Linkage Policy:** New model registrations in governance must link to a `governed_datasets` record (`dataset_id` FK), with explicit legacy-backfill support for pre-existing models such as `poster_layout_generator-v0.1.0`.

### Rejected / Deferred
- **Rejected:** Storing row-level poster ID manifests in Postgres (`train_poster_ids_sha256` already lives in ClearML retriever bundles) and automated PII/data-profiling scans (statistical drift profiling belongs to Rahul's drift module).

### Dependencies
- **Rahul (Data Drift):** References `governed_datasets` (`dataset_id`, `schema_version`, `storage_ref`) as baseline distributions.
- **Rahul S (Model Versioning):** Supplies `dataset_id` (or `dataset_name` + `dataset_version`) when registering models.
- **Ram Charan (Dashboard):** Displays dataset catalog, checksums, quality summaries, and downstream models.

---

## Area 2.5 — Model Metadata Requirements

**Status:** Approved

### Approved Requirements
1. **Option 1 — Relational Index + ClearML Reference:** Keep heavyweight artifacts, weights, full training configs, full feature schemas, and plots in ClearML; do not build a competing registry.
2. **Governance Model Record (`governed_models`):**
   - **Immutable Provenance & Identity:** `id` (UUID), `clearml_model_id` (unique), `model_name`, `model_family`, `semantic_version` (unique `(model_family, semantic_version)`), `architecture`, `parent_model_id`, `clearml_training_task_id`, `artifact_uri`, `checkpoint_sha256`, `git_commit`, `dataset_id` (FK → `governed_datasets.id`, with legacy fallback string for pre-existing models), `retriever_model_id` / `retriever_version` (dependency link for `poster_layout_generator`), `schema_version`, compact `environment_summary` (JSON), `dependencies_summary` (JSON), and `metrics_summary` (JSON).
   - **Governance State:** `owner`, `registered_at`, `lifecycle_status`, `was_ever_production` (boolean for controlled rollback eligibility), `compliance_status`, `drift_status`, and mutable `description` / `tags` (changes audited).
3. **Approval Accountability:** Approver identity and full multi-attempt approval/rejection history are stored in dedicated `approval_requests` and `approval_decisions` tables (never overwritten in a single column).
4. **Immutability Rule:** Core provenance fields (`clearml_model_id`, `semantic_version`, `checkpoint_sha256`, `clearml_training_task_id`) cannot be mutated post-registration.

### Rejected / Deferred
- **Rejected:** Background ClearML polling daemon (sync occurs via explicit API/adapter calls) and duplicating full ClearML `config_dict` / binary weights in Postgres.

### Dependencies
- **Rahul S (Model Versioning):** Registers/syncs model metadata matching the `governed_models` contract.
- **Rahul (Data Drift):** Updates/references `governed_models` (`drift_status`) when reporting drift events.
- **Ram Charan (Dashboard):** Queries `governed_models` for model inventory, lineage, metrics, and approval history.

---

## Area 2.6 — Audit Logging

**Status:** Approved

### Approved Requirements
1. **Option 1 — Append-Only `audit_events` Table:** Store structured, immutable audit records in PostgreSQL/SQLite via a dedicated `AuditService` exposing only `record()` (`INSERT`) and `query()` (`SELECT`).
2. **Required Event Coverage:**
   - Datasets: `dataset.registered`, `dataset.updated`
   - Models & Lifecycle: `model.registered`, `model.metadata_updated`, `model.lifecycle_transition`, `model.promoted_to_production`, `model.rollback`, `model.archived`, `model.transition_denied`
   - Approvals: `approval.requested`, `approval.approved`, `approval.rejected`, `approval.cancelled`, `approval.invalidated`
   - Policies & RBAC: `policy.created`, `policy.updated`, `role.assigned`, `role.revoked`
   - Drift Integration: `drift.event_recorded`
3. **Event Schema Fields:** `id` (UUID), `timestamp` (UTC), `actor_id`, `actor_role`, `action`, `entity_type`, `entity_id`, `related_model_version`, `related_dataset_version`, `previous_state` (JSON), `new_state` (JSON), `reason`, `outcome` (`success` | `denied` | `failed`), and `correlation_id`.
4. **Tamper Protection & Denied-Action Logging:**
   - Zero `PUT`/`PATCH`/`DELETE` API endpoints or ORM mutation methods on `audit_events`.
   - Denied/failed attempts are committed independently so business transaction rollbacks never erase failed/unauthorized audit records.
   - Alembic migration adds a PostgreSQL `BEFORE UPDATE OR DELETE` trigger enforcing append-only behavior in production.

### Rejected / Deferred
- **Rejected:** Cryptographic hash-chaining across rows and external SIEM log shipping for MVP.

### Dependencies
- **Rahul (Data Drift):** Emits `drift.event_recorded` events through the governance drift endpoint.
- **Rahul S (Model Versioning):** Emits `model.registered` events via governance registration contract.
- **Ram Charan (Dashboard):** Consumes read-only filtered/paginated `GET /api/v1/governance/audit-events`.

---

## Area 2.7 — User Roles and Permissions

**Status:** Approved

### Approved Requirements
1. **4-Role RBAC Model:** `admin`, `ml_engineer`, `reviewer`, and `viewer` (applied exclusively to `/api/v1/governance/*` routes; existing poster endpoints `/generate`, `/runs`, `/export/pdf` remain untouched).
2. **Permission Matrix & Separation of Duties:**
   - `viewer`: Read-only access across governed models, datasets, policies, approvals, drift events, and audit logs.
   - `ml_engineer`: Register datasets/models (`candidate`), edit mutable metadata, promote `candidate` → `staging`, submit `production` approval requests, archive non-production models, and record drift events (also used for training/drift service tokens).
   - `reviewer`: Inspect shadow comparisons & audit logs, and approve/reject `production` requests (cannot register models or edit governance policies).
   - `admin`: Manage governance policies & role assignments, promote approved models to `production`, archive `production` models, and execute emergency rollbacks.
   - **Self-Approval Guard:** Requesters cannot approve their own `ApprovalRequest` regardless of role unless `allow_self_approval` policy is explicitly enabled.
3. **Option A — Pluggable Authentication Dependency (`get_current_actor`):**
   - Supports **Bearer JWT** (Supabase-ready when user auth is added) and **Scoped Service API Tokens** (for CI/CD workflows, Rahul's drift detector, and Rahul S's versioning module).
   - Supports an explicit `GOV_AUTH_MODE=dev` (for local dev/tests only, disabled in strict/production mode) so dashboard/API testing is unblocked without building a separate login/user-registration subsystem.
   - Expose `GET /api/v1/governance/me` so Ram Charan's Dashboard can adapt UI actions to the caller's role and permissions.

### Rejected / Deferred
- **Rejected:** Building a standalone `users` / password-hashing `/login` & `/register` system inside governance (duplicates future Supabase Auth), and per-model custom ACL overrides.

### Dependencies
- **Ram Charan (Dashboard):** Authenticates dashboard calls and reads `/api/v1/governance/me` for role-aware UI controls.
- **Rahul & Rahul S / CI/CD:** Authenticate automated API calls via scoped service tokens.

---

## Area 2.8 — Governance Database Schema

**Status:** Approved

### Approved Requirements
1. **8-Table Lean Governance Schema (Coexisting with `generation_runs`):**
   - `governed_datasets`: Dataset identity, storage reference (`minidvc`/S3/ClearML Dataset), checksum, schema version, split/quality summaries, ownership, usage policy, and `parent_dataset_id` self-FK.
   - `governed_models`: ClearML index + governance state (`lifecycle_status`, `was_ever_production`, `compliance_status`, `drift_status`, `owner`), `dataset_id` FK (`ON DELETE RESTRICT`), `parent_model_id` self-FK, unique constraints on `clearml_model_id` and `(model_family, semantic_version)`.
   - `governance_policies`: DB-backed configurable policies with `policy_key`, `model_family`, `config` JSON, `enforcement_mode` (`blocking` | `warning`), and `is_enabled`.
   - `approval_requests`: Promotion/rollback requests (`pending` | `approved` | `rejected` | `cancelled` | `invalidated`), automated validation summary, `baseline_model_id`, and `shadow_comparison_report` JSON.
   - `approval_decisions`: Immutable approval/rejection decision records per request.
   - `drift_events`: Idempotent (`monitoring_run_id` unique) integration ledger for Rahul's data drift module.
   - `governance_role_assignments`: Persistent `actor_id` → `role` bindings (`admin` | `ml_engineer` | `reviewer` | `viewer`) paired with a static code-level `ROLE_PERMISSIONS` map.
   - `audit_events`: Append-only ledger with zero foreign keys (survives entity cleanup) and a PostgreSQL `BEFORE UPDATE OR DELETE` trigger.
2. **Dual-Dialect Compatibility:** Use SQLAlchemy 2.0 `Base`, `Uuid(as_uuid=True)`, `DateTime(timezone=True)`, and `JSONVariant` (`JSONB` on PostgreSQL, `JSON` on SQLite) matching `backend/app/models/generation_run.py`.
3. **Alembic Migrations:** Introduce Alembic under `backend/` (`backend/alembic.ini`, `backend/alembic/`) with dialect-aware migrations that safely coexist with existing `generation_runs` tables.

### Rejected / Deferred
- **Rejected:** Separate `roles`, `permissions`, and `role_permission_mappings` join tables (unnecessary for 4 fixed roles), duplicate `users` table, or duplicating `generation_runs`.

### Dependencies
- **All Teammates & CI/CD:** Schema supports stable foreign-keyed integration for Model Versioning (`governed_models`), Data Drift (`drift_events`), and Dashboard (`audit_events`, `approval_requests`, `governance_policies`).

---

## Phase 2 Sign-Off

All 8 governance areas (`2.1`–`2.8`) are approved and locked in.  
**Next Step:** Phase 3 Consolidated Governance Architecture & Specification (`GOVERNANCE_SPECIFICATION.md`). Implementation begins only after explicit user approval of the specification.
