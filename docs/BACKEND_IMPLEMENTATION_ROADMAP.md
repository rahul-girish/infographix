# Infographix — Backend Implementation Roadmap

**Execution Strategy:** Sequential, Test-Driven, Dependency-Aware Implementation  
**Branch:** `feature/backend-development`

---

## 1. Overview of Implementation Stages

```mermaid
flowchart LR
    T1["1. Scaffolding & Config"] --> T2["2. DB & Models"]
    T2 --> T3["3. Alembic Migrations"]
    T3 --> T4["4. JWT Auth & Users"]
    T4 --> T5["5. Project CRUD APIs"]
    T5 --> T6["6. Storage & Doc Upload"]
    T6 --> T7["7. PyMuPDF Extraction"]
    T7 --> T8["8. Poster JSON Persistence"]
    T8 --> T9["9. Asset Upload Service"]
    T9 --> T10["10. Submissions API"]
    T10 --> T11["11. Pytest Suite"]
    T11 --> T12["12. Frontend Integration"]
    T12 --> T13["13. Docker & Compose"]
```

---

## 2. Detailed Task Breakdown

### Task 1: Backend Foundation & Configuration
* **Description**: Create `backend/` directory structure, requirements file, environment configuration loader, and FastAPI application factory with CORS middleware.
* **Files to Create**:
  * `backend/requirements.txt`
  * `backend/.env.example`
  * `backend/app/main.py`
  * `backend/app/core/config.py`
  * `backend/app/api/v1/health.py`
* **Dependencies**: None.
* **Acceptance Criteria**: Running `uvicorn backend.app.main:app` boots cleanly; `GET /api/v1/health` returns status `200 OK`.

---

### Task 2: PostgreSQL Database Engine & Declarative Models
* **Description**: Configure SQLAlchemy engine, session maker, and all declarative database models.
* **Files to Create**:
  * `backend/app/db/base.py`
  * `backend/app/db/session.py`
  * `backend/app/models/user.py`
  * `backend/app/models/project.py`
  * `backend/app/models/section.py`
  * `backend/app/models/document.py`
  * `backend/app/models/poster.py`
  * `backend/app/models/asset.py`
  * `backend/app/models/submission.py`
* **Dependencies**: Task 1.
* **Acceptance Criteria**: Models compile without circular imports; table relationships and cascade behaviors are verified.

---

### Task 3: Alembic Migration Environment Setup
* **Description**: Initialize Alembic, configure `env.py` to bind with PostgreSQL metadata, and generate initial migration.
* **Files to Create**:
  * `backend/alembic.ini`
  * `backend/migrations/env.py`
  * `backend/migrations/versions/0001_initial_schema.py`
* **Dependencies**: Task 2.
* **Acceptance Criteria**: `alembic upgrade head` executes cleanly on PostgreSQL, generating all 7 tables and foreign keys.

---

### Task 4: JWT Authentication & User Management
* **Description**: Implement password hashing with bcrypt, JWT token generation, auth router (`/register`, `/login`, `/me`), and FastAPI auth dependency (`get_current_user`).
* **Files to Create**:
  * `backend/app/core/security.py`
  * `backend/app/core/dependencies.py`
  * `backend/app/schemas/auth.py`
  * `backend/app/services/auth_service.py`
  * `backend/app/api/v1/auth.py`
* **Dependencies**: Task 3.
* **Acceptance Criteria**: Registering a user creates an encrypted database record; login returns a valid JWT; invalid credentials return 401; `/me` returns user identity.

---

### Task 5: Project Management CRUD Endpoints
* **Description**: Implement project creation (with default template elements), listing, retrieval, renaming, duplication, and deletion with user ownership isolation.
* **Files to Create**:
  * `backend/app/schemas/project.py`
  * `backend/app/services/project_service.py`
  * `backend/app/api/v1/projects.py`
* **Dependencies**: Task 4.
* **Acceptance Criteria**: Users can only see and mutate their own projects; duplicating a project deep-copies sections and canvas elements with new UUIDs.

---

### Task 6: File Storage Service & Document Upload
* **Description**: Implement path-safe storage manager and multipart document upload route.
* **Files to Create**:
  * `backend/app/services/storage_service.py`
  * `backend/app/schemas/document.py`
  * `backend/app/api/v1/documents.py`
* **Dependencies**: Task 5.
* **Acceptance Criteria**: PDF/DOCX files <= 20 MB are stored under `backend/data/uploads/documents/`; MIME type validated; metadata recorded in database.

---

### Task 7: Deterministic Document Extraction Pipeline
* **Description**: Implement PyMuPDF and python-docx text extraction, heading detection, and section normalization with page number tracking.
* **Files to Create**:
  * `backend/app/services/document_parser.py`
  * `backend/app/schemas/section.py`
  * `backend/app/api/v1/extraction.py`
* **Dependencies**: Task 6.
* **Acceptance Criteria**: Uploaded academic paper PDF is parsed into `ResearchSection[]`; source page numbers are populated; review sections can be saved via `PUT /sections`.

---

### Task 8: Poster Canvas JSON Persistence Engine
* **Description**: Implement poster document loading and debounced autosave endpoint storing elements into PostgreSQL `JSONB`.
* **Files to Create**:
  * `backend/app/schemas/poster.py`
  * `backend/app/services/poster_service.py`
  * `backend/app/api/v1/posters.py`
* **Dependencies**: Task 5.
* **Acceptance Criteria**: Full 1000x700 Konva canvas JSON is saved and retrieved with 100% fidelity to element coordinates, layer order, and styling.

---

### Task 9: Image Asset Management
* **Description**: Multipart image upload endpoint, MIME verification, and authenticated file streaming.
* **Files to Create**:
  * `backend/app/schemas/asset.py`
  * `backend/app/services/asset_service.py`
  * `backend/app/api/v1/assets.py`
* **Dependencies**: Task 6.
* **Acceptance Criteria**: Images uploaded are saved securely and served via `/api/v1/assets/{id}/content`.

---

### Task 10: Submission Management
* **Description**: Endpoint to freeze poster snapshot and finalize project status as submitted.
* **Files to Create**:
  * `backend/app/schemas/submission.py`
  * `backend/app/api/v1/submissions.py`
* **Dependencies**: Task 8.
* **Acceptance Criteria**: `POST /projects/{id}/submit` updates status to `"submitted"` and creates an immutable snapshot.

---

### Task 11: Pytest Test Suite
* **Description**: Comprehensive integration and unit test suite verifying auth, permissions, file handling, extraction, and poster persistence.
* **Files to Create**:
  * `backend/tests/conftest.py`
  * `backend/tests/test_auth.py`
  * `backend/tests/test_projects.py`
  * `backend/tests/test_documents.py`
  * `backend/tests/test_posters.py`
* **Dependencies**: Tasks 1–10.
* **Acceptance Criteria**: All pytest test cases pass cleanly with zero mock regressions.

---

### Task 12: Frontend Integration
* **Description**: Wire frontend Axios client (`apiClient.ts`) and replace mock services (`auth.ts`, `projects.ts`, `documents.ts`, `extraction.ts`, `assets.ts`, `submissions.ts`) with real backend API calls.
* **Dependencies**: Tasks 1–11.
* **Acceptance Criteria**: Full user workflow runs from browser against the live backend server without errors.

---

### Task 13: Docker Containerization & Deployment Setup
* **Description**: Create Dockerfiles for frontend and backend, plus `docker-compose.yml` orchestrating FastAPI, PostgreSQL, and storage volumes.
* **Files to Create**:
  * `backend/Dockerfile`
  * `frontend/Dockerfile`
  * `docker-compose.yml`
  * `docs/DEPLOYMENT_GUIDE.md`
* **Dependencies**: Task 12.
* **Acceptance Criteria**: `docker compose up --build` launches the complete multi-container system with persistent PostgreSQL volume.

