# Infographix — Backend Test Execution Report

**Execution Date:** 2026-11-08  
**Test Framework:** `pytest 9.2.0` (Python 3.12.3)  
**Database:** PostgreSQL 16 (`infographix-postgres` container)  
**ORM:** SQLAlchemy 2.1.8  
**API Engine:** FastAPI 0.162.2 / Starlette TestClient  
**Git Branch:** `feature/backend-development`

---

## 1. Test Summary

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0
rootdir: /home/raksh/projects/infographix
collected 9 items

backend/tests/test_assets_and_submissions.py::test_asset_lifecycle PASSED        [ 11%]
backend/tests/test_assets_and_submissions.py::test_project_submission PASSED     [ 22%]
backend/tests/test_auth.py::test_register_user PASSED                           [ 33%]
backend/tests/test_auth.py::test_register_duplicate_user PASSED                 [ 44%]
backend/tests/test_auth.py::test_login_success PASSED                           [ 55%]
backend/tests/test_auth.py::test_login_invalid_password PASSED                  [ 66%]
backend/tests/test_auth.py::test_get_me PASSED                                  [ 77%]
backend/tests/test_documents.py::test_document_upload_and_extraction PASSED      [ 88%]
backend/tests/test_projects.py::test_project_lifecycle PASSED                   [100%]

========================= 9 passed, 1 warning in 6.69s =========================
```

---

## 2. Test Suite Breakdown

### 2.1 Authentication & Profile (`test_auth.py`)
| Test Case | Description | Result |
|---|---|---|
| `test_register_user` | Creates user account with bcrypt password hashing; returns JWT token. | **PASSED** |
| `test_register_duplicate_user` | Rejects duplicate email registration with 400 Bad Request error. | **PASSED** |
| `test_login_success` | Validates credentials against bcrypt hash and issues Bearer access token. | **PASSED** |
| `test_login_invalid_password` | Rejects incorrect password with 401 Unauthorized. | **PASSED** |
| `test_get_me` | Authenticated `GET /api/v1/auth/me` retrieves verified user identity. | **PASSED** |

### 2.2 Project Management & Poster Persistence (`test_projects.py`)
| Test Case | Description | Result |
|---|---|---|
| `test_project_lifecycle` | End-to-end project workflow: creates project with 13 starter research sections and 11 starter canvas elements; renames project; deep-clones project regenerating all element UUIDs; persists canvas JSON updates in PostgreSQL JSONB; deletes project with cascading cleanup. | **PASSED** |

### 2.3 Document Ingestion & Extraction (`test_documents.py`)
| Test Case | Description | Result |
|---|---|---|
| `test_document_upload_and_extraction` | Uploads generated 2-page academic PDF; executes deterministic PyMuPDF parser; detects title, authors, abstract (page 1) and methodology (page 2); preserves page numbers; verifies authenticated file download endpoint. | **PASSED** |

### 2.4 Assets & Submissions (`test_assets_and_submissions.py`)
| Test Case | Description | Result |
|---|---|---|
| `test_asset_lifecycle` | Uploads PNG image asset; verifies MIME type and size; streams image via `/api/v1/assets/{id}/content`; deletes asset and validates 404 response. | **PASSED** |
| `test_project_submission` | Submits draft poster; updates status to `"submitted"`; freezes immutable canvas snapshot into PostgreSQL JSONB; verifies submission retrieval. | **PASSED** |

---

## 3. Conclusion

All 9 integration and unit tests execute cleanly against real PostgreSQL. Zero mock fallbacks are used in the test suite. All endpoints conform strictly to the frontend TypeScript contracts.

