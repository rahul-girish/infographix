# Frontend/backend integration report

## Current integration status

The frontend is independently runnable and uses local mock services. No FastAPI routes or database integration are implemented or assumed to exist. `frontend/src/services/apiClient.ts` reads `VITE_API_BASE_URL` (default `/api`) and is the shared Axios client for a future integration.

## Mock-service boundaries

| Service | Current behavior | Backend replacement seam |
|---|---|---|
| `auth.ts` | Mock login/register, user identity in session storage; no password persistence | Replace service methods with auth API calls and map API errors to the existing form messages |
| `projects.ts` | Project CRUD, status and poster/section saves in local storage | Replace list/get/create/save/rename/duplicate/delete/status methods with project API calls |
| `documents.ts` | PDF/DOCX extension and 20 MB checks; simulated progress only | Send selected file as multipart form data and report transport progress |
| `extraction.ts` | Returns editable mock research sections | Request extraction for a stored document and map the response to `ResearchSection[]` |
| `posters.ts` | Poster JSON serialization, validation and SVG export | Keep the structured poster payload contract or adapt it at the API boundary |
| `assets.ts` | Reads images as data URLs, with a 1 MB local-storage limit | Upload assets and return stable asset IDs/URLs instead of embedding data URLs |
| `submissions.ts` | Marks the project submitted in local storage | Create a submission through the API and refresh the returned status |

Suggested API shapes below are integration proposals only; they are not implemented endpoints:

| Proposed operation | Frontend data |
|---|---|
| `POST /auth/login`, `POST /auth/register`, `POST /auth/password-reset` | Credentials submitted directly to the backend; token/session handled according to backend security requirements |
| `GET /projects`, `POST /projects`, `GET /projects/{id}`, `PATCH /projects/{id}`, `DELETE /projects/{id}` | `Project` and project status |
| `POST /projects/{id}/documents` | Multipart PDF/DOCX upload |
| `POST /documents/{id}/extract` | `ResearchSection[]` extraction result |
| `GET/PUT /projects/{id}/poster` | Structured `PosterDocument` JSON |
| `POST /assets` | Image bytes and returned asset reference |
| `POST /projects/{id}/submissions` | Submission request and resulting status |

## Integration considerations

- Keep UI components dependent on service methods and TypeScript models, not Axios calls.
- Agree on error envelopes, pagination, auth/session policy, file limits, asset URL lifetime and poster schema versioning with the backend team.
- Browser local storage has small quotas and is not suitable for production research documents or large image data. The current 20 MB document check is a UI mock constraint; uploads are not persisted.
- Image data URLs are limited to 1 MB in the mock editor because they are embedded in local project JSON.
- Configure the API URL through `VITE_API_BASE_URL`; do not commit credentials or tokens in environment files.
