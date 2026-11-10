# Infographix — Frontend-Backend Integration Gap Analysis

**Analysis Target:** Bridging existing React/Vite mock frontend with new FastAPI backend  
**Integration Strategy:** Surgical replacement of frontend mock service modules; zero rewriting of UI components.

---

## 1. Feature Gap Matrix

| Feature | Frontend Status | Backend Requirement | Integration Work | Priority |
|---|---|---|---|---|
| **User Registration & Login** | Mock session in `sessionStorage`. Passwords never verified or sent to network. | `POST /api/v1/auth/register`, `POST /api/v1/auth/login` returning JWT access tokens. | Replace `auth.ts` with real API calls; store JWT in session; add Axios auth interceptor in `apiClient.ts`. | **High** |
| **Project CRUD & List** | `localStorage` array of `Project` objects under `infographix.projects.v1`. | `/api/v1/projects` endpoints with user isolation and PostgreSQL persistence. | Replace `projects.ts` storage methods with Axios HTTP calls. Match JSON response keys. | **High** |
| **Document Upload** | Extension/size checks + simulated progress timers in `documents.ts`. File bytes discarded. | Multipart file upload to `/api/v1/projects/{id}/documents` with validation and local disk storage. | Update `documents.ts` to send `FormData` via `apiClient.post` and report Axios `onUploadProgress`. | **High** |
| **Document Extraction** | Mock `extractionService.extract()` returns static 13 sections with filename title. | Real PyMuPDF/python-docx deterministic parser at `/api/v1/projects/{id}/extract`. | Replace `extraction.ts` with call to backend extraction endpoint. Map response into `ResearchSection[]`. | **High** |
| **Poster Canvas Autosave** | Debounced 1.2s call to `projectsService.savePoster()`, writing to `localStorage`. | `PUT /api/v1/projects/{id}/poster` receiving structured `PosterDocument` and writing to `JSONB`. | Wire `projectsService.savePoster` to backend `PUT` endpoint. Ensure element UUIDs and coordinates are preserved. | **Critical** |
| **Image Asset Upload** | Base64 data URL encoded in browser; 1 MB limit in `assets.ts`. | `POST /api/v1/assets` multipart upload returning asset ID and hosted URL. | Update `assets.ts` to upload file to backend and return remote `/content` URL instead of data URL. | **Medium** |
| **Project Submission** | Sets `project.status = "submitted"` in `localStorage`. | `POST /api/v1/projects/{id}/submit` recording submission timestamp and freezing snapshot. | Wire `submissions.ts` to call submission API endpoint. | **Medium** |
| **Error Handling & Feedback** | Standard try/catch setting React `error` state string. | Standard FastAPI `{ "detail": "message" }` error envelope. | Add Axios response interceptor to extract `error.response?.data?.detail`. | **Medium** |

---

## 2. Detailed Seam Replacement Plan

### 2.1 HTTP Client Authentication Interceptor (`frontend/src/services/apiClient.ts`)
Add a request interceptor to attach the JWT access token from session storage automatically:
```typescript
apiClient.interceptors.request.use((config) => {
  const token = window.sessionStorage.getItem("infographix.token.v1");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
```

### 2.2 Project Management Seam (`frontend/src/services/projects.ts`)
Instead of reading `localStorage`, methods become thin wrappers over `apiClient`:
* `list()` -> `GET /api/v1/projects`
* `get(id)` -> `GET /api/v1/projects/${id}`
* `create(title)` -> `POST /api/v1/projects`
* `save(project)` -> `PUT /api/v1/projects/${project.id}`
* `rename(id, title)` -> `PATCH /api/v1/projects/${id}/rename`
* `duplicate(id)` -> `POST /api/v1/projects/${id}/duplicate`
* `delete(id)` -> `DELETE /api/v1/projects/${id}`
* `savePoster(id, poster)` -> `PUT /api/v1/projects/${id}/poster`
* `saveSections(id, sections)` -> `PUT /api/v1/projects/${id}/sections`

### 2.3 Document Upload Seam (`frontend/src/services/documents.ts`)
```typescript
async upload(projectId: string, file: File, onProgress?: (progress: number) => void) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await apiClient.post(`/projects/${projectId}/documents`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (event) => {
      if (event.total) onProgress?.(Math.round((event.loaded * 100) / event.total));
    },
  });
  return response.data;
}
```

---

## 3. Potential Integration Risks & Mitigations

1. **Autosave High Frequency**:
   * *Risk*: When a user drags or transforms elements continuously in the canvas, `EditorPage.tsx` fires autosave requests every 1,200 ms.
   * *Mitigation*: Backend `PUT /api/v1/projects/{id}/poster` must execute lightweight `UPDATE poster_documents SET elements = :elements, updated_at = :now WHERE project_id = :id` without complex relational joins.
2. **CORS Headers in Development**:
   * *Risk*: Browser blocking requests from `http://localhost:5173` to `http://localhost:8000`.
   * *Mitigation*: FastAPI backend initializes `CORSMiddleware` with explicit support for `localhost:5173` and wildcard development headers.
3. **Data Loss During Canvas Transformation**:
   * *Risk*: Loss of non-standard properties on Konva elements if the backend model is overly rigid.
   * *Mitigation*: Pydantic schema for `PosterElement` uses `extra="allow"` or accepts generic dictionary styling attributes, ensuring no element metadata is dropped.

