# Infographix — Backend REST API Specification

**Specification Version:** 1.0.0  
**Base URL:** `/api/v1`  
**Protocol:** HTTPS / HTTP (Development)  
**Standard Error Envelope:**
```json
{
  "detail": "Error description message"
}
```

---

## 1. Authentication & User Management

### 1.1 User Registration
* **Endpoint:** `POST /api/v1/auth/register`
* **Triggering UI:** `AuthPage.tsx (mode="register")` -> `authService.register(name, email, password)`
* **Auth Requirement:** None (Public)
* **Request Body (`application/json`):**
```json
{
  "name": "Alex Morgan",
  "email": "alex@university.edu",
  "password": "StrongPassword123!"
}
```
* **Success Response (`201 Created`):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Alex Morgan",
    "email": "alex@university.edu",
    "created_at": "2026-11-08T10:00:00Z"
  }
}
```
* **Error Responses:**
  * `400 Bad Request`: Email already registered or invalid input format.
  * `422 Unprocessable Entity`: Validation failure (e.g. password < 8 characters).

---

### 1.2 User Login
* **Endpoint:** `POST /api/v1/auth/login`
* **Triggering UI:** `AuthPage.tsx (mode="login")` -> `authService.login(email, password)`
* **Auth Requirement:** None (Public)
* **Request Body (`application/json`):**
```json
{
  "email": "alex@university.edu",
  "password": "StrongPassword123!"
}
```
* **Success Response (`200 OK`):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Alex Morgan",
    "email": "alex@university.edu"
  }
}
```
* **Error Responses:**
  * `401 Unauthorized`: Invalid email or password credentials.

---

### 1.3 Current User Profile
* **Endpoint:** `GET /api/v1/auth/me`
* **Triggering UI:** `WorkspaceHeader.tsx` / `authService.getCurrentUser()`
* **Auth Requirement:** Bearer Token (`Authorization: Bearer <token>`)
* **Success Response (`200 OK`):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Alex Morgan",
  "email": "alex@university.edu"
}
```

---

### 1.4 Password Reset Request
* **Endpoint:** `POST /api/v1/auth/forgot-password`
* **Triggering UI:** `AuthPage.tsx (mode="forgot")` -> `authService.requestPasswordReset(email)`
* **Request Body (`application/json`):**
```json
{
  "email": "alex@university.edu"
}
```
* **Success Response (`200 OK`):**
```json
{
  "message": "If an account matches this email, password reset instructions have been sent."
}
```

---

## 2. Project Management

### 2.1 List Projects
* **Endpoint:** `GET /api/v1/projects`
* **Triggering UI:** `DashboardPage.tsx` -> `projectsService.list()`
* **Auth Requirement:** Bearer Token
* **Query Parameters (Optional):**
  * `status`: `"draft"` | `"submitted"`
  * `search`: string
* **Success Response (`200 OK`):**
```json
[
  {
    "id": "c73a218c-3094-4d8b-967a-1e18cdca8810",
    "title": "A new perspective on research",
    "status": "draft",
    "updatedAt": "2026-11-08T10:30:00Z",
    "templateId": "research",
    "documentName": "neural_poster_draft.pdf",
    "sections": [
      {
        "id": "title",
        "title": "Title",
        "content": "A new perspective on research",
        "included": true
      }
    ]
  }
]
```

---

### 2.2 Create Project
* **Endpoint:** `POST /api/v1/projects`
* **Triggering UI:** `DashboardPage.tsx` -> `projectsService.create()`
* **Auth Requirement:** Bearer Token
* **Request Body (`application/json`):**
```json
{
  "title": "Untitled research poster",
  "templateId": "research"
}
```
* **Success Response (`201 Created`):**
```json
{
  "id": "c73a218c-3094-4d8b-967a-1e18cdca8810",
  "title": "Untitled research poster",
  "status": "draft",
  "updatedAt": "2026-11-08T10:30:00Z",
  "templateId": "research",
  "sections": [ ... ],
  "poster": {
    "id": "e44c8038-0810-410a-b51e-61bf035dbf41",
    "projectId": "c73a218c-3094-4d8b-967a-1e18cdca8810",
    "title": "Untitled research poster",
    "width": 1000,
    "height": 700,
    "orientation": "landscape",
    "background": "#ffffff",
    "templateId": "research",
    "columns": 3,
    "elements": [ ... ],
    "updatedAt": "2026-11-08T10:30:00Z"
  }
}
```

---

### 2.3 Get Project Details
* **Endpoint:** `GET /api/v1/projects/{projectId}`
* **Triggering UI:** `UploadPage.tsx`, `ReviewPage.tsx`, `ConfigurePage.tsx`, `EditorPage.tsx`, `PreviewPage.tsx` -> `projectsService.get(projectId)`
* **Auth Requirement:** Bearer Token (Verifies project ownership)
* **Success Response (`200 OK`):**
Returns the full `Project` object (including `sections`, `poster`, `documentName`, `status`).
* **Error Response:** `404 Not Found` if project does not exist or user is unauthorized.

---

### 2.4 Update Project
* **Endpoint:** `PUT /api/v1/projects/{projectId}`
* **Triggering UI:** `projectsService.save(project)`
* **Auth Requirement:** Bearer Token
* **Request Body (`application/json`):**
```json
{
  "title": "Updated Title",
  "templateId": "conference",
  "status": "draft"
}
```
* **Success Response (`200 OK`):** Returns updated `Project`.

---

### 2.5 Rename Project
* **Endpoint:** `PATCH /api/v1/projects/{projectId}/rename`
* **Triggering UI:** `DashboardPage.tsx` -> `projectsService.rename(id, title)`
* **Request Body (`application/json`):**
```json
{
  "title": "New Poster Title"
}
```
* **Success Response (`200 OK`):**
```json
{
  "id": "c73a218c-3094-4d8b-967a-1e18cdca8810",
  "title": "New Poster Title",
  "updatedAt": "2026-11-08T10:35:00Z"
}
```

---

### 2.6 Duplicate Project
* **Endpoint:** `POST /api/v1/projects/{projectId}/duplicate`
* **Triggering UI:** `DashboardPage.tsx` -> `projectsService.duplicate(id)`
* **Auth Requirement:** Bearer Token
* **Success Response (`201 Created`):** Returns a cloned `Project` with new UUIDs for the project, poster, and canvas elements.

---

### 2.7 Delete Project
* **Endpoint:** `DELETE /api/v1/projects/{projectId}`
* **Triggering UI:** `DashboardPage.tsx` -> `projectsService.delete(id)`
* **Auth Requirement:** Bearer Token
* **Success Response (`204 No Content`)**

---

### 2.8 Update Project Status
* **Endpoint:** `PATCH /api/v1/projects/{projectId}/status`
* **Triggering UI:** `submissionsService.submit(id)` / `projectsService.updateStatus(id, status)`
* **Request Body (`application/json`):**
```json
{
  "status": "submitted"
}
```
* **Success Response (`200 OK`)**

---

## 3. Documents & Information Extraction

### 3.1 Upload Research Document
* **Endpoint:** `POST /api/v1/projects/{projectId}/documents`
* **Triggering UI:** `UploadPage.tsx` -> `documentsService.upload(file)` & `projectsService.attachDocument(id, name)`
* **Auth Requirement:** Bearer Token
* **Request Type:** `multipart/form-data`
  * Form field `file`: Binary file data (.pdf or .docx)
* **Validation:** Extension must be `.pdf` or `.docx`, MIME type validated, file size <= 20 MB.
* **Success Response (`201 Created`):**
```json
{
  "id": "713f0a51-4091-4e76-8022-d7b32c694a31",
  "projectId": "c73a218c-3094-4d8b-967a-1e18cdca8810",
  "fileName": "deep_learning_scientific_layouts.pdf",
  "fileSize": 3456782,
  "contentType": "application/pdf",
  "pageCount": 8,
  "uploadedAt": "2026-11-08T10:40:00Z"
}
```

---

### 3.2 Trigger Document Extraction
* **Endpoint:** `POST /api/v1/projects/{projectId}/extract`
* **Triggering UI:** `UploadPage.tsx` -> `extractionService.extract(fileName)`
* **Auth Requirement:** Bearer Token
* **Query / Body:** Optional extraction parameters (e.g. force re-extract).
* **Success Response (`200 OK`):**
```json
{
  "projectId": "c73a218c-3094-4d8b-967a-1e18cdca8810",
  "documentName": "deep_learning_scientific_layouts.pdf",
  "sections": [
    {
      "id": "title",
      "title": "Title",
      "content": "Deep Learning for Scientific Layouts",
      "included": true,
      "pageNumber": 1
    },
    {
      "id": "authors",
      "title": "Authors",
      "content": "A. Researcher, B. Scientist",
      "included": true,
      "pageNumber": 1
    },
    {
      "id": "abstract",
      "title": "Abstract",
      "content": "This paper explores automated layout discovery using spatial clustering...",
      "included": true,
      "pageNumber": 1
    },
    {
      "id": "methodology",
      "title": "Methodology",
      "content": "We apply 2D anisotropic DBSCAN to group bounding boxes into coherent reading columns...",
      "included": true,
      "pageNumber": 3
    }
  ]
}
```

---

### 3.3 Save / Update Research Sections
* **Endpoint:** `PUT /api/v1/projects/{projectId}/sections`
* **Triggering UI:** `ReviewPage.tsx` -> `projectsService.saveSections(projectId, sections)`
* **Auth Requirement:** Bearer Token
* **Request Body (`application/json`):**
```json
{
  "sections": [
    {
      "id": "title",
      "title": "Title",
      "content": "Customized Title by Researcher",
      "included": true
    },
    {
      "id": "abstract",
      "title": "Abstract",
      "content": "Edited summary...",
      "included": false
    }
  ]
}
```
* **Success Response (`200 OK`):** Returns the persisted array of `ResearchSection[]`.

---

## 4. Poster Canvas Persistence

### 4.1 Get Poster Canvas
* **Endpoint:** `GET /api/v1/projects/{projectId}/poster`
* **Triggering UI:** `EditorPage.tsx`, `PreviewPage.tsx`
* **Auth Requirement:** Bearer Token
* **Success Response (`200 OK`):** Returns full `PosterDocument` JSON.

---

### 4.2 Save / Autosave Poster Canvas
* **Endpoint:** `PUT /api/v1/projects/{projectId}/poster`
* **Triggering UI:** `EditorPage.tsx` -> `projectsService.savePoster(projectId, document)` (Autosaved every 1.2s on edit)
* **Auth Requirement:** Bearer Token
* **Request Body (`application/json`):** Full `PosterDocument` JSON payload matching TypeScript interface.
* **Backend Processing:** Validates JSON schema integrity, persists to PostgreSQL `poster_documents` table (with elements stored in `JSONB`), updates `updatedAt` timestamp.
* **Success Response (`200 OK`):**
```json
{
  "id": "e44c8038-0810-410a-b51e-61bf035dbf41",
  "projectId": "c73a218c-3094-4d8b-967a-1e18cdca8810",
  "updatedAt": "2026-11-08T10:45:00Z"
}
```

---

## 5. Image & Media Asset Management

### 5.1 Upload Asset
* **Endpoint:** `POST /api/v1/assets`
* **Triggering UI:** `assetsService.readImage(file)` / Editor Left Panel image upload
* **Auth Requirement:** Bearer Token
* **Request Type:** `multipart/form-data`
  * Form field `file`: Image file (PNG, JPG, WEBP, SVG)
  * Form field `projectId`: Optional project association
* **Validation:** MIME type must start with `image/`, max file size 10 MB.
* **Success Response (`201 Created`):**
```json
{
  "id": "a12d34e5-67f8-4901-b234-c567890def12",
  "url": "/api/v1/assets/a12d34e5-67f8-4901-b234-c567890def12/content",
  "fileName": "neural_net_diagram.png",
  "contentType": "image/png",
  "size": 524288
}
```

---

### 5.2 Stream Asset Image Content
* **Endpoint:** `GET /api/v1/assets/{assetId}/content`
* **Triggering UI:** `<img src="..." />` / React-Konva `<Image>` elements
* **Auth Requirement:** Token or signed access parameter
* **Success Response (`200 OK`):** Raw image binary stream with appropriate `Content-Type` header (`image/png`, `image/jpeg`).

---

## 6. Submissions

### 6.1 Submit Project
* **Endpoint:** `POST /api/v1/projects/{projectId}/submit`
* **Triggering UI:** `PreviewPage.tsx` -> `submissionsService.submit(projectId)`
* **Auth Requirement:** Bearer Token
* **Success Response (`200 OK`):**
```json
{
  "id": "sub_883a912d-1140-4288-a901-72f10b392110",
  "projectId": "c73a218c-3094-4d8b-967a-1e18cdca8810",
  "status": "submitted",
  "submittedAt": "2026-11-08T10:50:00Z"
}
```

---

## 7. System Health Check
* **Endpoint:** `GET /api/v1/health`
* **Auth Requirement:** None
* **Success Response (`200 OK`):**
```json
{
  "status": "healthy",
  "database": "connected",
  "version": "1.0.0"
}
```

