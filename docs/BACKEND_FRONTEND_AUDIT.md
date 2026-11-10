# Infographix — Frontend Audit Report

**Audit Date:** 2026-11-08  
**Audit Target:** `frontend/` directory and codebase integration seams  
**Auditor:** Backend Development & Database Team  
**Git Branch:** `feature/backend-development` (Commit: `76b58f2`)

---

## 1. Executive Summary

This audit inspects the complete frontend application built for **Infographix**. The frontend provides a responsive, component-based user interface using React 19, TypeScript 6, Vite 8, Tailwind CSS v4, and React-Konva 19 for canvas manipulation.

The user interface implements all 9 stages of the product journey:
1. User Access & Authentication (`/`, `/login`, `/register`, `/forgot-password`)
2. Dashboard & Project Lifecycle (`/dashboard`)
3. Document Ingestion (`/projects/:projectId/upload`)
4. Information Extraction Review (`/projects/:projectId/review`)
5. Poster Layout Configuration (`/projects/:projectId/configure`)
6. Three-Panel Interactive Poster Editor (`/projects/:projectId/editor`)
7. Final Review, SVG Export & Submission (`/projects/:projectId/preview`)

**Key Architectural Finding:**  
The frontend is completely decoupled from any live backend. All data persistence currently resides in browser `localStorage` and `sessionStorage`. File uploads, document text extraction, and submissions are simulated using client-side JavaScript promises with artificial delays. However, a shared Axios HTTP client exists at [`frontend/src/services/apiClient.ts`](file:///wsl.localhost/Ubuntu-24.04/home/raksh/projects/infographix/frontend/src/services/apiClient.ts) pointing to `VITE_API_BASE_URL` (defaulting to `/api`), providing a clean seam for backend integration.

---

## 2. Framework, Dependencies & Environment

Inspected from [`frontend/package.json`](file:///wsl.localhost/Ubuntu-24.04/home/raksh/projects/infographix/frontend/package.json):

| Dependency | Version | Architectural Role |
|---|---|---|
| `react` / `react-dom` | `^19.3.1` | Core UI library |
| `react-router-dom` | `^7.18.5` | Client-side routing and URL parameters (`:projectId`) |
| `konva` / `react-konva` | `^10.9.1` / `^19.3.2` | 2D canvas rendering, drag-and-drop, bounding-box selection, transformers |
| `zustand` | `^5.1.0` | Global state management for canvas elements, selection, and history |
| `axios` | `^1.21.0` | HTTP client configured for future backend integration |
| `lucide-react` | `^1.70.0` | Iconography |
| `tailwindcss` / `@tailwindcss/vite` | `^4.3.5` | Styling and visual layout |
| `typescript` | `~6.0.3` | Type safety and contract enforcement |
| `vite` | `^8.3.6` | Development server and production bundler |

---

## 3. Application Routing and View Structure

Inspected from [`frontend/src/App.tsx`](file:///wsl.localhost/Ubuntu-24.04/home/raksh/projects/infographix/frontend/src/App.tsx):

```text
/                          -> LandingPage.tsx
/login                     -> AuthPage.tsx (mode="login")
/register                  -> AuthPage.tsx (mode="register")
/forgot-password           -> AuthPage.tsx (mode="forgot")
/dashboard                 -> DashboardPage.tsx
/projects/:projectId/upload      -> UploadPage.tsx
/projects/:projectId/review      -> ReviewPage.tsx
/projects/:projectId/configure   -> ConfigurePage.tsx
/projects/:projectId/editor      -> EditorPage.tsx
/projects/:projectId/preview     -> PreviewPage.tsx
*                          -> 404 Not Found fallback
```

---

## 4. Frontend Data Contracts & Types

Inspected from [`frontend/src/types/index.ts`](file:///wsl.localhost/Ubuntu-24.04/home/raksh/projects/infographix/frontend/src/types/index.ts):

### 4.1 Project & Workflow Models

```typescript
export type ProjectStatus = "draft" | "submitted";
export type Orientation = "portrait" | "landscape";
export type PosterElementType = "text" | "image" | "rect" | "circle" | "line" | "arrow";
export type TextAlignment = "left" | "center" | "right";

export interface ResearchSection {
  id: string;          // e.g. "title", "authors", "abstract", "methodology", or UUID
  title: string;       // Human-readable section heading
  content: string;     // Text body of the section
  included: boolean;   // Whether the section is active for the poster layout
}

export interface Project {
  id: string;
  title: string;
  status: ProjectStatus;
  updatedAt: string;
  templateId: string;
  sections: ResearchSection[];
  documentName?: string;
  poster?: PosterDocument;
}

export interface ProjectUser {
  name: string;
  email: string;
}
```

### 4.2 Canvas & Poster Document Models

```typescript
export interface PosterElement {
  id: string;
  type: PosterElementType;
  name: string;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  layer: number;
  visible: boolean;
  locked: boolean;
  groupId?: string;
  text?: string;
  fontFamily?: string;
  fontSize?: number;
  fontWeight?: "normal" | "bold";
  fontStyle?: "normal" | "italic";
  underline?: boolean;
  align?: TextAlignment;
  lineHeight?: number;
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
  opacity?: number;
  imageFit?: "contain" | "cover";
  assetId?: string;
  src?: string;
  points?: number[];
}

export interface PosterDocument {
  id: string;
  projectId: string;
  title: string;
  width: number;
  height: number;
  orientation: Orientation;
  background: string;
  templateId: string;
  columns: number;
  elements: PosterElement[];
  updatedAt: string;
}
```

---

## 5. Detailed Inspection of Mock Services

Located in `frontend/src/services/`:

### 5.1 `auth.ts`
* **Current Behavior**: Stores `{ name, email }` in `window.sessionStorage` under key `infographix.user.v1`. Password is not sent anywhere or verified; requires only length >= 8.
* **Seam for Backend**: Needs real API calls to `POST /api/v1/auth/login`, `POST /api/v1/auth/register`, and session token handling via JWT Bearer.

### 5.2 `projects.ts`
* **Current Behavior**: Reads and writes an array of `Project` objects in `window.localStorage` under key `infographix.projects.v1`.
* **Methods**:
  * `list()`: Returns all projects sorted by `updatedAt` descending.
  * `get(id)`: Returns project by ID.
  * `create(title)`: Initializes default project with `posterTemplates[0]`, starter sections, and starter canvas elements.
  * `save(project)`: Updates existing project in array.
  * `rename(id, title)`: Renames project and updates nested `poster.title`.
  * `duplicate(id)`: Clones project, generates new UUIDs for project, poster, and all canvas elements.
  * `delete(id)`: Removes project from array.
  * `updateStatus(id, status)`: Sets status to `"draft"` or `"submitted"`.
  * `savePoster(id, poster)`: Updates poster document and project title.
  * `saveSections(id, sections)`: Updates `project.sections`.
  * `attachDocument(id, name)`: Saves `project.documentName`.
* **Seam for Backend**: Direct 1-to-1 mapping to `/api/v1/projects` REST endpoints.

### 5.3 `documents.ts`
* **Current Behavior**: Validates file extension (`.pdf`, `.docx`) and file size (<= 20 MB). Simulates upload progress (18%, 42%, 68%, 91%, 100%) with 110ms timeouts. Returns `{ name: file.name }`. Does not store file content.
* **Seam for Backend**: Must be replaced with `multipart/form-data` upload to `POST /api/v1/documents/upload` or `POST /api/v1/projects/{projectId}/documents`.

### 5.4 `extraction.ts`
* **Current Behavior**: Returns a static copy of 13 predefined research sections (`defaultSections`), optionally replacing the title with the uploaded filename.
* **Seam for Backend**: Must call real backend deterministic parsing service (`POST /api/v1/documents/{documentId}/extract` or `/api/v1/projects/{projectId}/extract`).

### 5.5 `assets.ts`
* **Current Behavior**: Reads images via browser `FileReader.readAsDataURL()` and enforces a 1 MB file size limit to prevent overflowing browser `localStorage`.
* **Seam for Backend**: Must upload image to `POST /api/v1/assets`, receive an asset ID and URL, and assign `src` to the hosted endpoint.

### 5.6 `posters.ts`
* **Current Behavior**: Provides JSON `serialize()` and `deserialize()`, as well as client-side SVG generation (`toSvg()`) parsing Konva elements into SVG tags (`<rect>`, `<text>`, `<polyline>`, `<ellipse>`, `<image>`).
* **Seam for Backend**: The SVG export runs completely in the browser; backend only needs to store and return the exact `PosterDocument` JSON.

### 5.7 `submissions.ts`
* **Current Behavior**: Calls `projectsService.updateStatus(projectId, "submitted")`.
* **Seam for Backend**: Calls `POST /api/v1/projects/{projectId}/submit`, validating poster readiness and recording submission metadata.

---

## 6. Audit of the Canvas Editor & Store

Inspected from [`frontend/src/stores/editorStore.ts`](file:///wsl.localhost/Ubuntu-24.04/home/raksh/projects/infographix/frontend/src/stores/editorStore.ts) and [`frontend/src/components/editor/`](file:///wsl.localhost/Ubuntu-24.04/home/raksh/projects/infographix/frontend/src/components/editor/):

1. **State Store (`editorStore.ts`)**:
   * Uses Zustand with undo/redo snapshot history (`undoStack`, `redoStack`, max depth 30).
   * Holds `document: PosterDocument | null`.
   * Element mutations (`updateElement`, `addElement`, `deleteElement`, `bringForward`, `sendBackward`, `bringToFront`, `sendToBack`, `groupSelected`, `ungroupSelected`) operate immutably on `document.elements`.
2. **Autosave Engine (`EditorPage.tsx`)**:
   * Debounced autosave compares serialized JSON of `document` against `lastSavedDocument.current`.
   * Triggers `projectsService.savePoster(projectId, document)` every 1,200 ms if modified.
   * Backend must handle periodic `PUT /api/v1/projects/{id}/poster` requests efficiently.
3. **Konva Stage (`PosterCanvas.tsx`)**:
   * Renders canvas at exact poster dimensions (e.g., 1000x700 landscape or 700x1000 portrait).
   * Supports selection transformer, shift-select multi-selection, snapping to 10px grid, and alignment guides.

---

## 7. Known Frontend Limitations & Critical Notes

1. **Local Storage Limit**: Because `localStorage` is restricted to ~5 MB per domain in most browsers, saving multiple projects with embedded image data URLs causes `QuotaExceededError`. Backend persistence is urgently required.
2. **Data Isolation**: Currently, all mock projects are shared across whatever browser user is loaded; there is no tenant separation.
3. **No File Binary Retention**: Once uploaded, the raw PDF or DOCX is lost—only its filename is saved.
4. **Editable Element Preservation**: The canvas editor relies heavily on element IDs (`crypto.randomUUID()`) and layer indices (`0..N`). The backend database **must never reorder, flatten, or alter element IDs or coordinate floats**.

---

## 8. Conclusion
The frontend implementation is clean, robust, and completely ready for backend integration. The service layer provides an ideal boundary to swap mock operations for HTTP requests without modifying UI components.

