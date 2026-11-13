# INFOGRAPHIX — MASTER PROJECT CONTEXT AND SYSTEM UNDERSTANDING

## 1. ROLE AND OBJECTIVE

You are now part of the development team working on a project called:

# INFOGRAPHIX
### AI-Powered Technical Poster and Infographic Generator

Your first responsibility is to understand the complete project vision, its objectives, technical direction, existing development status, and intended system architecture.

You must understand the project before making any code changes or proposing implementation decisions.

This document establishes the foundational context for all future development activities.

**Important instructions:**

- Do not start coding.
- Do not modify existing files.
- Do not install dependencies.
- Do not create or modify database schemas.
- Do not implement APIs.
- Do not assume that proposed features have already been implemented.
- Do not make changes to the existing Git repository.
- Treat this document as the project's intended product direction, not proof of existing functionality.

Your task is to understand the project and establish a clear distinction between its vision, current implementation, and future development.

---

# 2. PROJECT BACKGROUND

Creating technical and scientific posters is a common requirement in academic and research environments.

Students, researchers, and professionals often need to convert lengthy research papers, technical reports, experimental results, and project documentation into visually structured posters.

This process generally involves:

1. Reading and understanding research documents.
2. Identifying important information.
3. Extracting relevant research sections.
4. Organizing the content.
5. Deciding the poster layout.
6. Designing visual elements.
7. Arranging text, figures, diagrams, and tables.
8. Reviewing and editing the final poster.

This process can be time-consuming and requires both technical understanding and design skills.

Infographix aims to simplify this workflow through a document-aware, structured, and editable technical poster creation platform.

The long-term objective is to combine document understanding, layout intelligence, machine learning, and an interactive visual editing environment.

---

# 3. PROJECT VISION

Infographix is intended to become an AI-assisted platform that transforms research documents and technical information into structured, visually organized, and editable technical posters.

The fundamental workflow is:

**Research Document → Information Extraction → Content Organization → Poster Layout → Visual Editing → Final Poster**

The platform should support users throughout the poster creation process rather than simply generating a final image.

Users must retain control over the content, design, layout, and individual poster elements.

The system should eventually use machine learning to assist with content understanding, layout recommendations, and intelligent poster generation.

However, these capabilities must be developed progressively and must not be represented as completed functionality until they actually exist.

---

# 4. TARGET USERS

The primary target audience includes:

- Undergraduate students.
- Postgraduate students.
- Research scholars.
- Academic project teams.
- Faculty members.
- Scientific researchers.
- Engineering professionals.
- Technical conference participants.

Typical use cases include:

- Converting research papers into academic posters.
- Creating scientific conference posters.
- Preparing engineering project posters.
- Creating AI/ML research posters.
- Presenting experimental results.
- Transforming technical reports into structured visual presentations.

The initial product scope is specifically focused on **technical, scientific, and academic posters**.

It is not intended to become a general-purpose graphic design platform like Canva in the initial release.

---

# 5. CORE PRODUCT WORKFLOW

The intended user journey is described below.

## Stage 1 — User Access

Users should be able to:

- Register.
- Log in.
- Access their dashboard.
- Create multiple projects.
- Save their work.
- Reopen existing projects.

Each project should belong to its respective user.

## Stage 2 — Project Creation

Users create a new technical poster project.

They should be able to:

- Enter a project name.
- Upload a research document.
- Start without a document if they prefer.
- Select the intended poster type.

## Stage 3 — Document Upload

The platform should support research document uploads.

Initial formats:

- PDF.
- DOCX.

The system should eventually process these documents and extract relevant information.

## Stage 4 — Information Extraction

The system should identify relevant information from uploaded documents.

Expected information includes:

| Category | Description |
|---|---|
| Title | Research or project title |
| Authors | Research authors |
| Affiliations | Institutions and departments |
| Abstract | Research summary |
| Introduction | Research background |
| Problem statement | Identified problem |
| Objectives | Research objectives |
| Methodology | Research methodology |
| Results | Experimental findings |
| Discussion | Interpretation |
| Conclusion | Final conclusions |
| References | Bibliography |
| Keywords | Important technical terms |
| Figures | Research figures |
| Tables | Experimental tables |

The extracted information should be organized into structured sections.

Extraction results must not be treated as automatically correct.

Users should be able to review and correct the extracted content.

The system should preserve document source references, such as page numbers, wherever technically possible.

## Stage 5 — Content Review

Users should be able to:

- Edit extracted text.
- Add missing information.
- Remove irrelevant information.
- Reorder sections.
- Include or exclude sections.
- Correct titles and author details.
- Review figures and tables.

The user must have final control over the content used in the poster.

## Stage 6 — Poster Configuration

Users should configure:

- Poster dimensions.
- Portrait or landscape orientation.
- Poster template.
- Number of columns.
- Color palette.
- Background.
- Typography.
- Content sections.

Initial template categories:

1. Research Paper Poster.
2. Scientific Conference Poster.
3. Engineering Project Poster.
4. AI/ML Research Poster.
5. Experimental Results Poster.

## Stage 7 — Poster Editor

This is one of the most important components of Infographix.

The platform should provide a full visual poster editing environment.

The intended interface consists of three main panels.

### Left panel — Assets and elements

- Templates.
- Text.
- Headings.
- Sections.
- Images.
- Shapes.
- Icons.
- Uploaded assets.

### Center panel — Editable canvas

- Poster canvas.
- Drag-and-drop elements.
- Selection.
- Resizing.
- Rotation.
- Alignment guides.
- Grid.
- Snapping.
- Zoom.
- Fit to screen.

### Right panel — Properties

Contextual editing controls for the selected element.

For text:

- Text content.
- Font family.
- Font size.
- Font weight.
- Color.
- Alignment.
- Spacing.
- Bold, italic, underline.

For images:

- Replace.
- Resize.
- Crop.
- Position.
- Rotation.
- Aspect ratio.

For shapes:

- Fill.
- Border.
- Border width.
- Position.
- Dimensions.
- Rotation.
- Opacity.

### Additional editing capabilities

- Add and delete elements.
- Duplicate elements.
- Move and resize.
- Layer management.
- Lock and unlock.
- Hide and show.
- Group and ungroup.
- Undo and redo.
- Copy and paste.
- Zoom.
- Alignment and snapping.

**Critical requirement: Every poster element must remain independently editable.**

For example, a methodology diagram should ideally be represented by separate shapes, arrows, and text objects rather than a single flattened image.

The poster must retain a structured representation of its elements.

## Stage 8 — Preview and Saving

Users should be able to:

- Preview the complete poster.
- Return to editing.
- Save the project.
- Reopen the project.
- Continue editing.
- Submit the project.

The system must preserve the poster's editable structure when saving and reopening.

## Stage 9 — Submission

When a user submits a project, the frontend should communicate the structured project data to the backend.

The backend should eventually validate and store:

- Project information.
- Document metadata.
- Extracted content.
- Poster structure.
- Poster element properties.
- Asset references.
- Submission information.

Saving and submitting should be treated as separate operations.

---

# 6. PRODUCT ARCHITECTURE

Infographix is intended to be a full-stack web application with a future AI/ML integration layer.

The major components are:

### A. Frontend

The frontend provides the complete user interface.

Expected technology direction:

- React.
- Vite.
- TypeScript.
- Tailwind CSS.
- React-Konva.
- Zustand.

Responsibilities:

- User interface.
- Navigation.
- Authentication screens.
- Dashboard.
- Document upload.
- Extraction review.
- Poster configuration.
- Poster editor.
- Preview.
- User interactions.
- Communication with backend APIs.

### B. Backend

The backend manages application logic and communication between the frontend, database, file storage, and future ML services.

Expected technology direction:

- Python.
- FastAPI.
- REST APIs.

Responsibilities:

- Authentication.
- User management.
- Project management.
- Document upload.
- Document processing.
- Information extraction.
- Asset management.
- Poster data storage.
- Poster retrieval.
- Submission management.
- Future ML service integration.

### C. Database

Expected database:

PostgreSQL.

Responsibilities:

- User information.
- Project metadata.
- Uploaded document metadata.
- Extracted research information.
- Poster metadata.
- Poster editable structures.
- Template information.
- Submission records.
- Project versions, if required.

The database design should preserve the complete editable poster structure.

Structured JSON/JSONB storage may be appropriate for poster elements and their properties, while relational tables should manage users, projects, documents, and other entities where relationships and querying matter.

### D. File Storage

Uploaded research documents and image assets should be stored separately from the primary relational database.

The database should retain their metadata and storage references.

### E. AI/ML Integration Layer

The long-term platform should integrate machine learning capabilities for:

- Document understanding.
- Research content analysis.
- Content summarization.
- Layout analysis.
- Layout recommendations.
- Intelligent content placement.
- Poster generation.

These capabilities are future development objectives, not assumptions about the current application.

---

# 7. EXISTING ML/RESEARCH PROJECT CONTEXT

Infographix is also an AI/ML-oriented project with a separate research and experimentation component.

The existing research work includes the SciPostLayout v2 dataset.

The project has investigated:

- Poster layout analysis.
- Bounding-box annotations.
- Document layout categories.
- Feature engineering.
- Layout clustering.
- Nearest-neighbour analysis.
- MLflow-based experiment tracking.
- Evaluation infrastructure.

The dataset has approximately:

- 7,855 poster images.
- 192,697 annotations.
- 9 annotation categories.

The project has previously explored K-means clustering and nearest-neighbour layout analysis.

Important distinction:

These experiments do not automatically constitute a production-ready poster-generation model.

Do not assume that a trained and deployable poster-generation model currently exists.

Do not modify existing notebooks, datasets, or ML research files unless explicitly instructed.

The web application should be architected so that future ML services can be integrated without requiring a complete redesign.

---

# 8. CURRENT DEVELOPMENT DIRECTION

The current development strategy is to build the platform progressively.

The immediate priority is to establish a functional full-stack application.

The initial release should support:

1. User authentication.
2. Project management.
3. Research document upload.
4. Document text extraction.
5. Extraction review.
6. Poster configuration.
7. Fully editable poster canvas.
8. Poster saving and retrieval.
9. Backend submission.
10. Structured data persistence.

Actual AI-powered poster generation will be developed separately when the required models and pipelines are ready.

The initial application can use conventional document-processing libraries and deterministic operations.

Do not unnecessarily introduce complex AI services into the initial backend.

---

# 9. TECHNOLOGY AND DEVELOPMENT PRINCIPLES

The development team should follow these principles:

### Modularity

Frontend, backend, database, document processing, and ML services should remain logically separated.

### API-driven architecture

The frontend should communicate with the backend through clearly defined APIs.

### Data consistency

The poster data model must be consistent between frontend serialization and backend storage.

### Editable-first design

Never flatten the poster into a single image as the primary representation.

### Model-independent architecture

The application should work without an ML model and support future ML integration.

### Security

Users must only access their own projects and associated documents unless explicit sharing functionality is implemented.

Uploaded files must be validated and handled safely.

### Maintainability

Use clear folder structures, modular services, versioned API contracts, and automated testing.

### Testability

Every major workflow should have appropriate tests.

### Deployment readiness

The application will eventually use Docker and cloud deployment.

Docker and deployment are planned responsibilities of the backend developer and will be addressed in a later development stage.

---

# 10. CURRENT TEAM RESPONSIBILITIES

The project work is divided between two primary development responsibilities.

### Frontend developer

Responsible for:

- Frontend foundation.
- Application pages.
- Dashboard.
- Authentication interfaces.
- Document upload interfaces.
- Extraction review.
- Poster configuration.
- Full poster editor.
- Preview.
- Frontend state management.
- Mock services.

### Backend developer

Responsible for:

- Backend foundation.
- FastAPI application.
- PostgreSQL database.
- Database schema and migrations.
- Authentication APIs.
- Project management APIs.
- Document upload and processing.
- Information extraction.
- Asset management.
- Poster storage and retrieval.
- Submission APIs.
- Frontend-backend integration.
- Testing.
- Docker and deployment.

The backend developer will inspect the completed frontend and establish the actual API contract and database structure based on its implementation.

Do not assume that all these components have already been completed.

---

# 11. PROJECT DEVELOPMENT STAGES

The intended development roadmap is:

### Stage 1 — Frontend development

Complete the user interface and editable poster editor.

### Stage 2 — Frontend audit

Inspect the actual frontend implementation, data models, mock services, and API requirements.

### Stage 3 — Backend architecture

Establish the API contract, database design, document-processing architecture, and backend folder structure.

### Stage 4 — Backend development

Implement FastAPI services, PostgreSQL persistence, authentication, document handling, and poster management.

### Stage 5 — Integration

Connect the frontend to the backend and replace mock services with real API operations.

### Stage 6 — Testing

Test the complete user journey, persistence, authentication, document processing, and poster reconstruction.

### Stage 7 — Containerization and deployment

Use Docker and Docker Compose, followed by deployment to an appropriate hosting platform.

### Stage 8 — AI/ML integration

Integrate trained and evaluated ML models when ready.

---

# 12. PROJECT STATUS AND IMPORTANT DISTINCTIONS

The following distinctions must be maintained throughout development.

| Category | Meaning |
|---|---|
| Product vision | What Infographix is intended to become |
| Planned feature | What we intend to implement |
| Frontend implementation | What is actually present in the frontend |
| Mock functionality | Simulated functionality using sample data |
| Backend implementation | What is actually supported by server-side code |
| ML capability | What has actually been trained, evaluated, and integrated |

Never describe a planned feature as an implemented feature.

Never claim that mock document extraction is real extraction.

Never claim that a poster-generation model exists unless its implementation and deployment have been verified.

---

# 13. YOUR FIRST TASK — UNDERSTAND AND RECORD THIS PROJECT CONTEXT

For now, do not implement anything.

Read this entire document and establish your understanding of Infographix.

Then provide a structured acknowledgement containing:

1. Your understanding of the project.
2. The problem the platform is intended to solve.
3. The primary users.
4. The complete user workflow.
5. The major frontend components.
6. The major backend components.
7. The database's role.
8. The role of future AI/ML integration.
9. The distinction between current implementation and future objectives.
10. The development stages.

Identify any important ambiguities in the project context.

Do not invent missing project details.

If the existing repository is available, you may use it only to understand the current project structure and identify the existing implementation. Do not modify any files.

Finally, save this document as:

`docs/INFOGRAPHIX_PROJECT_CONTEXT.md`

If the `docs` directory does not exist, create it.

Preserve the meaning and structure of this document.

This file will serve as the foundational project reference for future frontend, backend, database, testing, and deployment work.

**Final instruction: Understand the project first. Implementation begins only after receiving the next explicit development task.**

