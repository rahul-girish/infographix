# Infographix — Deployment & Supabase Setup Guide

This guide covers connecting **Supabase PostgreSQL**, running locally, and deploying the streamlined frontend and backend.

---

## 1. Supabase PostgreSQL Setup

1. In your Supabase project dashboard, open **Connect → Connection string → URI**.
2. Select **Session pooler** (port `5432`) so IPv4 hosts (Render, Railway, Docker, WSL) can connect reliably.
3. Convert the URI prefix to `postgresql+psycopg2://` and append `?sslmode=require`:
   ```env
   DATABASE_URL=postgresql+psycopg2://postgres.<project-ref>:<url-encoded-password>@aws-0-<region>.pooler.supabase.com:5432/postgres?sslmode=require
   ```
4. Place this in `backend/.env` for local testing or set it as an environment variable in your cloud host.
5. Initialize the `generation_runs` table (or simply start the FastAPI app, which creates missing tables on startup):
   ```bash
   PYTHONPATH=. python -m backend.app.db.init_db
   ```

---

## 2. Local Development

### Backend (FastAPI + ML Pipeline)
From the repository root:
```bash
uv pip install -r backend/requirements.txt
PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Health check: `http://localhost:8000/api/v1/health`
- Swagger UI: `http://localhost:8000/docs`

### Frontend (Vite + React)
```bash
cd frontend
npm install
npm run dev
```
- App URL: `http://localhost:5173` (proxies `/api/*` to `http://localhost:8000`).

---

## 3. Containerized Backend (`docker-compose.yml`)

The root [`docker-compose.yml`](../docker-compose.yml) builds [`backend/Dockerfile`](../backend/Dockerfile) from the repository root context so `backend/`, `src/`, `models/registry/`, `models/checkpoints/neural_layout/best_model.pt`, and `data/processed/features/v1/elements.parquet` are packaged together:

```bash
docker compose up --build -d
curl http://localhost:8000/api/v1/health
```

---

## 4. Separate Cloud Deployments (Monorepo)

- **Frontend (Vercel / Netlify / Cloudflare Pages):**
  - Root directory: `frontend`
  - Build command: `npm run build`
  - Output directory: `dist`
  - Env var: `VITE_API_BASE_URL=https://<your-backend-domain>/api`
- **Backend (Railway / Render / Fly.io / VPS):**
  - Build using `backend/Dockerfile` with build context `.` (repo root)
  - Env vars:
    - `DATABASE_URL=<supabase-session-pooler-uri>`
    - `CORS_ORIGINS=["https://<your-frontend-domain>"]`
    - `DEFAULT_ENGINE=adaptation`
