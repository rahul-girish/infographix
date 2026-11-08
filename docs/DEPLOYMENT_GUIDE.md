# Infographix — Deployment & Operations Guide

This guide covers running Infographix in local development, containerized testing via Docker Compose, and deploying to cloud infrastructure.

---

## 1. Local Development (WSL2 / Linux / Windows)

### 1.1 Prerequisites
* Python 3.12+ (or `uv` package manager)
* Node.js 20+ and npm
* PostgreSQL 16+ (or Docker Desktop)

### 1.2 Start PostgreSQL Database
Using Docker:
```bash
docker run -d --name infographix-postgres \
  -p 5432:5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=infographix \
  -v infographix_pgdata:/var/lib/postgresql/data \
  postgres:16-alpine
```

### 1.3 Start Backend Service (FastAPI)
From the repository root:
```bash
# 1. Install dependencies
uv pip install -r backend/requirements.txt

# 2. Run database initialization
PYTHONPATH=. python -m backend.app.db.init_db

# 3. Start Uvicorn development server
PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive API Documentation (Swagger UI): `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/api/v1/health`

### 1.4 Start Frontend Client (Vite + React)
From the repository root:
```bash
cd frontend
npm install
npm run dev
```
* Frontend runs at: `http://localhost:5173`
* Requests to `/api/*` automatically proxy to the FastAPI server at `http://localhost:8000`.

---

## 2. Containerized Deployment (Docker Compose)

The repository provides a production-grade multi-service orchestration setup via [`docker-compose.yml`](file:///wsl.localhost/Ubuntu-24.04/home/raksh/projects/infographix/docker-compose.yml):

```bash
# Build and launch all services in detached mode
docker compose up --build -d

# Check service logs
docker compose logs -f backend

# Verify health status
curl http://localhost:8000/api/v1/health
```

### Volume Persistence:
* `postgres_data`: Persists all PostgreSQL database tables, user records, and poster canvas JSONB across container restarts.
* `uploads_data`: Persists uploaded PDFs, DOCXs, and image assets mounted to `/app/backend/data/uploads`.

---

## 3. Production Hardening Checklist

1. **Secret Key**: Generate a secure 64-character secret key for `SECRET_KEY`:
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```
2. **CORS Configuration**: Restrict `CORS_ORIGINS` in `backend/.env` to the specific production frontend domain.
3. **Database Credentials**: Use strong passwords for `POSTGRES_USER` and `POSTGRES_PASSWORD` injected via container environment secrets.
4. **File Storage Quotas**: The backend automatically enforces a 20 MB cap on documents and 10 MB on image assets to prevent disk exhaustion.

