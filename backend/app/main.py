import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import backend.app.models  # noqa: F401
from backend.app.api.v1.router import api_router
from backend.app.core.config import settings
from backend.app.db.base import Base
from backend.app.db.session import SessionLocal, engine
from backend.app.governance.config import governance_settings
from backend.app.governance.exceptions import GovernanceError
from backend.app.governance.services.policy_service import PolicyService


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    if governance_settings.GOV_SEED_DEFAULT_POLICIES:
        db = SessionLocal()
        try:
            PolicyService.ensure_default_policies(db)
        finally:
            db.close()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Infographix AI Poster Layout Inference & MLOps Telemetry API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.exception_handler(GovernanceError)
async def governance_exception_handler(request: Request, exc: GovernanceError):
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "correlation_id": correlation_id,
            }
        },
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": "2.0.0",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
