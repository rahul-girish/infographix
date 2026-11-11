import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference_service.app.config import Settings, get_settings
from inference_service.app.routes.generation import router as generation_router
from inference_service.app.routes.health import router as health_router
from inference_service.app.routes.models import router as models_router
from inference_service.app.services.inference_manager import InferenceManager
from inference_service.app.services.memory_profiler import MemoryTracker


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    tracker = MemoryTracker()
    tracker.record("process_startup")
    manager = InferenceManager(memory_tracker=tracker)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = cfg
        app.state.inference_manager = manager
        if not cfg.SKIP_STARTUP_INIT:
            manager.initialize(cfg)
        yield
        manager.shutdown()

    app = FastAPI(
        title=cfg.SERVICE_NAME,
        description="Standalone ClearML-backed layout inference & HiGHS optimization service",
        version=cfg.SERVICE_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = cfg
    app.state.inference_manager = manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(generation_router)

    @app.get("/")
    def root():
        return {
            "service": cfg.SERVICE_NAME,
            "version": cfg.SERVICE_VERSION,
            "live": "/health/live",
            "ready": "/health/ready",
            "models": "/v1/models",
            "generate": "/v1/layouts/generate",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "inference_service.app.main:app",
        host="0.0.0.0",
        port=cfg.PORT,
        log_level=cfg.LOG_LEVEL.lower(),
    )
