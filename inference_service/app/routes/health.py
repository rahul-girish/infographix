from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from inference_service.app.config import get_settings
from inference_service.app.dependencies import get_inference_manager
from inference_service.app.schemas.layout import LiveHealthResponse, ReadyHealthResponse
from inference_service.app.services.inference_manager import InferenceManager

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live", response_model=LiveHealthResponse)
def liveness() -> LiveHealthResponse:
    """Lightweight liveness check requiring no model state or inference."""
    settings = get_settings()
    return LiveHealthResponse(
        status="alive",
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
    )


@router.get(
    "/ready",
    response_model=ReadyHealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadyHealthResponse}},
)
def readiness(manager: InferenceManager = Depends(get_inference_manager)):
    """Readiness probe reporting model, retriever, and data dependency status."""
    snapshot = manager.get_readiness_snapshot()
    code = status.HTTP_200_OK if manager.ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=snapshot)
