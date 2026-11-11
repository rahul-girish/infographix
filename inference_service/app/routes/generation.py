import logging

from fastapi import APIRouter, Depends, HTTPException, status

from inference_service.app.dependencies import get_inference_manager
from inference_service.app.schemas.layout import GenerateLayoutRequest, GenerateLayoutResponse
from inference_service.app.services.inference_manager import (
    InferenceExecutionError,
    InferenceManager,
    ServiceNotReadyError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/layouts", tags=["Layout Generation"])


@router.post("/generate", response_model=GenerateLayoutResponse)
def generate_layout(
    payload: GenerateLayoutRequest,
    manager: InferenceManager = Depends(get_inference_manager),
) -> GenerateLayoutResponse:
    """Generate a validated poster layout using the ClearML-loaded neural or adaptation engine."""
    try:
        result = manager.generate(payload)
        return GenerateLayoutResponse(**result)
    except ServiceNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service is not ready to serve requests.",
        ) from exc
    except InferenceExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unhandled inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during layout generation.",
        ) from exc
