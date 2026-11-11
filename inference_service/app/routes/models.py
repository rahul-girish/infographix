from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from inference_service.app.dependencies import get_inference_manager
from inference_service.app.schemas.layout import ModelsListResponse
from inference_service.app.services.inference_manager import InferenceManager

router = APIRouter(prefix="/v1", tags=["Models"])


@router.get(
    "/models",
    response_model=ModelsListResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ModelsListResponse}},
)
def list_loaded_models(manager: InferenceManager = Depends(get_inference_manager)):
    """Report the actual ClearML models and data dependencies loaded in memory."""
    snapshot = manager.get_loaded_models_snapshot()
    code = status.HTTP_200_OK if manager.ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=snapshot)
