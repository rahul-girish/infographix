from fastapi import Request
from inference_service.app.services.inference_manager import InferenceManager


def get_inference_manager(request: Request) -> InferenceManager:
    """Retrieve the singleton InferenceManager stored on FastAPI app state."""
    return request.app.state.inference_manager
