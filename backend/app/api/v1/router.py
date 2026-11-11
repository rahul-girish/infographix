from fastapi import APIRouter
from backend.app.api.v1.generation import router as generation_router
from backend.app.api.v1.health import router as health_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(generation_router)
