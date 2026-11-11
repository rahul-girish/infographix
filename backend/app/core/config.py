from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "Infographix MLOps API"
    API_V1_STR: str = "/api/v1"
    PORT: int = 8000

    # Database: Supabase PostgreSQL URI in production, local SQLite fallback for offline dev/tests
    DATABASE_URL: str = f"sqlite:///{(BACKEND_DIR / 'infographix_local.db').as_posix()}"

    # Standalone ClearML-backed Inference Service URL
    INFERENCE_SERVICE_URL: Optional[str] = None
    INFERENCE_TIMEOUT_SECONDS: float = 120.0
    DEFAULT_ENGINE: str = "adaptation"

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
