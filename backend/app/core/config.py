from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "Infographix MLOps API"
    API_V1_STR: str = "/api/v1"

    # Database: Supabase PostgreSQL URI in production, local SQLite fallback for offline dev/tests
    DATABASE_URL: str = f"sqlite:///{(BACKEND_DIR / 'infographix_local.db').as_posix()}"

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    # ML Pipeline Artifact Paths
    BASE_DIR: Path = BACKEND_DIR
    PROJECT_ROOT_DIR: Path = PROJECT_ROOT
    RETRIEVER_REGISTRY_ROOT: Path = PROJECT_ROOT / "models" / "registry"
    ELEMENTS_PARQUET_PATH: Path = PROJECT_ROOT / "data" / "processed" / "features" / "v1" / "elements.parquet"
    NEURAL_CHECKPOINT_PATH: Path = PROJECT_ROOT / "models" / "checkpoints" / "neural_layout" / "best_model.pt"
    DEFAULT_ENGINE: str = "adaptation"  # "adaptation" | "neural"

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
