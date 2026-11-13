from pathlib import Path
from typing import List, Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SERVICE_ROOT.parent


class Settings(BaseSettings):
    """Runtime configuration for the standalone Infographix inference service."""

    SERVICE_NAME: str = "Infographix Inference Service"
    SERVICE_VERSION: str = "1.0.0"
    PORT: int = 8001
    LOG_LEVEL: str = "INFO"

    # ClearML Registry credentials & endpoints (injected via environment variables)
    CLEARML_API_ACCESS_KEY: str = ""
    CLEARML_API_SECRET_KEY: str = ""
    CLEARML_API_HOST: str = "https://api.clear.ml"
    CLEARML_WEB_HOST: str = "https://app.clear.ml"
    CLEARML_FILES_HOST: str = "https://files.clear.ml"
    CLEARML_PROJECT_NAME: str = "Infographix"
    CLEARML_FORCE_DOWNLOAD: bool = False
    CLEARML_CACHE_DIR: Optional[str] = None

    # Explicit model selection (prefer immutable IDs or semantic versions)
    RETRIEVER_MODEL_FAMILY: str = "poster_layout_retriever"
    RETRIEVER_MODEL_ID: Optional[str] = None
    RETRIEVER_MODEL_VERSION: Optional[str] = None
    RETRIEVER_LIFECYCLE_STAGE: Optional[str] = None

    NEURAL_MODEL_FAMILY: str = "poster_layout_generator"
    NEURAL_MODEL_ID: Optional[str] = None
    NEURAL_MODEL_VERSION: Optional[str] = None
    NEURAL_LIFECYCLE_STAGE: Optional[str] = None

    DEFAULT_ENGINE: Literal["adaptation", "neural"] = "neural"
    TORCH_DEVICE: str = "cpu"

    # Template element geometry table dependency (data/processed/features/v1/elements.parquet)
    ELEMENTS_PARQUET_PATH: str = "/app/data_cache/elements.parquet"
    ELEMENTS_PARQUET_URL: Optional[str] = None
    ELEMENTS_PARQUET_SHA256: Optional[str] = None

    # Optional private AWS S3 source for elements.parquet
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_DEFAULT_REGION: Optional[str] = None
    S3_BUCKET_NAME: Optional[str] = None
    S3_OBJECT_KEY: Optional[str] = None

    # Optional ClearML Dataset or Minidvc source for elements.parquet
    CLEARML_DATASET_ID: Optional[str] = None
    CLEARML_DATASET_NAME: Optional[str] = None
    CLEARML_DATASET_VERSION: Optional[str] = None

    MINIDVC_SNAPSHOT: Optional[str] = None
    MINIDVC_CLIENT_ID: Optional[str] = None
    MINIDVC_CLIENT_SECRET: Optional[str] = None
    MINIDVC_REFRESH_TOKEN: Optional[str] = None

    # Startup behavior
    EAGER_LOAD_ADAPTATION_DATA: bool = False
    SKIP_STARTUP_INIT: bool = False

    CORS_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=str(SERVICE_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
