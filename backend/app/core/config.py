from pathlib import Path
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Infographix API"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = "infographix-development-secret-key-please-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # PostgreSQL Database
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/infographix"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    
    # Storage directories relative to backend root
    BASE_DIR: Path = BACKEND_DIR
    UPLOAD_DIR: Path = BACKEND_DIR / "data" / "uploads"
    DOCUMENTS_DIR: Path = BACKEND_DIR / "data" / "uploads" / "documents"
    ASSETS_DIR: Path = BACKEND_DIR / "data" / "uploads" / "assets"
    EXTRACTED_DIR: Path = BACKEND_DIR / "data" / "uploads" / "extracted"

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def setup_directories(self) -> None:
        """Ensure all upload and storage directories exist."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        self.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.setup_directories()
