import os
import uuid
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException, status
from backend.app.core.config import settings

class StorageService:
    @staticmethod
    def save_file(file: UploadFile, target_dir: Path, max_size_bytes: int = 20 * 1024 * 1024) -> tuple[str, str, int]:
        target_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(file.filename or "").suffix.lower()
        unique_name = f"{uuid.uuid4()}{ext}"
        target_path = target_dir / unique_name
        
        size = 0
        with open(target_path, "wb") as buffer:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > max_size_bytes:
                    buffer.close()
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {max_size_bytes // (1024 * 1024)} MB."
                    )
                buffer.write(chunk)
                
        return unique_name, str(target_path), size

    @staticmethod
    def validate_safe_path(file_path: str) -> Path:
        resolved = Path(file_path).resolve()
        if not str(resolved).startswith(str(settings.BASE_DIR.resolve())):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to the requested file path is forbidden."
            )
        if not resolved.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Requested file does not exist on disk."
            )
        return resolved

storage_service = StorageService()

