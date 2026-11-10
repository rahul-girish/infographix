import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.core.config import settings
from backend.app.core.dependencies import get_current_user
from backend.app.models.user import User
from backend.app.models.asset import Asset
from backend.app.schemas.asset import AssetOut
from backend.app.services.storage_service import storage_service

router = APIRouter(prefix="/assets", tags=["Assets"])

@router.post("", response_model=AssetOut, status_code=status.HTTP_201_CREATED)
async def upload_asset(
    file: UploadFile = File(...),
    project_id: Optional[uuid.UUID] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image files are permitted (PNG, JPEG, WEBP, SVG)."
        )

    stored_name, file_path, file_size = storage_service.save_file(
        file, settings.ASSETS_DIR, max_size_bytes=10 * 1024 * 1024
    )

    asset = Asset(
        user_id=current_user.id,
        project_id=project_id,
        file_name=file.filename or "image.png",
        file_path=file_path,
        file_size=file_size,
        content_type=file.content_type or "image/png",
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)

    return AssetOut(
        id=str(asset.id),
        url=f"{settings.API_V1_STR}/assets/{asset.id}/content",
        fileName=asset.file_name,
        contentType=asset.content_type,
        size=asset.file_size
    )

@router.get("/{asset_id}/content")
def get_asset_content(
    asset_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")
    
    safe_path = storage_service.validate_safe_path(asset.file_path)
    return FileResponse(safe_path, media_type=asset.content_type)

@router.delete("/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_asset(
    asset_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    asset = db.query(Asset).filter(Asset.id == asset_id, Asset.user_id == current_user.id).first()
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")
    
    try:
        Path(asset.file_path).unlink(missing_ok=True)
    except Exception:
        pass

    db.delete(asset)
    db.commit()
