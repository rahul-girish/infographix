import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.core.dependencies import get_current_user
from backend.app.models.user import User
from backend.app.models.document import UploadedDocument
from backend.app.schemas.section import ResearchSectionSchema
from backend.app.services.project_service import project_service
from backend.app.services.document_parser import document_parser_service

router = APIRouter(tags=["Extraction"])

@router.post("/projects/{project_id}/extract", response_model=List[ResearchSectionSchema])
def trigger_extraction(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = project_service.get_project_model(db, project_id, current_user.id)
    latest_doc = (
        db.query(UploadedDocument)
        .filter(UploadedDocument.project_id == project.id)
        .order_by(UploadedDocument.created_at.desc())
        .first()
    )
    if not latest_doc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No uploaded document found for this project."
        )

    try:
        metadata, extracted_sections = document_parser_service.parse(latest_doc.file_path, latest_doc.file_name)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document extraction failed: {str(e)}"
        )

    return project_service.save_sections(db, project.id, current_user.id, extracted_sections)

