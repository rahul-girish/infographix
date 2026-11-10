import uuid
from typing import List
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.core.config import settings
from backend.app.core.dependencies import get_current_user
from backend.app.models.user import User
from backend.app.models.document import UploadedDocument
from backend.app.schemas.document import DocumentOut, DocumentUploadResponse
from backend.app.schemas.section import ResearchSectionSchema
from backend.app.services.project_service import project_service
from backend.app.services.storage_service import storage_service
from backend.app.services.document_parser import document_parser_service

router = APIRouter(tags=["Documents"])

@router.post("/projects/{project_id}/documents", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    project_id: uuid.UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = project_service.get_project_model(db, project_id, current_user.id)
    filename = file.filename or "uploaded_paper.pdf"
    ext = Path(filename).suffix.lower()
    
    if ext not in [".pdf", ".docx", ".doc"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported document format. Please upload a PDF or DOCX file."
        )

    # Save to disk
    stored_name, file_path, file_size = storage_service.save_file(
        file, settings.DOCUMENTS_DIR, max_size_bytes=20 * 1024 * 1024
    )

    # Deterministic Extraction
    try:
        metadata, extracted_sections = document_parser_service.parse(file_path, filename)
        extraction_status = "extracted"
        page_count = metadata.get("page_count", 1)
        error_msg = None
    except Exception as e:
        extraction_status = "failed"
        page_count = None
        error_msg = str(e)
        extracted_sections = []

    # Record in database
    doc_record = UploadedDocument(
        project_id=project.id,
        file_name=filename,
        file_path=file_path,
        file_size=file_size,
        content_type=file.content_type or "application/octet-stream",
        page_count=page_count,
        status=extraction_status,
        error_message=error_msg,
    )
    db.add(doc_record)
    
    # Attach document to project
    project.document_name = filename
    db.commit()
    db.refresh(doc_record)

    # Save extracted sections if extraction was successful
    if extracted_sections:
        project_service.save_sections(db, project.id, current_user.id, extracted_sections)

    doc_out = DocumentOut(
        id=str(doc_record.id),
        projectId=str(doc_record.project_id),
        fileName=doc_record.file_name,
        fileSize=doc_record.file_size,
        contentType=doc_record.content_type,
        pageCount=doc_record.page_count,
        status=doc_record.status,
        uploadedAt=doc_record.created_at.isoformat(),
    )

    return DocumentUploadResponse(
        document=doc_out,
        name=filename,
        sections=extracted_sections
    )

@router.get("/projects/{project_id}/documents", response_model=List[DocumentOut])
def list_documents(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = project_service.get_project_model(db, project_id, current_user.id)
    docs = db.query(UploadedDocument).filter(UploadedDocument.project_id == project.id).all()
    return [
        DocumentOut(
            id=str(d.id),
            projectId=str(d.project_id),
            fileName=d.file_name,
            fileSize=d.file_size,
            contentType=d.content_type,
            pageCount=d.page_count,
            status=d.status,
            uploadedAt=d.created_at.isoformat(),
        )
        for d in docs
    ]

@router.get("/documents/{document_id}/download")
def download_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = db.query(UploadedDocument).filter(UploadedDocument.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    
    # Verify project ownership
    project_service.get_project_model(db, doc.project_id, current_user.id)
    safe_path = storage_service.validate_safe_path(doc.file_path)
    return FileResponse(safe_path, filename=doc.file_name, media_type=doc.content_type)
