import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.core.dependencies import get_current_user
from backend.app.models.user import User
from backend.app.models.submission import Submission
from backend.app.schemas.submission import SubmissionOut
from backend.app.services.project_service import project_service

router = APIRouter(tags=["Submissions"])

@router.post("/projects/{project_id}/submit", response_model=SubmissionOut, status_code=status.HTTP_201_CREATED)
def submit_project(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = project_service.get_project_model(db, project_id, current_user.id)
    
    # Mark status as submitted
    project.status = "submitted"
    project.updated_at = datetime.now(timezone.utc)
    
    # Create submission record with frozen poster snapshot
    snapshot = None
    if project.poster:
        snapshot = {
            "title": project.poster.title,
            "width": project.poster.width,
            "height": project.poster.height,
            "orientation": project.poster.orientation,
            "templateId": project.poster.template_id,
            "columns": project.poster.columns,
            "elements": project.poster.elements
        }

    submission = Submission(
        project_id=project.id,
        user_id=current_user.id,
        status="submitted",
        poster_snapshot=snapshot
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    return SubmissionOut(
        id=str(submission.id),
        projectId=str(submission.project_id),
        status=submission.status,
        submittedAt=submission.submitted_at.isoformat()
    )

@router.get("/projects/{project_id}/submission", response_model=SubmissionOut)
def get_submission(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = project_service.get_project_model(db, project_id, current_user.id)
    submission = (
        db.query(Submission)
        .filter(Submission.project_id == project.id, Submission.user_id == current_user.id)
        .order_by(Submission.submitted_at.desc())
        .first()
    )
    if not submission:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No submission record found for this project.")

    return SubmissionOut(
        id=str(submission.id),
        projectId=str(submission.project_id),
        status=submission.status,
        submittedAt=submission.submitted_at.isoformat()
    )
