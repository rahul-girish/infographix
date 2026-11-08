import uuid
from typing import List, Optional, Any
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.core.dependencies import get_current_user
from backend.app.models.user import User
from backend.app.schemas.project import ProjectCreate, ProjectUpdate, ProjectRename, ProjectStatusUpdate, ProjectOut
from backend.app.schemas.section import ResearchSectionSchema
from backend.app.services.project_service import project_service

router = APIRouter(prefix="/projects", tags=["Projects"])

@router.get("", response_model=List[ProjectOut])
def list_projects(
    filter: Optional[str] = Query(None, alias="filter"),
    search: Optional[str] = Query(None, alias="search"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return project_service.list_projects(db, current_user.id, status_filter=filter, search=search)

@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(
    project_in: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return project_service.create_project(db, current_user.id, project_in)

@router.get("/{project_id}", response_model=ProjectOut)
def get_project(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return project_service.get_project(db, project_id, current_user.id)

@router.patch("/{project_id}/rename", response_model=ProjectOut)
def rename_project(
    project_id: uuid.UUID,
    rename_in: ProjectRename,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return project_service.rename_project(db, project_id, current_user.id, rename_in.title)

@router.post("/{project_id}/duplicate", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def duplicate_project(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return project_service.duplicate_project(db, project_id, current_user.id)

@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    project_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project_service.delete_project(db, project_id, current_user.id)

@router.patch("/{project_id}/status", response_model=ProjectOut)
def update_status(
    project_id: uuid.UUID,
    status_in: ProjectStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return project_service.update_status(db, project_id, current_user.id, status_in.status)

@router.put("/{project_id}/sections", response_model=List[ResearchSectionSchema])
def save_sections(
    project_id: uuid.UUID,
    sections: List[ResearchSectionSchema],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return project_service.save_sections(db, project_id, current_user.id, sections)

@router.put("/{project_id}/poster")
def save_poster(
    project_id: uuid.UUID,
    poster: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return project_service.save_poster(db, project_id, current_user.id, poster)

