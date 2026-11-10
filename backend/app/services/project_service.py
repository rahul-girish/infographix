import uuid
import copy
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc
from fastapi import HTTPException, status

from backend.app.models.project import Project
from backend.app.models.section import ResearchSection
from backend.app.models.poster import PosterDocument
from backend.app.schemas.project import ProjectCreate, ProjectUpdate, ProjectOut
from backend.app.schemas.section import ResearchSectionSchema
from backend.app.schemas.poster import PosterDocumentSchema
from backend.app.services.templates import POSTER_TEMPLATES, DEFAULT_SECTIONS, create_starter_elements

class ProjectService:
    @staticmethod
    def _format_project_out(project: Project) -> ProjectOut:
        sections_out = [
            ResearchSectionSchema(
                id=s.section_key if s.section_key else str(s.id),
                title=s.title,
                content=s.content,
                included=s.included,
                pageNumber=s.page_number
            )
            for s in project.sections
        ]
        
        poster_out = None
        if project.poster:
            poster_out = PosterDocumentSchema(
                id=str(project.poster.id),
                project_id=str(project.poster.project_id),
                title=project.poster.title,
                width=project.poster.width,
                height=project.poster.height,
                orientation=project.poster.orientation,
                background=project.poster.background,
                template_id=project.poster.template_id,
                columns=project.poster.columns,
                elements=project.poster.elements,
                updated_at=project.poster.updated_at.isoformat()
            )

        return ProjectOut(
            id=str(project.id),
            title=project.title,
            status=project.status,
            updated_at=project.updated_at.isoformat(),
            template_id=project.template_id,
            document_name=project.document_name,
            sections=sections_out,
            poster=poster_out
        )

    @classmethod
    def list_projects(
        cls, db: Session, user_id: uuid.UUID, status_filter: Optional[str] = None, search: Optional[str] = None
    ) -> List[ProjectOut]:
        query = db.query(Project).filter(Project.user_id == user_id)
        if status_filter and status_filter != "all":
            query = query.filter(Project.status == status_filter)
        if search:
            query = query.filter(Project.title.ilike(f"%{search}%"))
        
        projects = query.order_by(desc(Project.updated_at)).all()
        return [cls._format_project_out(p) for p in projects]

    @classmethod
    def get_project_model(cls, db: Session, project_id: uuid.UUID, user_id: uuid.UUID) -> Project:
        project = db.query(Project).filter(Project.id == project_id, Project.user_id == user_id).first()
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
        return project

    @classmethod
    def get_project(cls, db: Session, project_id: uuid.UUID, user_id: uuid.UUID) -> ProjectOut:
        project = cls.get_project_model(db, project_id, user_id)
        return cls._format_project_out(project)

    @classmethod
    def create_project(cls, db: Session, user_id: uuid.UUID, project_in: ProjectCreate) -> ProjectOut:
        title = project_in.title or "Untitled research poster"
        template_id = project_in.templateId or "research"
        template = next((t for t in POSTER_TEMPLATES if t["id"] == template_id), POSTER_TEMPLATES[0])

        new_project = Project(
            user_id=user_id,
            title=title,
            status="draft",
            template_id=template["id"]
        )
        db.add(new_project)
        db.flush()

        # Add starter sections
        for idx, sec in enumerate(DEFAULT_SECTIONS):
            section = ResearchSection(
                project_id=new_project.id,
                section_key=sec["id"],
                title=sec["title"],
                content=sec["content"],
                included=True,
                order_index=idx
            )
            db.add(section)

        # Add default poster document
        starter_elements = create_starter_elements(title.upper(), template["accent"])
        poster = PosterDocument(
            project_id=new_project.id,
            title=title,
            width=1000,
            height=700,
            orientation="landscape",
            background=template["background"],
            template_id=template["id"],
            columns=template["columns"],
            elements=starter_elements
        )
        db.add(poster)

        db.commit()
        db.refresh(new_project)
        return cls._format_project_out(new_project)

    @classmethod
    def rename_project(cls, db: Session, project_id: uuid.UUID, user_id: uuid.UUID, title: str) -> ProjectOut:
        project = cls.get_project_model(db, project_id, user_id)
        project.title = title
        if project.poster:
            project.poster.title = title
        project.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(project)
        return cls._format_project_out(project)

    @classmethod
    def duplicate_project(cls, db: Session, project_id: uuid.UUID, user_id: uuid.UUID) -> ProjectOut:
        source = cls.get_project_model(db, project_id, user_id)
        title = f"{source.title} (copy)"
        
        copy_proj = Project(
            user_id=user_id,
            title=title,
            status="draft",
            template_id=source.template_id,
            document_name=source.document_name
        )
        db.add(copy_proj)
        db.flush()

        # Duplicate sections
        for sec in source.sections:
            copy_sec = ResearchSection(
                project_id=copy_proj.id,
                section_key=sec.section_key,
                title=sec.title,
                content=sec.content,
                included=sec.included,
                order_index=sec.order_index,
                page_number=sec.page_number
            )
            db.add(copy_sec)

        # Duplicate poster with new element UUIDs
        if source.poster:
            new_elements = copy.deepcopy(source.poster.elements)
            for elem in new_elements:
                elem["id"] = str(uuid.uuid4())
            
            copy_poster = PosterDocument(
                project_id=copy_proj.id,
                title=title,
                width=source.poster.width,
                height=source.poster.height,
                orientation=source.poster.orientation,
                background=source.poster.background,
                template_id=source.poster.template_id,
                columns=source.poster.columns,
                elements=new_elements
            )
            db.add(copy_poster)

        db.commit()
        db.refresh(copy_proj)
        return cls._format_project_out(copy_proj)

    @classmethod
    def delete_project(cls, db: Session, project_id: uuid.UUID, user_id: uuid.UUID) -> None:
        project = cls.get_project_model(db, project_id, user_id)
        db.delete(project)
        db.commit()

    @classmethod
    def update_status(cls, db: Session, project_id: uuid.UUID, user_id: uuid.UUID, status_str: str) -> ProjectOut:
        project = cls.get_project_model(db, project_id, user_id)
        project.status = status_str
        project.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(project)
        return cls._format_project_out(project)

    @classmethod
    def save_sections(cls, db: Session, project_id: uuid.UUID, user_id: uuid.UUID, sections_in: List[ResearchSectionSchema]) -> List[ResearchSectionSchema]:
        project = cls.get_project_model(db, project_id, user_id)
        # Clear existing sections and replace
        db.query(ResearchSection).filter(ResearchSection.project_id == project_id).delete()
        
        for idx, sec in enumerate(sections_in):
            new_sec = ResearchSection(
                project_id=project_id,
                section_key=sec.id,
                title=sec.title,
                content=sec.content,
                included=sec.included,
                order_index=idx,
                page_number=sec.pageNumber
            )
            db.add(new_sec)
        
        project.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(project)
        return [
            ResearchSectionSchema(
                id=s.section_key if s.section_key else str(s.id),
                title=s.title,
                content=s.content,
                included=s.included,
                pageNumber=s.page_number
            )
            for s in project.sections
        ]

    @classmethod
    def save_poster(cls, db: Session, project_id: uuid.UUID, user_id: uuid.UUID, poster_in: dict) -> dict:
        project = cls.get_project_model(db, project_id, user_id)
        poster = project.poster
        now = datetime.now(timezone.utc)
        
        if not poster:
            poster = PosterDocument(
                project_id=project.id,
                title=poster_in.get("title", project.title),
                width=poster_in.get("width", 1000),
                height=poster_in.get("height", 700),
                orientation=poster_in.get("orientation", "landscape"),
                background=poster_in.get("background", "#ffffff"),
                template_id=poster_in.get("templateId", "research"),
                columns=poster_in.get("columns", 3),
                elements=poster_in.get("elements", [])
            )
            db.add(poster)
        else:
            if "title" in poster_in:
                poster.title = poster_in["title"]
                project.title = poster_in["title"]
            if "width" in poster_in:
                poster.width = poster_in["width"]
            if "height" in poster_in:
                poster.height = poster_in["height"]
            if "orientation" in poster_in:
                poster.orientation = poster_in["orientation"]
            if "background" in poster_in:
                poster.background = poster_in["background"]
            if "templateId" in poster_in:
                poster.template_id = poster_in["templateId"]
                project.template_id = poster_in["templateId"]
            if "columns" in poster_in:
                poster.columns = poster_in["columns"]
            if "elements" in poster_in:
                poster.elements = poster_in["elements"]
            poster.updated_at = now

        project.updated_at = now
        db.commit()
        db.refresh(poster)
        return {
            "id": str(poster.id),
            "projectId": str(poster.project_id),
            "updatedAt": poster.updated_at.isoformat()
        }

project_service = ProjectService()
