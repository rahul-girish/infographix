from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from backend.app.schemas.section import ResearchSectionSchema
from backend.app.schemas.poster import PosterDocumentSchema

class ProjectCreate(BaseModel):
    title: Optional[str] = "Untitled research poster"
    templateId: Optional[str] = Field("research", alias="template_id")

class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    templateId: Optional[str] = Field(None, alias="template_id")
    status: Optional[str] = None

class ProjectRename(BaseModel):
    title: str

class ProjectStatusUpdate(BaseModel):
    status: str

class ProjectOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str
    status: str
    updatedAt: str = Field(..., alias="updated_at")
    templateId: str = Field(..., alias="template_id")
    documentName: Optional[str] = Field(None, alias="document_name")
    sections: List[ResearchSectionSchema] = Field(default_factory=list)
    poster: Optional[PosterDocumentSchema] = None

