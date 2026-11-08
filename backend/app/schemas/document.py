from typing import Optional, List
from pydantic import BaseModel, ConfigDict
from backend.app.schemas.section import ResearchSectionSchema

class DocumentOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    projectId: str
    fileName: str
    fileSize: int
    contentType: str
    pageCount: Optional[int] = None
    status: str
    uploadedAt: str

class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document: DocumentOut
    name: str
    sections: List[ResearchSectionSchema]
