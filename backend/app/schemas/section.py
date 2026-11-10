from typing import Optional
from pydantic import BaseModel, ConfigDict

class ResearchSectionSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: str
    title: str
    content: str
    included: bool = True
    pageNumber: Optional[int] = None

class ResearchSectionsUpdate(BaseModel):
    sections: list[ResearchSectionSchema]
