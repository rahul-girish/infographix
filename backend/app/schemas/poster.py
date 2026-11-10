from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field

class PosterElementSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str  # "text" | "image" | "rect" | "circle" | "line" | "arrow"
    name: str
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    layer: int = 0
    visible: bool = True
    locked: bool = False
    groupId: Optional[str] = None
    text: Optional[str] = None
    fontFamily: Optional[str] = None
    fontSize: Optional[float] = None
    fontWeight: Optional[str] = None
    fontStyle: Optional[str] = None
    underline: Optional[bool] = None
    align: Optional[str] = None
    lineHeight: Optional[float] = None
    fill: Optional[str] = None
    stroke: Optional[str] = None
    strokeWidth: Optional[float] = None
    opacity: Optional[float] = None
    imageFit: Optional[str] = None
    assetId: Optional[str] = None
    src: Optional[str] = None
    points: Optional[List[float]] = None

class PosterDocumentSchema(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    projectId: str = Field(..., alias="project_id")
    title: str
    width: int = 1000
    height: int = 700
    orientation: str = "landscape"
    background: str = "#ffffff"
    templateId: str = Field("research", alias="template_id")
    columns: int = 3
    elements: List[dict[str, Any]] = Field(default_factory=list)
    updatedAt: str = Field(..., alias="updated_at")

