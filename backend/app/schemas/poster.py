from typing import Any, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class ContentBlockInput(BaseModel):
    """Single user-supplied poster content block (text section, figure, table, or list)."""

    id: str
    type: Literal["section", "text", "figure", "table", "list", "caption"] = "section"
    heading: Optional[str] = None
    text: Optional[str] = None
    imageDataUrl: Optional[str] = None
    aspectRatio: Optional[float] = Field(default=None, ge=0.2, le=5.0)
    tableRows: Optional[int] = Field(default=4, ge=1, le=20)
    tableCols: Optional[int] = Field(default=3, ge=1, le=12)
    caption: Optional[str] = None
    priority: float = Field(default=1.0, ge=0.1, le=2.0)


class PosterGenerateRequest(BaseModel):
    """Input payload for generating a poster layout via the AI model pipeline."""

    title: str = Field(..., min_length=1, max_length=300)
    authors: str = Field(default="", max_length=500)
    affiliations: str = Field(default="", max_length=500)
    orientation: Literal["landscape", "portrait"] = "landscape"
    preferredColumns: Optional[int] = Field(default=None, ge=1, le=4)
    accentColor: str = "#3458e8"
    background: str = "#ffffff"
    engine: Literal["adaptation", "neural"] = "adaptation"
    blocks: List[ContentBlockInput] = Field(default_factory=list)


class PosterElementSchema(BaseModel):
    """Editable Konva canvas element with optional MLOps provenance fields."""

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
    src: Optional[str] = None
    points: Optional[List[float]] = None
    # MLOps provenance
    sourceBlockId: Optional[str] = None
    semanticRole: Optional[str] = None
    bboxNorm: Optional[List[float]] = None


class PosterDocumentSchema(BaseModel):
    """Canonical editable poster document shared between backend and frontend editor."""

    model_config = ConfigDict(extra="allow")

    id: str
    title: str
    width: int = 1000
    height: int = 700
    orientation: Literal["landscape", "portrait"] = "landscape"
    background: str = "#ffffff"
    accentColor: str = "#3458e8"
    columns: int = 3
    elements: List[dict[str, Any]] = Field(default_factory=list)
    updatedAt: str


class GenerationRunOut(BaseModel):
    """Response returned after generating or fetching a poster run."""

    id: str
    createdAt: str
    updatedAt: str
    title: str
    authors: str
    orientation: str
    engine: str
    modelName: str
    modelVersion: str
    retrievedTemplateId: Optional[str] = None
    candidateTemplates: List[dict[str, Any]] = Field(default_factory=list)
    generationMetrics: dict[str, Any] = Field(default_factory=dict)
    poster: PosterDocumentSchema
    wasEdited: bool = False
    exportedFormats: List[str] = Field(default_factory=list)
    feedbackRating: Optional[int] = None
    feedbackNotes: Optional[str] = None


class GenerationRunSummaryOut(BaseModel):
    """Lightweight run metadata for listing/monitoring collected runs."""

    id: str
    createdAt: str
    title: str
    orientation: str
    engine: str
    modelName: str
    modelVersion: str
    retrievedTemplateId: Optional[str] = None
    wasEdited: bool
    exportedFormats: List[str] = Field(default_factory=list)
    feedbackRating: Optional[int] = None
    generationMetrics: dict[str, Any] = Field(default_factory=dict)


class RunUpdateRequest(BaseModel):
    """Updates an existing run with user edits, export telemetry, or rating."""

    editedLayout: Optional[dict[str, Any]] = None
    exportedFormat: Optional[Literal["png", "pdf", "svg"]] = None
    feedbackRating: Optional[int] = Field(default=None, ge=1, le=5)
    feedbackNotes: Optional[str] = Field(default=None, max_length=1000)


class PdfExportRequest(BaseModel):
    """Payload to export an edited poster canvas to PDF."""

    title: str = "infographix-poster"
    width: int = Field(default=1000, ge=100, le=10000)
    height: int = Field(default=700, ge=100, le=10000)
    imageDataUrl: Optional[str] = None
    poster: Optional[PosterDocumentSchema] = None
    runId: Optional[str] = None
