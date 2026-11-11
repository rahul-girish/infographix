from typing import Any, List, Literal, Optional, Union
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class ContentBlockInput(BaseModel):
    """Single user-supplied poster content block."""

    id: str = Field(..., min_length=1, max_length=128)
    type: Literal["section", "text", "figure", "table", "list", "caption"] = "section"
    heading: Optional[str] = None
    text: Optional[str] = None
    imageDataUrl: Optional[str] = None
    aspectRatio: Optional[float] = Field(default=None, ge=0.2, le=5.0)
    tableRows: Optional[int] = Field(default=4, ge=1, le=20)
    tableCols: Optional[int] = Field(default=3, ge=1, le=12)
    caption: Optional[str] = None
    priority: float = Field(default=1.0, ge=0.1, le=2.0)


class PosterRequestPayload(BaseModel):
    """Structured poster input payload compatible with the application backend and frontend."""

    model_config = ConfigDict(populate_by_name=True)

    title: str = Field(..., min_length=1, max_length=300)
    authors: Union[str, List[str]] = ""
    affiliations: Union[str, List[str]] = ""
    orientation: Literal["landscape", "portrait"] = "landscape"
    preferredColumns: Optional[int] = Field(
        default=None,
        ge=1,
        le=4,
        validation_alias=AliasChoices("preferredColumns", "columns"),
    )
    accentColor: str = "#3458e8"
    background: str = "#ffffff"
    blocks: List[ContentBlockInput] = Field(
        default_factory=list,
        validation_alias=AliasChoices("blocks", "content_blocks"),
    )

    @field_validator("authors", "affiliations", mode="before")
    @classmethod
    def _normalize_str_or_list(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        return str(value)


class GenerateLayoutRequest(BaseModel):
    """Primary inference request for POST /v1/layouts/generate."""

    request: PosterRequestPayload
    engine: Optional[Literal["neural", "adaptation"]] = None


class PosterElementOut(BaseModel):
    """Konva-compatible poster element with normalized coordinates and provenance."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str
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
    sourceBlockId: Optional[str] = None
    semanticRole: Optional[str] = None
    bboxNorm: Optional[List[float]] = None


class PosterDocumentOut(BaseModel):
    """Structured poster layout matching the frontend editor's Konva PosterDocument."""

    model_config = ConfigDict(extra="allow")

    id: str
    title: str
    width: int
    height: int
    orientation: Literal["landscape", "portrait"]
    background: str = "#ffffff"
    accentColor: str = "#3458e8"
    columns: int = 3
    elements: List[PosterElementOut] = Field(default_factory=list)
    updatedAt: str


class GenerateLayoutResponse(BaseModel):
    """Inference output and full ClearML/template provenance."""

    engine: Literal["neural", "adaptation"]
    model_name: str
    model_version: str
    model_id: Optional[str] = None
    retriever_version: str
    retriever_model_id: str
    retrieved_template_id: Optional[str] = None
    candidate_templates: List[dict[str, Any]] = Field(default_factory=list)
    generation_metrics: dict[str, Any] = Field(default_factory=dict)
    request_features: dict[str, Any] = Field(default_factory=dict)
    layout: PosterDocumentOut


class LoadedModelInfo(BaseModel):
    """Descriptor for an actually loaded model artifact in memory."""

    role: Literal["retriever", "neural_generator"]
    model_name: str
    model_family: str
    clearml_model_id: str
    semantic_version: str
    architecture: str
    checkpoint_sha256: str
    lifecycle_status: str = "unknown"
    dataset_identifier: str = "unknown"
    supported_engines: List[str] = Field(default_factory=list)


class ModelsListResponse(BaseModel):
    """Response for GET /v1/models."""

    status: str
    default_engine: str
    supported_engines: List[str]
    models: List[LoadedModelInfo]
    data_dependency: dict[str, Any]


class LiveHealthResponse(BaseModel):
    status: Literal["alive"]
    service: str
    version: str


class ReadyHealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_initialized: bool
    retriever_initialized: bool
    data_dependency_ready: bool
    selected_models: dict[str, dict[str, Optional[str]]]
    supported_engines: List[str]
    error: Optional[str] = None
