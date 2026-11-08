"""Structured, fully editable Poster Document Schema for Infographix (Stage 3).

Provides the canonical, editable JSON representation of an adapted poster:
  - `PosterCanvas`: width, height, aspect ratio, units (pixels/points/inches)
  - `PosterElement`: stable ID, semantic type, normalized and pixel bounding boxes,
    actual content (text, image URL/path, table structure), styling, group/section,
    source content ID, and spatial relationships.
  - `PosterDocument`: complete container, convertible to/from COCO JSON format,
    preserving full provenance from user input through template adaptation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.schemas.coco import (
    CATEGORY_NAME_TO_ID,
    COCOAnnotation,
    COCOCategory,
    COCODataset,
    COCOImage,
    DEFAULT_POSTER_CATEGORIES,
)


@dataclass
class PosterCanvas:
    """Canvas physical and coordinate dimensions."""

    width: int = 1920
    height: int = 1080
    unit: str = "px"  # px, pt, mm, in
    dpi: int = 150
    background_color: str = "#ffffff"
    margin_px: int = 36

    @property
    def aspect_ratio(self) -> float:
        return float(self.width / max(1, self.height))

    @property
    def orientation(self) -> str:
        if self.aspect_ratio < 0.95:
            return "portrait"
        elif self.aspect_ratio > 1.05:
            return "landscape"
        return "square"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ElementStyle:
    """Styling properties for an individual rendered element."""

    font_family: str = "Inter, system-ui, sans-serif"
    font_size_pt: float = 16.0
    font_weight: str = "normal"  # normal, bold, 600, etc.
    line_height: float = 1.35
    text_color: str = "#1e293b"
    background_color: str = "transparent"
    border_color: str = "transparent"
    border_width_px: int = 0
    border_radius_px: int = 4
    padding_px: int = 8
    text_align: str = "left"  # left, center, right, justify
    opacity: float = 1.0
    shadow: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PosterElement:
    """Individual editable layout element on the poster canvas."""

    id: str
    type: str  # title, author_info, section, text, figure, table, caption, list, logo
    bbox: list[float]  # [x, y, width, height] in canvas pixels/points
    bbox_norm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # [x0, y0, w, h] normalized in [0, 1]
    content: str | dict[str, Any] | list[Any] | None = None
    style: ElementStyle = field(default_factory=ElementStyle)
    group_id: str | None = None  # header, col_0_sec_1, etc.
    column_idx: int | None = None  # 0, 1, 2... or None for header
    source_block_id: str | None = None  # Provenance ID from user's PosterRequest
    associated_element_ids: list[str] = field(default_factory=list)  # e.g., caption -> figure ID
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 coordinates [x, y, w, h], got {self.bbox}")
        if isinstance(self.style, dict):
            self.style = ElementStyle(**self.style)

    @property
    def area(self) -> float:
        return float(self.bbox[2] * self.bbox[3])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "bbox": [round(float(v), 2) for v in self.bbox],
            "bbox_norm": [round(float(v), 4) for v in self.bbox_norm],
            "content": self.content,
            "style": self.style.to_dict() if hasattr(self.style, "to_dict") else self.style,
            "group_id": self.group_id,
            "column_idx": self.column_idx,
            "source_block_id": self.source_block_id,
            "associated_element_ids": self.associated_element_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PosterElement:
        c_data = dict(data)
        if "style" in c_data and isinstance(c_data["style"], dict):
            c_data["style"] = ElementStyle(**c_data["style"])
        return cls(**c_data)


@dataclass
class PosterDocument:
    """Top-level container for an adapted, fully editable scientific poster."""

    poster_id: str
    source_template_id: int | str
    canvas: PosterCanvas
    elements: list[PosterElement] = field(default_factory=list)
    title: str = ""
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "poster_id": self.poster_id,
            "source_template_id": self.source_template_id,
            "canvas": self.canvas.to_dict(),
            "elements": [el.to_dict() for el in self.elements],
            "title": self.title,
            "authors": self.authors,
            "affiliations": self.affiliations,
            "metadata": self.metadata,
        }

    def save_json(self, output_path: str | Path) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PosterDocument:
        canvas = PosterCanvas(**data.get("canvas", {}))
        elements = [PosterElement.from_dict(el) for el in data.get("elements", [])]
        return cls(
            poster_id=data.get("poster_id", "generated_poster"),
            source_template_id=data.get("source_template_id", 0),
            canvas=canvas,
            elements=elements,
            title=data.get("title", ""),
            authors=data.get("authors", []),
            affiliations=data.get("affiliations", []),
            metadata=data.get("metadata", {}),
        )

    def to_coco(self) -> COCODataset:
        """Export this PosterDocument to standardized COCO JSON format."""
        coco_img = COCOImage(
            id=self.poster_id,
            width=self.canvas.width,
            height=self.canvas.height,
            file_name=f"{self.poster_id}.png",
            extra_metadata={
                "source_template_id": self.source_template_id,
                "title": self.title,
                "authors": self.authors,
            },
        )
        coco_anns: list[COCOAnnotation] = []
        for idx, el in enumerate(self.elements):
            cat_id = CATEGORY_NAME_TO_ID.get(el.type.lower(), 10)  # 10=unknown
            txt = el.content if isinstance(el.content, str) else None
            ann = COCOAnnotation(
                id=idx + 1,
                image_id=self.poster_id,
                category_id=cat_id,
                bbox=el.bbox,
                area=float(el.bbox[2] * el.bbox[3]),
                text_content=txt,
                custom_styling=el.style.to_dict() if hasattr(el.style, "to_dict") else {},
            )
            coco_anns.append(ann)

        return COCODataset(images=[coco_img], annotations=coco_anns)
