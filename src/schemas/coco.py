"""COCO schema and conversion utilities for Infographix.

This module provides standard COCO data models, validation functions, and
serialization utilities connecting document extraction, poster templates,
and the visual canvas editor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_POSTER_CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "title", "supercategory": "header"},
    {"id": 2, "name": "author_info", "supercategory": "header"},
    {"id": 3, "name": "abstract", "supercategory": "content"},
    {"id": 4, "name": "section", "supercategory": "structure"},
    {"id": 5, "name": "text", "supercategory": "content"},
    {"id": 6, "name": "figure", "supercategory": "media"},
    {"id": 7, "name": "caption", "supercategory": "media"},
    {"id": 8, "name": "table", "supercategory": "media"},
    {"id": 9, "name": "list", "supercategory": "content"},
    {"id": 10, "name": "unknown", "supercategory": "other"},
]

CATEGORY_NAME_TO_ID: dict[str, int] = {
    cat["name"]: cat["id"] for cat in DEFAULT_POSTER_CATEGORIES
}
CATEGORY_ID_TO_NAME: dict[int, str] = {
    cat["id"]: cat["name"] for cat in DEFAULT_POSTER_CATEGORIES
}


@dataclass
class COCOCategory:
    """COCO category definition."""
    id: int
    name: str
    supercategory: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class COCOImage:
    """COCO image/canvas metadata."""
    id: int | str
    width: int
    height: int
    file_name: str
    license: int = 1
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["extra_metadata"]:
            data.pop("extra_metadata")
        return data


@dataclass
class COCOAnnotation:
    """COCO bounding box annotation for a poster element."""
    id: int
    image_id: int | str
    category_id: int
    bbox: list[float]  # [x, y, width, height]
    area: float
    iscrowd: int = 0
    score: float = 1.0
    text_content: str | None = None
    custom_styling: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must contain 4 values [x, y, w, h], got {self.bbox}")
        if self.area <= 0:
            self.area = float(self.bbox[2] * self.bbox[3])

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": [float(v) for v in self.bbox],
            "area": float(self.area),
            "iscrowd": self.iscrowd,
            "score": float(self.score),
        }
        if self.text_content is not None:
            res["text_content"] = self.text_content
        if self.custom_styling:
            res["custom_styling"] = self.custom_styling
        return res


@dataclass
class COCODataset:
    """Standard COCO container for images, annotations, and categories."""
    images: list[COCOImage] = field(default_factory=list)
    annotations: list[COCOAnnotation] = field(default_factory=list)
    categories: list[COCOCategory] = field(
        default_factory=lambda: [COCOCategory(**c) for c in DEFAULT_POSTER_CATEGORIES]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": [img.to_dict() for img in self.images],
            "annotations": [ann.to_dict() for ann in self.annotations],
            "categories": [cat.to_dict() for cat in self.categories],
        }

    def save_json(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> COCODataset:
        images = [
            COCOImage(
                id=img["id"],
                width=img["width"],
                height=img["height"],
                file_name=img.get("file_name", f"image_{img['id']}.png"),
                extra_metadata=img.get("extra_metadata", {}),
            )
            for img in data.get("images", [])
        ]
        categories = [
            COCOCategory(
                id=c["id"],
                name=c["name"],
                supercategory=c.get("supercategory", "general"),
            )
            for c in data.get("categories", DEFAULT_POSTER_CATEGORIES)
        ]
        annotations = [
            COCOAnnotation(
                id=ann["id"],
                image_id=ann["image_id"],
                category_id=ann["category_id"],
                bbox=ann["bbox"],
                area=ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                iscrowd=ann.get("iscrowd", 0),
                score=ann.get("score", 1.0),
                text_content=ann.get("text_content"),
                custom_styling=ann.get("custom_styling", {}),
            )
            for ann in data.get("annotations", [])
        ]
        return cls(images=images, annotations=annotations, categories=categories)

    @classmethod
    def load_json(cls, input_path: str | Path) -> COCODataset:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def validate_coco_structure(data: dict[str, Any]) -> tuple[bool, str]:
    """Validate a dictionary against the required COCO layout specification."""
    required_keys = {"images", "annotations", "categories"}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - set(data.keys())
        return False, f"Missing required top-level COCO keys: {missing}"

    if not isinstance(data["images"], list):
        return False, "'images' must be a list"
    if not isinstance(data["annotations"], list):
        return False, "'annotations' must be a list"
    if not isinstance(data["categories"], list):
        return False, "'categories' must be a list"

    image_ids = {img.get("id") for img in data["images"]}
    cat_ids = {c.get("id") for c in data["categories"]}

    for idx, ann in enumerate(data["annotations"]):
        if ann.get("image_id") not in image_ids:
            return False, f"Annotation at index {idx} references unknown image_id {ann.get('image_id')}"
        if ann.get("category_id") not in cat_ids:
            return False, f"Annotation at index {idx} references unknown category_id {ann.get('category_id')}"
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False, f"Annotation at index {idx} has invalid bbox format {bbox}"
        if bbox[2] <= 0 or bbox[3] <= 0:
            return False, f"Annotation at index {idx} has non-positive width/height in bbox: {bbox}"

    return True, "Valid COCO structure"


def convert_extracted_document_to_coco(
    doc_id: str | int,
    extracted_sections: list[dict[str, Any]],
    poster_width: int = 1920,
    poster_height: int = 1080,
    file_name: str = "extracted_poster.png",
) -> COCODataset:
    """Convert extracted document components into a structured COCODataset.

    Each extracted section should be a dict with:
        - category_name: str (e.g. 'title', 'abstract', 'figure', 'text')
        - bbox: [x, y, w, h] (in poster pixel coordinates)
        - text_content: optional str
        - custom_styling: optional dict
    """
    image = COCOImage(
        id=doc_id,
        width=poster_width,
        height=poster_height,
        file_name=file_name,
    )
    annotations: list[COCOAnnotation] = []

    for idx, sec in enumerate(extracted_sections, start=1):
        cat_name = sec.get("category_name", "unknown").lower()
        cat_id = CATEGORY_NAME_TO_ID.get(cat_name, CATEGORY_NAME_TO_ID["unknown"])
        bbox = sec.get("bbox", [0.0, 0.0, 100.0, 100.0])
        area = float(bbox[2] * bbox[3])

        ann = COCOAnnotation(
            id=idx,
            image_id=doc_id,
            category_id=cat_id,
            bbox=bbox,
            area=area,
            text_content=sec.get("text_content"),
            custom_styling=sec.get("custom_styling", {}),
        )
        annotations.append(ann)

    return COCODataset(images=[image], annotations=annotations)


def convert_canvas_to_coco(
    project_id: str | int,
    canvas_elements: list[dict[str, Any]],
    canvas_width: int,
    canvas_height: int,
) -> dict[str, Any]:
    """Convert frontend canvas elements into a COCO-compliant dictionary."""
    dataset = convert_extracted_document_to_coco(
        doc_id=project_id,
        extracted_sections=canvas_elements,
        poster_width=canvas_width,
        poster_height=canvas_height,
        file_name=f"{project_id}_canvas.png",
    )
    return dataset.to_dict()
