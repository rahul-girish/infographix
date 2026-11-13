"""Constants, SciPostLayout v2 category mappings, and configuration defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

SPLITS: tuple[str, ...] = ("train", "dev", "test")
DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)

# Canonical SciPostLayout v2 category mapping (defined by the dataset)
SCIPOST_CATEGORIES: dict[int, str] = {
    487601213: "Title",
    674852286: "Author Info",
    481284670: "Section",
    2871247081: "Text",
    2837082304: "List",
    3731383018: "Figure",
    3405481596: "Table",
    788514875: "Caption",
    3591166397: "Unknown",
}

KNOWN_CATEGORY_SLUGS: tuple[str, ...] = (
    "title",
    "author_info",
    "section",
    "text",
    "list",
    "figure",
    "table",
    "caption",
)

HEADER_CAT_IDS: set[int] = {487601213, 674852286}  # Title, Author Info
VISUAL_CAT_IDS: set[int] = {3731383018, 3405481596}  # Figure, Table
TEXTUAL_CAT_IDS: set[int] = {2871247081, 2837082304, 788514875}  # Text, List, Caption
BODY_STRUCT_CAT_IDS: set[int] = {
    481284670,
    2871247081,
    2837082304,
    3731383018,
    3405481596,
}
UNKNOWN_CAT_ID: int = 3591166397

EPS: float = 1e-6
PORTRAIT_MAX_ASPECT: float = 0.95
LANDSCAPE_MIN_ASPECT: float = 1.05

DEFAULT_PIPELINE_CONFIG_PATH: Path = (
    Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration container for the v2 feature engineering pipeline."""

    pipeline_version: str = "2.0.0"
    raw_dir: Path = Path("data/interim/scipostlayout/poster/png")
    out_dir: Path = Path("data/processed/features/v2")
    splits: tuple[str, ...] = SPLITS
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS
    random_state: int = 42
    stratify_by_orientation: bool = True
    boundary_tolerance_px: float = 1.0
    align_tol: float = 0.015
    knn_k: int = 3
    grid_res: int = 32
    export_legacy: bool = True
    overwrite_legacy: bool = True


def load_pipeline_config(config_path: str | Path | None = None) -> PipelineConfig:
    """Load pipeline configuration from a YAML file, falling back to defaults."""
    target_path = Path(config_path) if config_path is not None else DEFAULT_PIPELINE_CONFIG_PATH
    if not target_path.exists():
        if config_path is not None:
            raise FileNotFoundError(f"Pipeline config file not found: {target_path}")
        return PipelineConfig()

    with open(target_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    paths = raw_cfg.get("paths", {})
    splits_cfg = raw_cfg.get("splits", {})
    val_cfg = raw_cfg.get("validation", {})
    rel_cfg = raw_cfg.get("relations", {})
    posters_cfg = raw_cfg.get("posters", {})
    export_cfg = raw_cfg.get("export", {})

    ratios_list = splits_cfg.get("ratios", list(DEFAULT_SPLIT_RATIOS))
    split_ratios = (float(ratios_list[0]), float(ratios_list[1]), float(ratios_list[2]))

    return PipelineConfig(
        pipeline_version=str(raw_cfg.get("pipeline_version", "2.0.0")),
        raw_dir=Path(paths.get("raw_dir", "data/interim/scipostlayout/poster/png")),
        out_dir=Path(paths.get("out_dir", "data/processed/features/v2")),
        splits=tuple(splits_cfg.get("names", SPLITS)),
        split_ratios=split_ratios,
        random_state=int(splits_cfg.get("random_state", 42)),
        stratify_by_orientation=bool(splits_cfg.get("stratify_by_orientation", True)),
        boundary_tolerance_px=float(val_cfg.get("boundary_tolerance_px", 1.0)),
        align_tol=float(rel_cfg.get("align_tol", 0.015)),
        knn_k=int(rel_cfg.get("knn_k", 3)),
        grid_res=int(posters_cfg.get("grid_res", 32)),
        export_legacy=bool(export_cfg.get("export_legacy", True)),
        overwrite_legacy=bool(export_cfg.get("overwrite_legacy", True)),
    )
