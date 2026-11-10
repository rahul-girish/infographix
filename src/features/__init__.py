"""Modular Feature Engineering Pipeline (v2) package for Infographix."""

from src.features.config import (
    BODY_STRUCT_CAT_IDS,
    DEFAULT_PIPELINE_CONFIG_PATH,
    DEFAULT_SPLIT_RATIOS,
    EPS,
    HEADER_CAT_IDS,
    KNOWN_CATEGORY_SLUGS,
    LANDSCAPE_MIN_ASPECT,
    PORTRAIT_MAX_ASPECT,
    SCIPOST_CATEGORIES,
    SPLITS,
    TEXTUAL_CAT_IDS,
    UNKNOWN_CAT_ID,
    VISUAL_CAT_IDS,
    PipelineConfig,
    load_pipeline_config,
)
from src.features.exporters import (
    export_legacy_v1_compatible_parquets,
    export_v2_tables_and_metadata,
)
from src.features.helpers import (
    assign_pooled_splits,
    category_slug,
    classify_aspect_orientation,
    compute_orientation,
)
from src.features.pipeline import (
    build_pipeline_report,
    run_feature_pipeline_v2,
    run_pipeline_from_config,
)
from src.features.posters import build_posters_table
from src.features.relations import build_relations_and_spatial_summaries
from src.features.schema import NON_FEATURE_POSTER_COLS, generate_feature_schema_metadata
from src.features.validation import validate_and_extract_elements

__all__ = [
    "SPLITS",
    "DEFAULT_SPLIT_RATIOS",
    "SCIPOST_CATEGORIES",
    "KNOWN_CATEGORY_SLUGS",
    "HEADER_CAT_IDS",
    "VISUAL_CAT_IDS",
    "TEXTUAL_CAT_IDS",
    "BODY_STRUCT_CAT_IDS",
    "UNKNOWN_CAT_ID",
    "EPS",
    "PORTRAIT_MAX_ASPECT",
    "LANDSCAPE_MIN_ASPECT",
    "DEFAULT_PIPELINE_CONFIG_PATH",
    "PipelineConfig",
    "load_pipeline_config",
    "category_slug",
    "classify_aspect_orientation",
    "compute_orientation",
    "assign_pooled_splits",
    "validate_and_extract_elements",
    "build_relations_and_spatial_summaries",
    "build_posters_table",
    "NON_FEATURE_POSTER_COLS",
    "generate_feature_schema_metadata",
    "export_v2_tables_and_metadata",
    "export_legacy_v1_compatible_parquets",
    "build_pipeline_report",
    "run_feature_pipeline_v2",
    "run_pipeline_from_config",
]
