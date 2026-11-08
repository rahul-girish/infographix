"""Feature engineering v2 package for Infographix."""

from src.features.pipeline_v2 import (
    KNOWN_CATEGORY_SLUGS,
    SCIPOST_CATEGORIES,
    SPLITS,
    build_posters_table,
    build_relations_and_spatial_summaries,
    run_feature_pipeline_v2,
    validate_and_extract_elements,
)
from src.features.request_schema import (
    ContentBlockDemand,
    PosterRequest,
    compute_request_candidate_compatibility,
    estimate_rendered_text_geometry,
    simulate_request_from_heldout_poster,
)
from src.features.visual_features import (
    extract_single_poster_visual_features,
    extract_visual_features_for_sample,
)

__all__ = [
    "KNOWN_CATEGORY_SLUGS",
    "SCIPOST_CATEGORIES",
    "SPLITS",
    "validate_and_extract_elements",
    "build_relations_and_spatial_summaries",
    "build_posters_table",
    "run_feature_pipeline_v2",
    "ContentBlockDemand",
    "PosterRequest",
    "estimate_rendered_text_geometry",
    "compute_request_candidate_compatibility",
    "simulate_request_from_heldout_poster",
    "extract_single_poster_visual_features",
    "extract_visual_features_for_sample",
]
