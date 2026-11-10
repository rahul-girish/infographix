"""Group F: User Content-Demand Schema and Layout Compatibility Bridge.

Addresses Requirement 4 of the Infographix redesign:
Users provide structured content (text paragraphs, figures, tables, authors),
NOT bounding boxes. This module:
  1. Defines `ContentBlockDemand` and `PosterRequest` separating `semantic_role`
     from `visual_type` (`scipost_category`).
  2. Provides a typography-aware dimension estimator that wraps text at candidate
     column widths to compute width-dependent height and area demands.
  3. Maps a `PosterRequest` into the shared structural feature vocabulary so it
     can query the layout retriever directly.
  4. Computes pairwise request-to-candidate compatibility features (canvas,
     inventory shortfall/surplus, area capacity, figure aspect matching, columns)
     for candidate ranking.
  5. Provides `simulate_request_from_heldout_poster` to evaluate retrieval on
     held-out dev/test posters by hiding their spatial coordinates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.features.pipeline_v2 import EPS, KNOWN_CATEGORY_SLUGS

# Mapping from user visual types to SciPostLayout category slugs
VISUAL_TYPE_TO_SCIPOST: dict[str, str] = {
    "title": "title",
    "author_info": "author_info",
    "authors": "author_info",
    "section": "section",
    "heading": "section",
    "text": "text",
    "paragraph": "text",
    "abstract": "text",
    "references": "text",
    "list": "list",
    "bullets": "list",
    "figure": "figure",
    "image": "figure",
    "logo": "figure",
    "table": "table",
    "caption": "caption",
}


def estimate_rendered_text_geometry(
    text: str | None = None,
    word_count: int | None = None,
    char_count: int | None = None,
    candidate_col_width_norm: float = 0.28,
    canvas_width_pt: float = 2592.0,   # e.g. 36 inches * 72 pt/in
    canvas_height_pt: float = 1728.0,  # e.g. 24 inches * 72 pt/in
    font_size_pt: float = 24.0,
    line_height_mult: float = 1.35,
    avg_char_width_em: float = 0.52,
) -> dict[str, float]:
    """Estimate normalized width, height, and area for a text block using font metrics.

    Because text height is inversely proportional to column width, this function
    computes wrapped line counts at the candidate column width.
    """
    if text is not None:
        words = text.split()
        w_cnt = len(words)
        c_cnt = len(text)
    else:
        w_cnt = word_count if word_count is not None else 60
        c_cnt = char_count if char_count is not None else w_cnt * 6

    box_width_pt = max(144.0, candidate_col_width_norm * canvas_width_pt)
    char_width_pt = font_size_pt * avg_char_width_em
    chars_per_line = max(15.0, (box_width_pt * 0.92) / char_width_pt)
    est_lines = max(1.0, np.ceil(max(1, c_cnt) / chars_per_line))

    line_height_pt = font_size_pt * line_height_mult
    est_height_pt = est_lines * line_height_pt + 16.0  # padding
    h_norm = float(np.clip(est_height_pt / canvas_height_pt, 0.015, 0.85))
    w_norm = float(np.clip(candidate_col_width_norm, 0.10, 0.95))

    return {
        "word_count": float(w_cnt),
        "char_count": float(c_cnt),
        "estimated_lines": float(est_lines),
        "min_width": float(w_norm * 0.75),
        "preferred_width": w_norm,
        "max_width": float(min(0.95, w_norm * 1.5)),
        "min_height": float(h_norm * 0.75),
        "preferred_height": h_norm,
        "max_height": float(min(0.90, h_norm * 1.4)),
        "preferred_area_norm": float(w_norm * h_norm),
    }


@dataclass
class ContentBlockDemand:
    """Individual user-supplied content block specification."""

    block_id: str
    semantic_role: str  # e.g., 'title', 'abstract', 'methods', 'results', 'conclusion', 'logo'
    visual_type: str    # e.g., 'title', 'author_info', 'section', 'text', 'list', 'figure', 'table', 'caption'
    priority: float = 1.0
    order_index: int = 0
    # Content-specific properties (optional at input; used to estimate geometry)
    word_count: int | float | None = None
    char_count: int | float | None = None
    estimated_lines: float | None = None
    preferred_area_norm: float | None = None
    aspect_ratio: float | None = None  # For figures/images (width / height)
    table_rows: int | None = None
    table_cols: int | None = None
    list_items: int | None = None
    # Normalized dimension demands (can be explicitly provided or auto-estimated)
    min_width: float = 0.05
    preferred_width: float = 0.25
    max_width: float = 0.95
    min_height: float = 0.02
    preferred_height: float = 0.12
    max_height: float = 0.85
    can_resize: bool = True
    can_split: bool = False
    can_crop: bool = False
    must_be_adjacent_to: str | None = None

    @property
    def scipost_category(self) -> str:
        return VISUAL_TYPE_TO_SCIPOST.get(self.visual_type.lower(), "text")

    @property
    def preferred_area(self) -> float:
        return float(self.preferred_width * self.preferred_height)


@dataclass
class PosterRequest:
    """Complete user content-demand specification at runtime."""

    request_id: str
    requested_aspect_ratio: float = 1.414  # Default landscape (e.g. A0 landscape ~1.41)
    requested_orientation: str = "landscape"  # 'portrait', 'landscape', 'square'
    preferred_column_count: int | None = None
    blocks: list[ContentBlockDemand] = field(default_factory=list)
    author_count: int = 3
    affiliation_count: int = 1
    style_preferences: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.requested_aspect_ratio < 0.95:
            self.requested_orientation = "portrait"
        elif self.requested_aspect_ratio > 1.05:
            self.requested_orientation = "landscape"
        else:
            self.requested_orientation = "square"

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert user request into structural demand features aligned with `posters.parquet`."""
        counts = {slug: 0 for slug in KNOWN_CATEGORY_SLUGS}
        area_sums = {slug: 0.0 for slug in KNOWN_CATEGORY_SLUGS}
        all_areas: list[float] = []
        all_widths: list[float] = []
        all_heights: list[float] = []

        for b in self.blocks:
            slug = b.scipost_category
            if slug in counts:
                counts[slug] += 1
                area_sums[slug] += b.preferred_area
            all_areas.append(b.preferred_area)
            all_widths.append(b.preferred_width)
            all_heights.append(b.preferred_height)

        known_total = sum(counts.values())
        denom = max(1, known_total)

        visual_cnt = counts["figure"] + counts["table"]
        textual_cnt = counts["text"] + counts["list"] + counts["caption"]
        heading_cnt = counts["title"] + counts["section"]

        visual_area = area_sums["figure"] + area_sums["table"]
        textual_area = area_sums["text"] + area_sums["list"] + area_sums["caption"]
        header_area = area_sums["title"] + area_sums["author_info"]
        total_area_demand = sum(all_areas)

        # Infer preferred column count from orientation and block inventory if not explicitly specified
        if self.preferred_column_count is not None:
            est_cols = int(self.preferred_column_count)
        else:
            if self.requested_orientation == "portrait":
                est_cols = 2 if known_total >= 12 else 2
            else:
                est_cols = 3 if known_total < 28 else 4

        arr_a = np.array(all_areas) if all_areas else np.array([0.02])
        arr_w = np.array(all_widths) if all_widths else np.array([0.25])
        arr_h = np.array(all_heights) if all_heights else np.array([0.10])

        summary: dict[str, Any] = {
            "request_id": self.request_id,
            "aspect_ratio": float(self.requested_aspect_ratio),
            "log_aspect_ratio": float(np.log(self.requested_aspect_ratio + EPS)),
            "orientation": self.requested_orientation,
            "is_landscape": float(self.requested_orientation == "landscape"),
            "is_portrait": float(self.requested_orientation == "portrait"),
            "total_element_count": int(known_total),
            "known_element_count": int(known_total),
            "visual_element_count": int(visual_cnt),
            "textual_element_count": int(textual_cnt),
            "heading_element_count": int(heading_cnt),
            "media_to_text_count_ratio": float(visual_cnt / (textual_cnt + 1.0)),
            "visual_area_sum": float(visual_area),
            "textual_area_sum": float(textual_area),
            "header_area_sum": float(header_area),
            "sum_box_area_fraction": float(total_area_demand),
            "visual_to_body_area_ratio": float(
                visual_area / (visual_area + textual_area + EPS)
            ),
            "estimated_column_count": int(est_cols),
            "area_norm_p25": float(np.percentile(arr_a, 25)),
            "area_norm_p50": float(np.percentile(arr_a, 50)),
            "area_norm_p75": float(np.percentile(arr_a, 75)),
            "w_norm_p50": float(np.percentile(arr_w, 50)),
            "h_norm_p50": float(np.percentile(arr_h, 50)),
        }

        for slug in KNOWN_CATEGORY_SLUGS:
            c = counts[slug]
            summary[f"count_{slug}"] = int(c)
            summary[f"log1p_count_{slug}"] = float(np.log1p(c))
            summary[f"has_{slug}"] = float(c > 0)
            summary[f"fraction_{slug}"] = float(c / denom)
            summary[f"area_sum_{slug}"] = float(area_sums[slug])

        return summary


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    return int(round(_safe_float(val, float(default))))


def compute_request_candidate_compatibility(
    request: PosterRequest | dict[str, Any],
    candidate_poster_row: pd.Series | dict[str, Any],
) -> dict[str, float]:
    """Compute explicit request-to-candidate compatibility features (Section 11.2 of Spec).

    Returns a dictionary of interpretable mismatch/compatibility signals and a
    composite compatibility score in [0, 1] (higher = better match).
    """
    req = (
        request.to_summary_dict()
        if isinstance(request, PosterRequest)
        else dict(request)
    )
    cand = dict(candidate_poster_row)

    # 1. Canvas compatibility
    abs_log_aspect_diff = abs(
        _safe_float(req.get("log_aspect_ratio")) - _safe_float(cand.get("log_aspect_ratio"))
    )
    orientation_match = float(req.get("orientation") == cand.get("orientation"))

    # 2. Category inventory compatibility
    shortfall_total = 0
    surplus_total = 0
    weighted_l1_count = 0.0

    # Category importance weights for layout suitability
    cat_weights = {
        "figure": 2.0,
        "table": 2.2,
        "text": 1.2,
        "section": 1.0,
        "list": 1.0,
        "title": 0.8,
        "author_info": 0.8,
        "caption": 0.7,
    }

    for slug in KNOWN_CATEGORY_SLUGS:
        r_c = _safe_int(req.get(f"count_{slug}"), 0)
        c_c = _safe_int(cand.get(f"count_{slug}"), 0)
        diff = r_c - c_c
        if diff > 0:
            shortfall_total += diff
        else:
            surplus_total += (-diff)
        weighted_l1_count += cat_weights.get(slug, 1.0) * abs(diff)

    req_known = _safe_float(req.get("known_element_count"), 1.0)
    cand_known = _safe_float(cand.get("known_element_count"), 0.0)
    rel_count_mismatch = abs(req_known - cand_known) / max(5.0, req_known)

    # 3. Capacity compatibility (Visual & Textual area demands vs candidate capacity)
    vis_area_diff = abs(
        _safe_float(req.get("visual_area_sum")) - _safe_float(cand.get("visual_area_sum"))
    )
    txt_area_diff = abs(
        _safe_float(req.get("textual_area_sum")) - _safe_float(cand.get("textual_area_sum"))
    )
    vis_ratio_diff = abs(
        _safe_float(req.get("visual_to_body_area_ratio"))
        - _safe_float(cand.get("visual_to_body_area_ratio"))
    )
    total_coverage_diff = abs(
        _safe_float(req.get("sum_box_area_fraction"), 0.5)
        - _safe_float(cand.get("sum_box_area_fraction"), 0.5)
    )

    # 4. Structural / Column compatibility
    col_diff = abs(
        _safe_float(req.get("estimated_column_count"), 2.0)
        - _safe_float(cand.get("estimated_column_count"), 2.0)
    )

    # Composite Penalty -> Normalized Compatibility Score in (0, 1]
    penalty = (
        3.5 * (1.0 - orientation_match)
        + 2.5 * abs_log_aspect_diff
        + 0.12 * weighted_l1_count
        + 1.8 * vis_area_diff
        + 1.2 * txt_area_diff
        + 1.5 * vis_ratio_diff
        + 0.45 * col_diff
    )
    compatibility_score = float(np.exp(-0.35 * penalty))

    return {
        "compatibility_score": compatibility_score,
        "orientation_match": orientation_match,
        "abs_log_aspect_diff": abs_log_aspect_diff,
        "shortfall_element_count": float(shortfall_total),
        "surplus_element_count": float(surplus_total),
        "weighted_l1_count_mismatch": weighted_l1_count,
        "rel_total_count_mismatch": rel_count_mismatch,
        "visual_area_capacity_diff": vis_area_diff,
        "textual_area_capacity_diff": txt_area_diff,
        "visual_to_body_ratio_diff": vis_ratio_diff,
        "total_coverage_diff": total_coverage_diff,
        "column_count_diff": col_diff,
    }


def simulate_request_from_heldout_poster(
    poster_row: pd.Series | dict[str, Any],
    poster_elements_df: pd.DataFrame,
    noise_scale: float = 0.05,
    random_state: int = 42,
) -> PosterRequest:
    """Construct a realistic `PosterRequest` from a held-out poster by hiding coordinates.

    Follows Section 6 of the user specification:
      1. Hides the original poster's exact bounding-box positions (x0, y0, cx, cy).
      2. Extracts its structural content requirements (orientation, aspect ratio,
         element types, and approximate block sizes with slight estimation noise
         simulating runtime text/asset sizing).
      3. Returns a `PosterRequest` object to query the retrieval system.
    """
    rng = np.random.default_rng(random_state + int(poster_row["poster_id"]) % 10000)
    blocks: list[ContentBlockDemand] = []

    valid_els = poster_elements_df[~poster_elements_df["is_unknown"]]
    for idx, (_, el) in enumerate(valid_els.iterrows()):
        slug = str(el["category_slug"])
        # Simulate slight runtime content-size estimation variance (no exact coordinates!)
        jitter_w = float(rng.normal(1.0, noise_scale))
        jitter_h = float(rng.normal(1.0, noise_scale))
        pref_w = float(np.clip(el["w_norm"] * jitter_w, 0.03, 0.95))
        pref_h = float(np.clip(el["h_norm"] * jitter_h, 0.015, 0.85))

        blocks.append(
            ContentBlockDemand(
                block_id=f"blk_{idx}",
                semantic_role=slug,
                visual_type=slug,
                priority=1.5 if slug in ("title", "figure", "table") else 1.0,
                order_index=idx,
                aspect_ratio=float(pref_w / (pref_h + EPS)),
                preferred_width=pref_w,
                preferred_height=pref_h,
            )
        )

    return PosterRequest(
        request_id=f"sim_req_{int(poster_row['poster_id'])}",
        requested_aspect_ratio=float(poster_row["aspect_ratio"]),
        requested_orientation=str(poster_row["orientation"]),
        preferred_column_count=int(poster_row["estimated_column_count"]),
        blocks=blocks,
    )
