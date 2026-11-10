"""Multi-table Feature Engineering Pipeline (v2.0.0) for Infographix.

Implements the canonical multi-table representation specified in
`infographix_feature_engineering_spec.md`:
  - Group A: Poster and canvas features (`posters.parquet`)
  - Group B: Element-level features (`elements.parquet`)
  - Group C: Pairwise spatial relationships (`relations.parquet`)
  - Group D: Global composition, occupancy grids, profiles, and column structure (`posters.parquet`)
  - Data quality log (`rejected_annotations.parquet`, `feature_report.json`, `feature_schema.json`)

Key engineering guarantees:
  1. Zero valid posters are dropped (fixes the v1 bug where single-element posters were dropped).
  2. Strict train-only fitting for all reference statistics (category medians, scalers).
  3. Explicit `has_<category>` flags and distinction between absent categories and zero coordinates.
  4. Accurate rasterized union coverage (`union_box_coverage`) vs `sum_box_area_fraction`.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

SPLITS = ("train", "dev", "test")
DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)

# Canonical SciPostLayout v2 category mapping
# This is defined by the dataset itself
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

KNOWN_CATEGORY_SLUGS = (
    "title",
    "author_info",
    "section",
    "text",
    "list",
    "figure",
    "table",
    "caption",
)

HEADER_CAT_IDS = {487601213, 674852286}  # Title, Author Info
VISUAL_CAT_IDS = {3731383018, 3405481596}  # Figure, Table
TEXTUAL_CAT_IDS = {2871247081, 2837082304, 788514875}  # Text, List, Caption
BODY_STRUCT_CAT_IDS = {481284670, 2871247081, 2837082304, 3731383018, 3405481596}
UNKNOWN_CAT_ID = 3591166397

# epsilon, i assume for preventing divide by zero
EPS = 1e-6

# all this for a snake case name.
def category_slug(name: str) -> str:
    """Create stable snake_case identifier from a category name."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return slug or "unnamed"


def _compute_orientation(widths: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """Classify poster aspect ratios into portrait, landscape, or square."""
    aspect = widths / heights
    return np.select(
        [aspect < 0.95, aspect > 1.05],
        ["portrait", "landscape"],
        default="square",
    )


def assign_pooled_splits(
    pooled_images_df: pd.DataFrame,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = 42,
    stratify_by_orientation: bool = True,
) -> pd.DataFrame:
    """Deterministically assign pooled posters into train/dev/test splits."""
    if len(split_ratios) != 3 or any(r <= 0 for r in split_ratios):
        raise ValueError(f"split_ratios must contain 3 positive floats, got {split_ratios}")
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError(
            f"split_ratios must sum to 1.0, got {split_ratios} (sum={sum(split_ratios)})"
        )

    df = pooled_images_df.sort_values("id", kind="mergesort").reset_index(drop=True).copy()
    r_train, r_dev, _ = split_ratios
    rng = np.random.default_rng(random_state)
    assigned_splits = np.empty(len(df), dtype=object)

    def _assign_group(indices: np.ndarray) -> None:
        perm = indices[rng.permutation(len(indices))]
        n = len(perm)
        n_train = int(round(n * r_train))
        n_dev = int(round(n * r_dev))
        if n_train + n_dev > n:
            n_dev = max(0, n - n_train)
        assigned_splits[perm[:n_train]] = "train"
        assigned_splits[perm[n_train : n_train + n_dev]] = "dev"
        assigned_splits[perm[n_train + n_dev :]] = "test"

    if stratify_by_orientation:
        orientations = _compute_orientation(
            df["width"].to_numpy(dtype=float),
            df["height"].to_numpy(dtype=float),
        )
        for orient in sorted(np.unique(orientations)):
            _assign_group(np.where(orientations == orient)[0])
    else:
        _assign_group(np.arange(len(df)))

    df["split"] = assigned_splits
    return df


# entry of the data processing.
def validate_and_extract_elements(
    raw_dir: Path,
    splits: tuple[str, ...] = SPLITS,
    boundary_tolerance_px: float = 1.0,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = 42,
    stratify_by_orientation: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Step 1: Pool all source split JSONs, re-split into 70/15/15, validate annotations, and build canonical element table.

    Returns:
        images_by_split: dict of split -> images DataFrame
        elements_df: combined validated elements DataFrame (all splits)
        rejected_df: rejected annotations with explicit rejection_reason
        quality_summary: dictionary of counts and integrity checks
    """
    raw_image_frames: list[pd.DataFrame] = []
    raw_ann_frames: list[pd.DataFrame] = []
    cat_lookup: dict[int, str] = {}

    for source_split in splits:
        json_path = raw_dir / f"{source_split}.json"
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        cat_lookup.update({c["id"]: c["name"] for c in coco["categories"]})
        images_df = pd.DataFrame(coco["images"]).copy()
        ann_df = pd.DataFrame(coco["annotations"]).copy()

        # Basic integrity assertions on images
        assert images_df["id"].is_unique, f"Duplicate image IDs in {source_split}"
        assert (images_df["width"] > 0).all() and (images_df["height"] > 0).all()

        images_df["source_split"] = source_split
        ann_df["source_split"] = source_split
        raw_image_frames.append(images_df)
        raw_ann_frames.append(ann_df)

    pooled_images_df = pd.concat(raw_image_frames, ignore_index=True)
    assert pooled_images_df["id"].is_unique, "Duplicate image IDs across pooled splits"

    pooled_images_df = assign_pooled_splits(
        pooled_images_df,
        split_ratios=split_ratios,
        random_state=random_state,
        stratify_by_orientation=stratify_by_orientation,
    )
    images_by_split: dict[str, pd.DataFrame] = {
        split: pooled_images_df[pooled_images_df["split"] == split].reset_index(drop=True)
        for split in SPLITS
    }

    ann_df = pd.concat(raw_ann_frames, ignore_index=True)

    # Unpack COCO bbox [x, y, w, h]
    bbox_cols = pd.DataFrame(
        ann_df["bbox"].tolist(),
        columns=["raw_x", "raw_y", "raw_w", "raw_h"],
        index=ann_df.index,
    ).apply(pd.to_numeric, errors="coerce")

    ann_df = pd.concat([ann_df, bbox_cols], axis=1)
    img_meta = pooled_images_df.set_index("id")
    ann_df["canvas_width_px"] = ann_df["image_id"].map(img_meta["width"]).astype(float)
    ann_df["canvas_height_px"] = ann_df["image_id"].map(img_meta["height"]).astype(float)
    ann_df["split"] = ann_df["image_id"].map(img_meta["split"]).fillna(ann_df["source_split"])

    # Clip tiny floating-point boundary overshoots within boundary_tolerance_px
    x_Clip = np.where(
        (ann_df["raw_x"] < 0) & (ann_df["raw_x"] >= -boundary_tolerance_px),
        0.0,
        ann_df["raw_x"],
    )
    y_Clip = np.where(
        (ann_df["raw_y"] < 0) & (ann_df["raw_y"] >= -boundary_tolerance_px),
        0.0,
        ann_df["raw_y"],
    )
    x2_raw = x_Clip + ann_df["raw_w"]
    y2_raw = y_Clip + ann_df["raw_h"]
    w_Clip = np.where(
        (x2_raw > ann_df["canvas_width_px"])
        & (x2_raw <= ann_df["canvas_width_px"] + boundary_tolerance_px),
        ann_df["canvas_width_px"] - x_Clip,
        ann_df["raw_w"],
    )
    h_Clip = np.where(
        (y2_raw > ann_df["canvas_height_px"])
        & (y2_raw <= ann_df["canvas_height_px"] + boundary_tolerance_px),
        ann_df["canvas_height_px"] - y_Clip,
        ann_df["raw_h"],
    )

    # Validation rules
    valid_img_ids = set(pooled_images_df["id"])
    is_finite = np.asarray(
        np.isfinite(ann_df[["raw_x", "raw_y", "raw_w", "raw_h"]]).all(axis=1),
        dtype=bool,
    )
    has_valid_img = np.asarray(ann_df["image_id"].isin(valid_img_ids), dtype=bool)
    has_valid_cat = np.asarray(ann_df["category_id"].isin(cat_lookup.keys()), dtype=bool)
    non_neg_origin = np.asarray((x_Clip >= 0) & (y_Clip >= 0), dtype=bool)
    pos_dims = np.asarray((w_Clip > 0) & (h_Clip > 0), dtype=bool)
    in_bounds = np.asarray(
        (x_Clip + w_Clip <= ann_df["canvas_width_px"].to_numpy() + 1e-5)
        & (y_Clip + h_Clip <= ann_df["canvas_height_px"].to_numpy() + 1e-5),
        dtype=bool,
    )

    valid_mask = (
        is_finite
        & has_valid_img
        & has_valid_cat
        & non_neg_origin
        & pos_dims
        & in_bounds
    )
    inv_mask = ~valid_mask

    # Determine explicit rejection reason for invalid rows
    if inv_mask.any():
        rej = ann_df.loc[inv_mask].copy()
        reasons = np.full(len(rej), "unknown_error", dtype=object)
        reasons = np.where(~in_bounds[inv_mask], "exceeds_canvas_bounds", reasons)
        reasons = np.where(~non_neg_origin[inv_mask], "negative_origin", reasons)
        reasons = np.where(~pos_dims[inv_mask], "non_positive_dimensions", reasons)
        reasons = np.where(~has_valid_cat[inv_mask], "invalid_category_id", reasons)
        reasons = np.where(~has_valid_img[inv_mask], "orphan_image_id", reasons)
        reasons = np.where(~is_finite[inv_mask], "non_finite_coordinates", reasons)
        rej["rejection_reason"] = reasons
        rejected_df = (
            rej[
                [
                    "id",
                    "image_id",
                    "split",
                    "source_split",
                    "category_id",
                    "raw_x",
                    "raw_y",
                    "raw_w",
                    "raw_h",
                    "canvas_width_px",
                    "canvas_height_px",
                    "rejection_reason",
                ]
            ]
            .rename(columns={"id": "element_id", "image_id": "poster_id"})
            .reset_index(drop=True)
        )
    else:
        rejected_df = pd.DataFrame()

    # Process valid elements
    val = ann_df.loc[valid_mask].copy()
    val["raw_x"] = x_Clip[valid_mask]
    val["raw_y"] = y_Clip[valid_mask]
    val["raw_w"] = w_Clip[valid_mask]
    val["raw_h"] = h_Clip[valid_mask]

    W = val["canvas_width_px"].to_numpy()
    H = val["canvas_height_px"].to_numpy()
    x = val["raw_x"].to_numpy()
    y = val["raw_y"].to_numpy()
    w = val["raw_w"].to_numpy()
    h = val["raw_h"].to_numpy()

    x0 = np.clip(x / W, 0.0, 1.0)
    y0 = np.clip(y / H, 0.0, 1.0)
    w_norm = np.clip(w / W, EPS, 1.0)
    h_norm = np.clip(h / H, EPS, 1.0)
    x1 = np.clip(x0 + w_norm, 0.0, 1.0)
    y1 = np.clip(y0 + h_norm, 0.0, 1.0)
    cx = np.clip(x0 + w_norm / 2.0, 0.0, 1.0)
    cy = np.clip(y0 + h_norm / 2.0, 0.0, 1.0)
    area_norm = w_norm * h_norm
    bbox_aspect = w_norm / (h_norm + EPS)

    cat_names = val["category_id"].map(cat_lookup)
    cat_slugs = cat_names.map(category_slug)

    elements_df = pd.DataFrame(
        {
            "poster_id": val["image_id"].to_numpy(dtype=np.int64),
            "element_id": val["id"].to_numpy(dtype=np.int64),
            "split": val["split"].to_numpy(),
            "source_split": val["source_split"].to_numpy(),
            "category_id": val["category_id"].to_numpy(dtype=np.int64),
            "category_name": cat_names.to_numpy(),
            "category_slug": cat_slugs.to_numpy(),
            "is_unknown": (val["category_id"].to_numpy() == UNKNOWN_CAT_ID),
            "canvas_width_px": W,
            "canvas_height_px": H,
            "raw_x": x,
            "raw_y": y,
            "raw_w": w,
            "raw_h": h,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "w_norm": w_norm,
            "h_norm": h_norm,
            "cx": cx,
            "cy": cy,
            "area_norm": area_norm,
            "bbox_aspect_ratio": bbox_aspect,
            "log_w_norm": np.log(w_norm + EPS),
            "log_h_norm": np.log(h_norm + EPS),
            "log_area_norm": np.log(area_norm + EPS),
            "log_bbox_aspect_ratio": np.log(bbox_aspect + EPS),
            "left_margin": x0,
            "right_margin": np.maximum(0.0, 1.0 - x1),
            "top_margin": y0,
            "bottom_margin": np.maximum(0.0, 1.0 - y1),
        }
    )

    elements_df["min_horizontal_margin"] = np.minimum(
        elements_df["left_margin"], elements_df["right_margin"]
    )
    elements_df["min_vertical_margin"] = np.minimum(
        elements_df["top_margin"], elements_df["bottom_margin"]
    )
    elements_df["center_dist_to_edge"] = np.minimum.reduce(
        [
            elements_df["cx"],
            1.0 - elements_df["cx"],
            elements_df["cy"],
            1.0 - elements_df["cy"],
        ]
    )

    # Within-poster relative size ranks and ratios
    poster_med_area = elements_df.groupby("poster_id")["area_norm"].transform("median")
    elements_df["ratio_to_poster_median_area"] = elements_df["area_norm"] / (poster_med_area + EPS)
    elements_df["poster_area_rank_pct"] = elements_df.groupby("poster_id")["area_norm"].rank(
        pct=True
    )
    elements_df["category_area_rank_pct"] = elements_df.groupby(["poster_id", "category_slug"])[
        "area_norm"
    ].rank(pct=True)
    desc_rank = elements_df.groupby("poster_id")["area_norm"].rank(ascending=False, method="min")
    elements_df["is_top3_largest"] = desc_rank <= 3

    # Positional bins
    elements_df["horiz_zone"] = np.select(
        [elements_df["cx"] < 1.0 / 3.0, elements_df["cx"] <= 2.0 / 3.0],
        ["left", "center"],
        default="right",
    )
    elements_df["vert_zone"] = np.select(
        [elements_df["cy"] < 0.25, elements_df["cy"] < 0.50, elements_df["cy"] < 0.75],
        ["top", "upper_mid", "lower_mid"],
        default="bottom",
    )
    gx = np.clip((elements_df["cx"].to_numpy() * 4).astype(int), 0, 3)
    gy = np.clip((elements_df["cy"].to_numpy() * 4).astype(int), 0, 3)
    elements_df["grid_4x4_cell"] = gy * 4 + gx

    quality_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_imgs = images_by_split[split]
        split_valid = elements_df[elements_df["split"] == split]
        split_anns_count = int((ann_df["split"] == split).sum())
        split_rej_count = int((rejected_df["split"] == split).sum()) if len(rejected_df) > 0 else 0
        quality_rows.append(
            {
                "split": split,
                "total_posters": len(split_imgs),
                "posters_retained": len(split_imgs),
                "posters_with_valid_elements": int(split_valid["poster_id"].nunique()),
                "total_annotations": split_anns_count,
                "valid_annotations": len(split_valid),
                "rejected_annotations": split_rej_count,
            }
        )

    # Fit category reference medians STRICTLY on new train split
    train_els = elements_df[elements_df["split"] == "train"]
    train_cat_medians = (
        train_els.groupby("category_slug")[["w_norm", "h_norm", "area_norm"]]
        .median()
        .to_dict(orient="index")
    )
    global_train_medians = {
        "w_norm": float(train_els["w_norm"].median()) if len(train_els) > 0 else 0.0,
        "h_norm": float(train_els["h_norm"].median()) if len(train_els) > 0 else 0.0,
        "area_norm": float(train_els["area_norm"].median()) if len(train_els) > 0 else 0.0,
    }

    ref_w = elements_df["category_slug"].map(
        lambda s: train_cat_medians.get(s, global_train_medians)["w_norm"]
    )
    ref_h = elements_df["category_slug"].map(
        lambda s: train_cat_medians.get(s, global_train_medians)["h_norm"]
    )
    ref_area = elements_df["category_slug"].map(
        lambda s: train_cat_medians.get(s, global_train_medians)["area_norm"]
    )

    elements_df["ratio_w_to_train_cat_median"] = elements_df["w_norm"] / (ref_w + EPS)
    elements_df["ratio_h_to_train_cat_median"] = elements_df["h_norm"] / (ref_h + EPS)
    elements_df["ratio_area_to_train_cat_median"] = elements_df["area_norm"] / (ref_area + EPS)

    source_to_new_counts = (
        pooled_images_df.groupby(["source_split", "split"]).size().unstack(fill_value=0).to_dict()
    )

    quality_summary = {
        "split_policy": {
            "mode": "pooled_resplit",
            "source_splits": list(splits),
            "split_ratios": list(split_ratios),
            "random_state": random_state,
            "stratify_by_orientation": stratify_by_orientation,
        },
        "splits": quality_rows,
        "source_split_to_new_split_counts": source_to_new_counts,
        "train_category_reference_medians": train_cat_medians,
        "rejected_by_reason": (
            rejected_df["rejection_reason"].value_counts().to_dict()
            if len(rejected_df) > 0
            else {}
        ),
    }

    return images_by_split, elements_df, rejected_df, quality_summary


def build_relations_and_spatial_summaries(
    elements_df: pd.DataFrame,
    align_tol: float = 0.015,
    knn_k: int = 3,
) -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    """Step 2: Build sparse pairwise spatial relationship graph (Group C) and per-poster relational stats.

    Edge selection policy:
      1. Positive overlap or containment (IoU > 0 or intersection_over_min_area > 0.02)
      2. Top-k nearest neighbors by normalized center distance (k=3)
      3. Nearest directional neighbor in each cardinal direction (left, right, above, below)
      4. Domain pairing hypotheses:
         - Figure/Table -> nearest Caption within distance <= 0.15
         - Section -> nearest Text/List/Figure directly below within same column
    """
    edge_records: list[dict[str, Any]] = []
    poster_rel_stats: dict[int, dict[str, float]] = {}

    # Pre-extract arrays grouped by poster_id for fast numpy processing
    for poster_id, grp in elements_df.groupby("poster_id", sort=False):
        pid = int(poster_id)
        n = len(grp)
        if n <= 1:
            poster_rel_stats[pid] = {
                "overlap_pair_count": 0.0,
                "overlap_pair_fraction": 0.0,
                "max_pair_iou": 0.0,
                "mean_intersection_over_smaller_box": 0.0,
                "frac_aligned_left": 0.0,
                "frac_aligned_right": 0.0,
                "frac_aligned_top": 0.0,
                "frac_aligned_bottom": 0.0,
                "frac_aligned_cx": 0.0,
                "frac_aligned_cy": 0.0,
                "mean_nn_distance": 0.0,
                "std_nn_distance": 0.0,
                "fig_caption_paired_fraction": 0.0,
            }
            continue

        split = grp["split"].iloc[0]
        el_ids = grp["element_id"].to_numpy(dtype=np.int64)
        cat_slugs = grp["category_slug"].to_numpy()
        x0 = grp["x0"].to_numpy()
        y0 = grp["y0"].to_numpy()
        x1 = grp["x1"].to_numpy()
        y1 = grp["y1"].to_numpy()
        w = grp["w_norm"].to_numpy()
        h = grp["h_norm"].to_numpy()
        cx = grp["cx"].to_numpy()
        cy = grp["cy"].to_numpy()
        area = grp["area_norm"].to_numpy()

        # Pairwise matrices (n x n), where row=source i, col=target j
        dx = cx[None, :] - cx[:, None]
        dy = cy[None, :] - cy[:, None]
        abs_dx = np.abs(dx)
        abs_dy = np.abs(dy)
        dist = np.hypot(dx, dy)
        np.fill_diagonal(dist, np.inf)

        # Interval gaps (negative if overlapping)
        horiz_gap = np.maximum(x0[None, :] - x1[:, None], x0[:, None] - x1[None, :])
        vert_gap = np.maximum(y0[None, :] - y1[:, None], y0[:, None] - y1[None, :])

        # Intersection & IoU
        inter_w = np.maximum(0.0, np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :]))
        inter_h = np.maximum(0.0, np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :]))
        inter_area = inter_w * inter_h
        np.fill_diagonal(inter_area, 0.0)

        union_area = area[:, None] + area[None, :] - inter_area + EPS
        iou = inter_area / union_area
        min_area = np.minimum(area[:, None], area[None, :]) + EPS
        inter_over_min = inter_area / min_area
        contains_target = (
            (x0[:, None] <= x0[None, :] + 1e-4)
            & (y0[:, None] <= y0[None, :] + 1e-4)
            & (x1[:, None] >= x1[None, :] - 1e-4)
            & (y1[:, None] >= y1[None, :] - 1e-4)
        )
        np.fill_diagonal(contains_target, False)

        # Alignment checks (upper triangle for poster-level summary)
        tri_u = np.triu_indices(n, k=1)
        n_pairs = len(tri_u[0])

        al_left = np.abs(x0[:, None] - x0[None, :]) <= align_tol
        al_right = np.abs(x1[:, None] - x1[None, :]) <= align_tol
        al_top = np.abs(y0[:, None] - y0[None, :]) <= align_tol
        al_bottom = np.abs(y1[:, None] - y1[None, :]) <= align_tol
        al_cx = abs_dx <= align_tol
        al_cy = abs_dy <= align_tol

        # Same column / row hypotheses: significant x-interval or y-interval overlap
        x_overlap_ratio = inter_w / (np.minimum(w[:, None], w[None, :]) + EPS)
        y_overlap_ratio = inter_h / (np.minimum(h[:, None], h[None, :]) + EPS)
        same_col = (x_overlap_ratio >= 0.5) & (horiz_gap <= 0.0)
        same_row = (y_overlap_ratio >= 0.5) & (vert_gap <= 0.0)

        # Poster-level overlap & alignment statistics
        iou_u = iou[tri_u]
        iom_u = inter_over_min[tri_u]
        ov_mask = iou_u > 1e-4
        ov_count = int(np.sum(ov_mask))
        nn_dists = np.min(dist, axis=1)

        # Figure-caption pairing check
        fig_indices = np.where((cat_slugs == "figure") | (cat_slugs == "table"))[0]
        cap_indices = np.where(cat_slugs == "caption")[0]
        paired_figs = 0
        fig_cap_pairs: set[tuple[int, int]] = set()
        if len(fig_indices) > 0 and len(cap_indices) > 0:
            for fi in fig_indices:
                sub_d = dist[fi, cap_indices]
                best_ci_idx = int(np.argmin(sub_d))
                if sub_d[best_ci_idx] <= 0.25:
                    paired_figs += 1
                    fig_cap_pairs.add((int(fi), int(cap_indices[best_ci_idx])))

        poster_rel_stats[pid] = {
            "overlap_pair_count": float(ov_count),
            "overlap_pair_fraction": float(ov_count / n_pairs) if n_pairs > 0 else 0.0,
            "max_pair_iou": float(np.max(iou_u)) if n_pairs > 0 else 0.0,
            "mean_intersection_over_smaller_box": (
                float(np.mean(iom_u[ov_mask])) if ov_count > 0 else 0.0
            ),
            "frac_aligned_left": float(np.mean(al_left[tri_u])),
            "frac_aligned_right": float(np.mean(al_right[tri_u])),
            "frac_aligned_top": float(np.mean(al_top[tri_u])),
            "frac_aligned_bottom": float(np.mean(al_bottom[tri_u])),
            "frac_aligned_cx": float(np.mean(al_cx[tri_u])),
            "frac_aligned_cy": float(np.mean(al_cy[tri_u])),
            "mean_nn_distance": float(np.mean(nn_dists)),
            "std_nn_distance": float(np.std(nn_dists)),
            "fig_caption_paired_fraction": (
                float(paired_figs / len(fig_indices)) if len(fig_indices) > 0 else 1.0
            ),
        }

        # Select sparse graph edges per element i
        k_eff = min(knn_k, n - 1)
        knn_Matrix = np.argsort(dist, axis=1)[:, :k_eff]

        for i in range(n):
            selected_targets: dict[int, list[str]] = {}

            # 1. KNN edges
            for j in knn_Matrix[i]:
                selected_targets.setdefault(int(j), []).append("knn")

            # 2. Overlap / containment edges
            ov_js = np.where((iou[i] > 0.01) | contains_target[i])[0]
            for j in ov_js:
                if j != i:
                    selected_targets.setdefault(int(j), []).append("overlap")

            # 3. Directional nearest neighbors (above, below, left, right)
            # Below: dy > 0 and same_col
            below_js = np.where((dy[i] > 0) & same_col[i])[0]
            if len(below_js) > 0:
                best_b = int(below_js[np.argmin(dist[i, below_js])])
                selected_targets.setdefault(best_b, []).append("nearest_below_col")

            # Right: dx > 0 and same_row
            right_js = np.where((dx[i] > 0) & same_row[i])[0]
            if len(right_js) > 0:
                best_r = int(right_js[np.argmin(dist[i, right_js])])
                selected_targets.setdefault(best_r, []).append("nearest_right_row")

            # 4. Figure-caption pair
            for fi, ci in fig_cap_pairs:
                if i == fi:
                    selected_targets.setdefault(ci, []).append("figure_caption_hypothesis")

            for j, reasons in selected_targets.items():
                # Primary direction label
                if abs_dy[i, j] >= abs_dx[i, j]:
                    direction = "below" if dy[i, j] > 0 else "above"
                else:
                    direction = "right_of" if dx[i, j] > 0 else "left_of"

                edge_records.append(
                    {
                        "poster_id": pid,
                        "split": split,
                        "source_element_id": int(el_ids[i]),
                        "target_element_id": int(el_ids[j]),
                        "source_category": str(cat_slugs[i]),
                        "target_category": str(cat_slugs[j]),
                        "dx": float(dx[i, j]),
                        "dy": float(dy[i, j]),
                        "abs_dx": float(abs_dx[i, j]),
                        "abs_dy": float(abs_dy[i, j]),
                        "distance": float(dist[i, j]),
                        "horizontal_gap": float(horiz_gap[i, j]),
                        "vertical_gap": float(vert_gap[i, j]),
                        "iou": float(iou[i, j]),
                        "intersection_over_min_area": float(inter_over_min[i, j]),
                        "contains_target": bool(contains_target[i, j]),
                        "area_ratio": float(area[j] / (area[i] + EPS)),
                        "aligned_left": bool(al_left[i, j]),
                        "aligned_right": bool(al_right[i, j]),
                        "aligned_top": bool(al_top[i, j]),
                        "aligned_bottom": bool(al_bottom[i, j]),
                        "aligned_cx": bool(al_cx[i, j]),
                        "aligned_cy": bool(al_cy[i, j]),
                        "same_column_hypothesis": bool(same_col[i, j]),
                        "same_row_hypothesis": bool(same_row[i, j]),
                        "direction": direction,
                        "edge_types": ",".join(sorted(set(reasons))),
                    }
                )

    relations_df = pd.DataFrame(edge_records)
    return relations_df, poster_rel_stats


def build_posters_table(
    images_by_split: dict[str, pd.DataFrame],
    elements_df: pd.DataFrame,
    poster_rel_stats: dict[int, dict[str, float]],
    grid_res: int = 32,
) -> pd.DataFrame:
    """Step 3: Build canonical poster & global composition table (`posters.parquet`, Groups A & D)."""
    all_images = pd.concat(list(images_by_split.values()), ignore_index=True)
    el_groups = {int(pid): grp for pid, grp in elements_df.groupby("poster_id", sort=False)}

    poster_rows: list[dict[str, Any]] = []

    for _, img in all_images.iterrows():
        pid = int(img["id"])
        split = str(img["split"])
        source_split = str(img["source_split"]) if "source_split" in img else split
        W = float(img["width"])
        H = float(img["height"])
        aspect = W / H
        log_aspect = float(np.log(aspect + EPS))

        if aspect < 0.95:
            orientation = "portrait"
        elif aspect > 1.05:
            orientation = "landscape"
        else:
            orientation = "square"

        grp = el_groups.get(pid)
        if grp is None or len(grp) == 0:
            # Handle degenerate poster without dropping it
            row: dict[str, Any] = {
                "poster_id": pid,
                "split": split,
                "source_split": source_split,
                "canvas_width_px": W,
                "canvas_height_px": H,
                "aspect_ratio": aspect,
                "log_aspect_ratio": log_aspect,
                "orientation": orientation,
                "is_landscape": float(orientation == "landscape"),
                "is_portrait": float(orientation == "portrait"),
                "total_element_count": 0,
                "known_element_count": 0,
                "unknown_element_count": 0,
                "unknown_element_fraction": 0.0,
            }
            poster_rows.append(row)
            continue

        total_count = len(grp)
        unk_mask = grp["is_unknown"].to_numpy()
        unk_count = int(np.sum(unk_mask))
        known_count = total_count - unk_count

        cat_slugs_arr = grp["category_slug"].to_numpy()
        cat_ids = grp["category_id"].to_numpy()
        x0_all = grp["x0"].to_numpy()
        y0_all = grp["y0"].to_numpy()
        x1_all = grp["x1"].to_numpy()
        y1_all = grp["y1"].to_numpy()
        w_all = grp["w_norm"].to_numpy()
        h_all = grp["h_norm"].to_numpy()
        cx_all = grp["cx"].to_numpy()
        cy_all = grp["cy"].to_numpy()
        area_all = grp["area_norm"].to_numpy()
        log_area_all = grp["log_area_norm"].to_numpy()

        row = {
            "poster_id": pid,
            "split": split,
            "source_split": source_split,
            "canvas_width_px": W,
            "canvas_height_px": H,
            "aspect_ratio": aspect,
            "log_aspect_ratio": log_aspect,
            "orientation": orientation,
            "is_landscape": float(orientation == "landscape"),
            "is_portrait": float(orientation == "portrait"),
            "total_element_count": total_count,
            "known_element_count": known_count,
            "unknown_element_count": unk_count,
            "unknown_element_fraction": float(unk_count / total_count) if total_count > 0 else 0.0,
        }

        # --- Group A2 & A4: Category Inventory & Geometry Quantiles (Single-pass NumPy) ---
        denom_known = max(1, known_count)
        geom_5_stack = np.vstack([w_all, h_all, area_all, cx_all, cy_all])
        gcol_names = ("w_norm", "h_norm", "area_norm", "cx", "cy")

        for slug in KNOWN_CATEGORY_SLUGS:
            c_mask = cat_slugs_arr == slug
            c = int(np.sum(c_mask))
            row[f"count_{slug}"] = c
            row[f"log1p_count_{slug}"] = float(np.log1p(c))
            row[f"has_{slug}"] = float(c > 0)
            row[f"fraction_{slug}"] = float(c / denom_known)

            if c > 0:
                row[f"area_sum_{slug}"] = float(np.sum(area_all[c_mask]))
                if c == 1:
                    single_vals = geom_5_stack[:, c_mask][:, 0]
                    for g_i, gcol in enumerate(gcol_names):
                        v = float(single_vals[g_i])
                        row[f"{gcol}_p25_{slug}"] = v
                        row[f"{gcol}_p50_{slug}"] = v
                        row[f"{gcol}_p75_{slug}"] = v
                else:
                    q_mat = np.percentile(geom_5_stack[:, c_mask], [25, 50, 75], axis=1)
                    for g_i, gcol in enumerate(gcol_names):
                        row[f"{gcol}_p25_{slug}"] = float(q_mat[0, g_i])
                        row[f"{gcol}_p50_{slug}"] = float(q_mat[1, g_i])
                        row[f"{gcol}_p75_{slug}"] = float(q_mat[2, g_i])
            else:
                row[f"area_sum_{slug}"] = 0.0
                for gcol in gcol_names:
                    row[f"{gcol}_p25_{slug}"] = np.nan
                    row[f"{gcol}_p50_{slug}"] = np.nan
                    row[f"{gcol}_p75_{slug}"] = np.nan

        row["visual_element_count"] = row["count_figure"] + row["count_table"]
        row["textual_element_count"] = (
            row["count_text"] + row["count_list"] + row["count_caption"]
        )
        row["heading_element_count"] = row["count_title"] + row["count_section"]
        row["media_to_text_count_ratio"] = float(
            row["visual_element_count"] / (row["textual_element_count"] + 1.0)
        )

        # --- Group A3: Overall Box-Size Quantiles (Single vectorized call) ---
        overall_4_stack = np.vstack([w_all, h_all, area_all, log_area_all])
        ov_q = np.percentile(overall_4_stack, [10, 25, 50, 75, 90], axis=1)
        for col_i, col_name in enumerate(("w_norm", "h_norm", "area_norm", "log_area_norm")):
            row[f"{col_name}_p10"] = float(ov_q[0, col_i])
            row[f"{col_name}_p25"] = float(ov_q[1, col_i])
            row[f"{col_name}_p50"] = float(ov_q[2, col_i])
            row[f"{col_name}_p75"] = float(ov_q[3, col_i])
            row[f"{col_name}_p90"] = float(ov_q[4, col_i])
            row[f"{col_name}_iqr"] = float(ov_q[3, col_i] - ov_q[1, col_i])

        row["visual_area_sum"] = row["area_sum_figure"] + row["area_sum_table"]
        row["textual_area_sum"] = (
            row["area_sum_text"] + row["area_sum_list"] + row["area_sum_caption"]
        )
        row["header_area_sum"] = row["area_sum_title"] + row["area_sum_author_info"]
        total_box_area_sum = float(np.sum(area_all))
        row["sum_box_area_fraction"] = total_box_area_sum
        row["visual_to_body_area_ratio"] = float(
            row["visual_area_sum"]
            / (row["visual_area_sum"] + row["textual_area_sum"] + EPS)
        )

        # --- Group A5: Content Extent & Margins ---
        c_left = float(np.min(x0_all))
        c_right = float(max(0.0, 1.0 - np.max(x1_all)))
        c_top = float(np.min(y0_all))
        c_bottom = float(max(0.0, 1.0 - np.max(y1_all)))
        ext_w = float(max(0.0, 1.0 - c_left - c_right))
        ext_h = float(max(0.0, 1.0 - c_top - c_bottom))

        row["content_left_margin"] = c_left
        row["content_right_margin"] = c_right
        row["content_top_margin"] = c_top
        row["content_bottom_margin"] = c_bottom
        row["content_extent_width"] = ext_w
        row["content_extent_height"] = ext_h
        row["content_extent_area"] = ext_w * ext_h

        # Header band bottom Y
        hdr_mask = ((cat_ids == 487601213) | (cat_ids == 674852286)) & (cy_all < 0.35)
        row["header_band_bottom_y"] = (
            float(np.max(y1_all[hdr_mask])) if np.any(hdr_mask) else float(c_top)
        )

        # --- Group A6 & Group D: Rasterized Union Coverage, Grids, Profiles & Columns ---
        # Rasterize on grid_res x grid_res (32x32) normalized canvas
        canvas_mask = np.zeros((grid_res, grid_res), dtype=np.float32)
        body_horiz_profile = np.zeros(grid_res, dtype=np.float32)
        body_vert_profile = np.zeros(grid_res, dtype=np.float32)
        text_mask = np.zeros((grid_res, grid_res), dtype=np.float32)
        vis_mask = np.zeros((grid_res, grid_res), dtype=np.float32)

        cat_ids = grp["category_id"].to_numpy()

        for bx0, by0, bx1, by1, cid in zip(x0_all, y0_all, x1_all, y1_all, cat_ids):
            ix0 = max(0, min(grid_res - 1, int(np.floor(bx0 * grid_res))))
            ix1 = max(ix0 + 1, min(grid_res, int(np.ceil(bx1 * grid_res))))
            iy0 = max(0, min(grid_res - 1, int(np.floor(by0 * grid_res))))
            iy1 = max(iy0 + 1, min(grid_res, int(np.ceil(by1 * grid_res))))

            canvas_mask[iy0:iy1, ix0:ix1] = 1.0
            if cid in TEXTUAL_CAT_IDS:
                text_mask[iy0:iy1, ix0:ix1] = 1.0
            elif cid in VISUAL_CAT_IDS:
                vis_mask[iy0:iy1, ix0:ix1] = 1.0

            if cid in BODY_STRUCT_CAT_IDS and (bx1 - bx0) < 0.80:
                # Accumulate vertical span into horizontal profile for column detection
                body_horiz_profile[ix0:ix1] += float(by1 - by0)
                body_vert_profile[iy0:iy1] += float(bx1 - bx0)

        union_cov = float(np.mean(canvas_mask))
        row["union_box_coverage"] = union_cov
        row["overlap_excess"] = max(0.0, total_box_area_sum - union_cov)
        row["unannotated_whitespace_fraction"] = max(0.0, 1.0 - union_cov)

        # 4x4 Occupancy Grid (reshape 32x32 -> 4x8x4x8)
        grid_4x4 = canvas_mask.reshape(4, grid_res // 4, 4, grid_res // 4).mean(axis=(1, 3))
        for r_idx in range(4):
            for c_idx in range(4):
                row[f"occ_grid_r{r_idx}_c{c_idx}"] = float(grid_4x4[r_idx, c_idx])

        # Category-group regional occupancy summaries
        half = grid_res // 2
        row["text_occ_top_half"] = float(text_mask[:half, :].mean())
        row["text_occ_bottom_half"] = float(text_mask[half:, :].mean())
        row["visual_occ_left_half"] = float(vis_mask[:, :half].mean())
        row["visual_occ_right_half"] = float(vis_mask[:, half:].mean())
        row["visual_occ_center"] = float(
            vis_mask[grid_res // 4 : 3 * grid_res // 4, grid_res // 4 : 3 * grid_res // 4].mean()
        )

        # 16-bin Horizontal and Vertical Occupancy Profiles
        h_prof_16 = canvas_mask.mean(axis=0).reshape(16, grid_res // 16).mean(axis=1)
        v_prof_16 = canvas_mask.mean(axis=1).reshape(16, grid_res // 16).mean(axis=1)
        for b_idx in range(16):
            row[f"horiz_occ_{b_idx:02d}"] = float(h_prof_16[b_idx])
            row[f"vert_occ_{b_idx:02d}"] = float(v_prof_16[b_idx])

        # --- Group D3: Column & Row Structure Detection ---
        # Smooth body_horiz_profile to detect vertical columns and gutters
        if body_horiz_profile.max() > 0:
            prof_norm = body_horiz_profile / body_horiz_profile.max()
            smoothed = np.convolve(prof_norm, np.array([0.2, 0.6, 0.2]), mode="same")
            peaks, props = find_peaks(smoothed, height=0.25, distance=4, prominence=0.08)
            est_cols = max(1, min(6, len(peaks)))
            # Column assignment confidence based on peak-to-valley contrast
            if len(peaks) >= 2:
                valleys = []
                for p_idx in range(len(peaks) - 1):
                    seg = smoothed[peaks[p_idx] : peaks[p_idx + 1] + 1]
                    valleys.append(float(np.min(seg)))
                mean_peak = float(np.mean(smoothed[peaks]))
                mean_valley = float(np.mean(valleys))
                col_conf = float(np.clip((mean_peak - mean_valley) / (mean_peak + EPS), 0.0, 1.0))
                gutter_width_est = float(np.mean(smoothed < 0.25))
            else:
                col_conf = 0.5
                gutter_width_est = float(1.0 - union_cov)
        else:
            est_cols = 1
            col_conf = 0.0
            gutter_width_est = 1.0

        if body_vert_profile.max() > 0:
            v_norm = body_vert_profile / body_vert_profile.max()
            v_smooth = np.convolve(v_norm, np.array([0.2, 0.6, 0.2]), mode="same")
            v_peaks, _ = find_peaks(v_smooth, height=0.25, distance=4, prominence=0.08)
            est_rows = max(1, min(6, len(v_peaks)))
        else:
            est_rows = 1

        w_all = grp["w_norm"].to_numpy()
        col_unit_width = 1.0 / max(1, est_cols)
        row["estimated_column_count"] = int(est_cols)
        row["column_assignment_confidence"] = col_conf
        row["estimated_gutter_fraction"] = gutter_width_est
        row["estimated_row_band_count"] = int(est_rows)
        row["full_width_region_count"] = int(np.sum(w_all >= 0.70))
        row["multi_column_span_count"] = int(
            np.sum((w_all >= 1.5 * col_unit_width) & (w_all < 0.70))
        )

        # --- Group D5: Balance, Symmetry & Center of Mass ---
        left_occ = float(h_prof_16[:8].mean())
        right_occ = float(h_prof_16[8:].mean())
        top_occ = float(v_prof_16[:8].mean())
        bot_occ = float(v_prof_16[8:].mean())

        row["left_right_area_balance"] = float(
            (left_occ - right_occ) / (left_occ + right_occ + EPS)
        )
        row["top_bottom_area_balance"] = float(
            (top_occ - bot_occ) / (top_occ + bot_occ + EPS)
        )

        # Cosine symmetry of horizontal and vertical profiles
        h_left, h_right_rev = h_prof_16[:8], h_prof_16[8:][::-1]
        h_denom = float(np.linalg.norm(h_left) * np.linalg.norm(h_right_rev))
        row["horiz_profile_symmetry"] = (
            float(np.dot(h_left, h_right_rev) / h_denom) if h_denom > EPS else 0.0
        )

        v_top, v_bot_rev = v_prof_16[:8], v_prof_16[8:][::-1]
        v_denom = float(np.linalg.norm(v_top) * np.linalg.norm(v_bot_rev))
        row["vert_profile_symmetry"] = (
            float(np.dot(v_top, v_bot_rev) / v_denom) if v_denom > EPS else 0.0
        )

        row["center_of_mass_x"] = float(np.sum(cx_all * area_all) / (total_box_area_sum + EPS))
        row["center_of_mass_y"] = float(np.sum(cy_all * area_all) / (total_box_area_sum + EPS))

        # Merge relational summary statistics (overlap & alignment regularity)
        rel_stat = poster_rel_stats.get(pid, {})
        row.update(rel_stat)

        poster_rows.append(row)

    posters_df = pd.DataFrame(poster_rows)
    return posters_df


def generate_feature_schema_metadata(posters_df: pd.DataFrame, elements_df: pd.DataFrame, relations_df: pd.DataFrame) -> dict[str, Any]:
    """Generate comprehensive versioned feature schema documentation."""
    non_feature_poster_cols = {
        "poster_id",
        "split",
        "source_split",
        "canvas_width_px",
        "canvas_height_px",
        "orientation",
    }
    poster_numeric_cols = [
        c for c in posters_df.columns if c not in non_feature_poster_cols
    ]

    # Categorize poster features into logical groups for structured modeling & ablation
    feature_groups = {
        "canvas_geometry": [
            "aspect_ratio",
            "log_aspect_ratio",
            "is_landscape",
            "is_portrait",
            "total_element_count",
            "known_element_count",
        ],
        "category_inventory": [
            c
            for c in poster_numeric_cols
            if c.startswith(("count_", "log1p_count_", "has_", "fraction_"))
            or c
            in {
                "visual_element_count",
                "textual_element_count",
                "heading_element_count",
                "media_to_text_count_ratio",
            }
        ],
        "size_and_capacity": [
            c
            for c in poster_numeric_cols
            if any(
                k in c
                for k in (
                    "_p10",
                    "_p25",
                    "_p50",
                    "_p75",
                    "_p90",
                    "_iqr",
                    "area_sum_",
                    "visual_area_sum",
                    "textual_area_sum",
                    "header_area_sum",
                    "visual_to_body_area_ratio",
                )
            )
        ],
        "coverage_and_margins": [
            "sum_box_area_fraction",
            "union_box_coverage",
            "overlap_excess",
            "unannotated_whitespace_fraction",
            "overlap_pair_count",
            "overlap_pair_fraction",
            "max_pair_iou",
            "mean_intersection_over_smaller_box",
            "content_left_margin",
            "content_right_margin",
            "content_top_margin",
            "content_bottom_margin",
            "content_extent_width",
            "content_extent_height",
            "content_extent_area",
            "header_band_bottom_y",
        ],
        "spatial_occupancy_and_profiles": [
            c
            for c in poster_numeric_cols
            if c.startswith(("occ_grid_", "horiz_occ_", "vert_occ_", "text_occ_", "visual_occ_"))
        ],
        "structure_columns_and_alignment": [
            "estimated_column_count",
            "column_assignment_confidence",
            "estimated_gutter_fraction",
            "estimated_row_band_count",
            "full_width_region_count",
            "multi_column_span_count",
            "frac_aligned_left",
            "frac_aligned_right",
            "frac_aligned_top",
            "frac_aligned_bottom",
            "frac_aligned_cx",
            "frac_aligned_cy",
            "left_right_area_balance",
            "top_bottom_area_balance",
            "horiz_profile_symmetry",
            "vert_profile_symmetry",
            "center_of_mass_x",
            "center_of_mass_y",
            "mean_nn_distance",
            "std_nn_distance",
            "fig_caption_paired_fraction",
        ],
    }

    return {
        "schema_version": "2.0.0",
        "dataset": "SciPostLayout v2",
        "tables": {
            "posters": {
                "row_count": len(posters_df),
                "total_columns": len(posters_df.columns),
                "numeric_feature_columns": poster_numeric_cols,
                "feature_groups": feature_groups,
            },
            "elements": {
                "row_count": len(elements_df),
                "columns": list(elements_df.columns),
            },
            "relations": {
                "row_count": len(relations_df),
                "columns": list(relations_df.columns),
            },
        },
    }


def export_legacy_v1_compatible_parquets(
    posters_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    legacy_out_dir: Path,
    overwrite: bool = True,
) -> None:
    """Ensure legacy v1 parquet files exist and stay aligned with current splits."""
    legacy_out_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and (legacy_out_dir / "train_features.parquet").exists():
        return

    # Reconstruct the 92 legacy columns from elements_df & posters_df
    for split in SPLITS:
        p_sub = posters_df[posters_df["split"] == split].copy()
        e_sub = elements_df[elements_df["split"] == split].copy()

        df = pd.DataFrame(
            {
                "poster_id": p_sub["poster_id"].to_numpy(),
                "split": split,
                "aspect_ratio": p_sub["aspect_ratio"].to_numpy(),
                "pixel_width": p_sub["canvas_width_px"].to_numpy(),
                "pixel_height": p_sub["canvas_height_px"].to_numpy(),
                "element_count": p_sub["total_element_count"].to_numpy(),
                "element_density": p_sub["total_element_count"].astype(float).to_numpy(),
                "sum_bbox_area_ratio": p_sub["sum_box_area_fraction"].to_numpy(),
            }
        )

        geom = (
            e_sub.groupby("poster_id")
            .agg(
                median_element_area=("area_norm", "median"),
                median_element_width=("w_norm", "median"),
                median_element_height=("h_norm", "median"),
                median_center_x=("cx", "median"),
                median_center_y=("cy", "median"),
                spread_center_x=("cx", lambda s: s.std(ddof=0)),
                spread_center_y=("cy", lambda s: s.std(ddof=0)),
            )
            .reindex(df["poster_id"])
            .fillna(0.0)
            .reset_index(drop=True)
        )
        df = pd.concat([df, geom], axis=1)

        counts = df["element_count"].to_numpy()
        for cid, cname in SCIPOST_CATEGORIES.items():
            slug = category_slug(cname)
            c_sub = e_sub[e_sub["category_id"] == cid]
            c_cnt = (
                c_sub.groupby("poster_id")
                .size()
                .reindex(df["poster_id"], fill_value=0)
                .to_numpy()
            )
            df[f"count_{slug}"] = c_cnt
            df[f"proportion_{slug}"] = np.divide(
                c_cnt, counts, out=np.zeros(len(counts), dtype=float), where=counts > 0
            )
            c_geom = (
                c_sub.groupby("poster_id")
                .agg(
                    area=("area_norm", "median"),
                    width=("w_norm", "median"),
                    height=("h_norm", "median"),
                    center_x=("cx", "median"),
                    center_y=("cy", "median"),
                    spread_x=("cx", lambda s: s.std(ddof=0)),
                    spread_y=("cy", lambda s: s.std(ddof=0)),
                )
                .reindex(df["poster_id"])
                .fillna(0.0)
                .reset_index(drop=True)
            )
            for stat in c_geom.columns:
                df[f"{stat}_{slug}"] = c_geom[stat].to_numpy()

        df.to_parquet(legacy_out_dir / f"{split}_features.parquet", index=False)


def run_feature_pipeline_v2(
    raw_dir: str | Path,
    out_dir: str | Path,
    export_legacy: bool = True,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = 42,
    stratify_by_orientation: bool = True,
) -> dict[str, Any]:
    """Execute the full v2 multi-table feature engineering pipeline and save artifacts."""
    t0 = time.time()
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("[1/4] Pooling source splits, assigning 70/15/15 splits, and validating elements...")
    images_by_split, elements_df, rejected_df, quality_summary = validate_and_extract_elements(
        raw_path,
        split_ratios=split_ratios,
        random_state=random_state,
        stratify_by_orientation=stratify_by_orientation,
    )

    print("[2/4] Building sparse pairwise spatial relations graph...")
    relations_df, poster_rel_stats = build_relations_and_spatial_summaries(elements_df)

    print("[3/4] Building poster-level & global composition features...")
    posters_df = build_posters_table(images_by_split, elements_df, poster_rel_stats)

    print("[4/4] Saving versioned v2 feature tables and schema metadata...")
    posters_df.to_parquet(out_path / "posters.parquet", index=False)
    elements_df.to_parquet(out_path / "elements.parquet", index=False)
    relations_df.to_parquet(out_path / "relations.parquet", index=False)
    if len(rejected_df) > 0:
        rejected_df.to_parquet(out_path / "rejected_annotations.parquet", index=False)

    schema_meta = generate_feature_schema_metadata(posters_df, elements_df, relations_df)
    (out_path / "feature_schema.json").write_text(
        json.dumps(schema_meta, indent=2), encoding="utf-8"
    )

    elapsed = round(time.time() - t0, 2)
    report = {
        "pipeline_version": "2.0.0",
        "execution_time_seconds": elapsed,
        "quality_summary": quality_summary,
        "orientation_distribution": (
            posters_df.groupby(["split", "orientation"]).size().unstack(fill_value=0).to_dict()
        ),
        "estimated_columns_distribution": (
            posters_df["estimated_column_count"].value_counts().sort_index().to_dict()
        ),
    }
    (out_path / "feature_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if export_legacy:
        export_legacy_v1_compatible_parquets(posters_df, elements_df, out_path.parent)

    print(f"Done in {elapsed}s! Outputs written to {out_path}")
    return {
        "posters": posters_df,
        "elements": elements_df,
        "relations": relations_df,
        "rejected": rejected_df,
        "schema": schema_meta,
        "report": report,
    }
