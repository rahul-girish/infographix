"""Step 2: Build sparse pairwise spatial relationship graph and per-poster relational summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.config import EPS

RELATION_COLUMNS: tuple[str, ...] = (
    "poster_id",
    "split",
    "source_element_id",
    "target_element_id",
    "source_category",
    "target_category",
    "dx",
    "dy",
    "abs_dx",
    "abs_dy",
    "distance",
    "horizontal_gap",
    "vertical_gap",
    "iou",
    "intersection_over_min_area",
    "contains_target",
    "area_ratio",
    "aligned_left",
    "aligned_right",
    "aligned_top",
    "aligned_bottom",
    "aligned_cx",
    "aligned_cy",
    "same_column_hypothesis",
    "same_row_hypothesis",
    "direction",
    "edge_types",
)


def _empty_poster_rel_stats() -> dict[str, float]:
    """Return zeroed relational summary statistics for single-element posters."""
    return {
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


def _compute_poster_relations(
    pid: int,
    grp: pd.DataFrame,
    align_tol: float,
    knn_k: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Compute sparse edges and poster-level relational summaries for a single poster."""
    n = len(grp)
    if n <= 1:
        return [], _empty_poster_rel_stats()

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

    dx = cx[None, :] - cx[:, None]
    dy = cy[None, :] - cy[:, None]
    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    dist = np.hypot(dx, dy)
    np.fill_diagonal(dist, np.inf)

    horiz_gap = np.maximum(x0[None, :] - x1[:, None], x0[:, None] - x1[None, :])
    vert_gap = np.maximum(y0[None, :] - y1[:, None], y0[:, None] - y1[None, :])

    inter_w = np.maximum(
        0.0, np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    )
    inter_h = np.maximum(
        0.0, np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    )
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

    tri_u = np.triu_indices(n, k=1)
    n_pairs = len(tri_u[0])

    al_left = np.abs(x0[:, None] - x0[None, :]) <= align_tol
    al_right = np.abs(x1[:, None] - x1[None, :]) <= align_tol
    al_top = np.abs(y0[:, None] - y0[None, :]) <= align_tol
    al_bottom = np.abs(y1[:, None] - y1[None, :]) <= align_tol
    al_cx = abs_dx <= align_tol
    al_cy = abs_dy <= align_tol

    x_overlap_ratio = inter_w / (np.minimum(w[:, None], w[None, :]) + EPS)
    y_overlap_ratio = inter_h / (np.minimum(h[:, None], h[None, :]) + EPS)
    same_col = (x_overlap_ratio >= 0.5) & (horiz_gap <= 0.0)
    same_row = (y_overlap_ratio >= 0.5) & (vert_gap <= 0.0)

    iou_u = iou[tri_u]
    iom_u = inter_over_min[tri_u]
    ov_mask = iou_u > 1e-4
    ov_count = int(np.sum(ov_mask))
    nn_dists = np.min(dist, axis=1)

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

    rel_stats = {
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

    edges: list[dict[str, Any]] = []
    k_eff = min(knn_k, n - 1)
    knn_matrix = np.argsort(dist, axis=1)[:, :k_eff]

    for i in range(n):
        selected_targets: dict[int, list[str]] = {}

        for j in knn_matrix[i]:
            selected_targets.setdefault(int(j), []).append("knn")

        ov_js = np.where((iou[i] > 0.01) | contains_target[i])[0]
        for j in ov_js:
            if j != i:
                selected_targets.setdefault(int(j), []).append("overlap")

        below_js = np.where((dy[i] > 0) & same_col[i])[0]
        if len(below_js) > 0:
            best_b = int(below_js[np.argmin(dist[i, below_js])])
            selected_targets.setdefault(best_b, []).append("nearest_below_col")

        right_js = np.where((dx[i] > 0) & same_row[i])[0]
        if len(right_js) > 0:
            best_r = int(right_js[np.argmin(dist[i, right_js])])
            selected_targets.setdefault(best_r, []).append("nearest_right_row")

        for fi, ci in fig_cap_pairs:
            if i == fi:
                selected_targets.setdefault(ci, []).append("figure_caption_hypothesis")

        for j, reasons in selected_targets.items():
            if abs_dy[i, j] >= abs_dx[i, j]:
                direction = "below" if dy[i, j] > 0 else "above"
            else:
                direction = "right_of" if dx[i, j] > 0 else "left_of"

            edges.append(
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

    return edges, rel_stats


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

    for poster_id, grp in elements_df.groupby("poster_id", sort=False):
        pid = int(poster_id)
        edges, rel_stats = _compute_poster_relations(pid, grp, align_tol=align_tol, knn_k=knn_k)
        poster_rel_stats[pid] = rel_stats
        edge_records.extend(edges)

    relations_df = (
        pd.DataFrame(edge_records)
        if edge_records
        else pd.DataFrame(columns=list(RELATION_COLUMNS))
    )
    return relations_df, poster_rel_stats
