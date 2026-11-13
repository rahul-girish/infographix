"""Direct Constraint-Aware Nearest-Neighbour Layout Retrieval Engine for Infographix (v2.0).

Implements the primary modeling recommendation from `infographix_feature_engineering_spec.md`:
  - Separates feature groups (Canvas, Category Inventory, Area Capacity, Spatial Structure)
    and normalizes each group independently on the training split.
  - Supports querying by a runtime `PosterRequest` (user content specification without
    bounding boxes) as well as poster-to-poster structural retrieval for feature ablation.
  - Applies hard orientation/aspect constraints with graceful fallback for unusual queries.
  - Implements Maximal Marginal Relevance (MMR) diversity reranking over 4x4 spatial
    occupancy grids so top-k retrieved templates offer distinct layout arrangements.
  - Evaluates retrieval on simulated user requests constructed from held-out test posters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import RobustScaler

from src.features.config import EPS, KNOWN_CATEGORY_SLUGS
from src.features.request_schema import (
    PosterRequest,
    compute_request_candidate_compatibility,
    simulate_request_from_heldout_poster,
)

# Shared vocabulary between `PosterRequest.to_summary_dict()` and `posters.parquet`
SHARED_CANVAS_COLS = ["log_aspect_ratio", "is_landscape", "is_portrait"]

SHARED_INVENTORY_COLS = (
    [f"log1p_count_{s}" for s in KNOWN_CATEGORY_SLUGS]
    + [f"fraction_{s}" for s in KNOWN_CATEGORY_SLUGS]
    + [
        "known_element_count",
        "visual_element_count",
        "textual_element_count",
        "heading_element_count",
        "media_to_text_count_ratio",
    ]
)

SHARED_CAPACITY_COLS = (
    [f"area_sum_{s}" for s in KNOWN_CATEGORY_SLUGS]
    + [
        "visual_area_sum",
        "textual_area_sum",
        "header_area_sum",
        "sum_box_area_fraction",
        "visual_to_body_area_ratio",
        "area_norm_p25",
        "area_norm_p50",
        "area_norm_p75",
        "w_norm_p50",
        "h_norm_p50",
    ]
)

SHARED_STRUCTURE_COLS = ["estimated_column_count"]

POSTER_COMPOSITION_COLS = (
    ["estimated_column_count", "union_box_coverage", "horiz_profile_symmetry", "frac_aligned_left"]
    + [f"occ_grid_r{r}_c{c}" for r in range(4) for c in range(4)]
    + [f"horiz_occ_{b:02d}" for b in range(16)]
    + [f"vert_occ_{b:02d}" for b in range(16)]
)


class LayoutRetriever:
    """Group-normalized, constraint-aware layout retrieval engine."""

    def __init__(
        self,
        group_weights: dict[str, float] | None = None,
        mmr_lambda: float = 0.78,
        strict_orientation: bool = True,
    ):
        self.group_weights = group_weights or {
            "canvas": 2.5,
            "inventory": 3.0,
            "capacity": 2.5,
            "structure": 2.0,
        }
        self.mmr_lambda = mmr_lambda
        self.strict_orientation = strict_orientation

        self.train_posters_: pd.DataFrame | None = None
        self.scalers_: dict[str, tuple[SimpleImputer, RobustScaler]] = {}
        self.train_group_matrices_: dict[str, np.ndarray] = {}
        self.train_occ_grids_: np.ndarray | None = None
        self.cluster_labels_: np.ndarray | None = None

    def fit(
        self,
        train_posters_df: pd.DataFrame,
        cluster_labels: np.ndarray | None = None,
    ) -> LayoutRetriever:
        """Fit group-specific imputers and robust scalers strictly on training posters."""
        # Filter to valid reference posters with at least 1 known element
        valid_mask = (train_posters_df["known_element_count"].fillna(0) > 0).to_numpy(dtype=bool)
        self.train_posters_ = train_posters_df.loc[valid_mask].reset_index(drop=True).copy()
        if cluster_labels is not None:
            self.cluster_labels_ = np.asarray(cluster_labels, dtype=int)[valid_mask]
            self.train_posters_["archetype_cluster"] = self.cluster_labels_
        else:
            self.cluster_labels_ = None

        group_defs = {
            "canvas": SHARED_CANVAS_COLS,
            "inventory": SHARED_INVENTORY_COLS,
            "capacity": SHARED_CAPACITY_COLS,
            "structure": SHARED_STRUCTURE_COLS,
            "full_composition": POSTER_COMPOSITION_COLS,
        }

        self.scalers_ = {}
        self.train_group_matrices_ = {}

        for g_name, cols in group_defs.items():
            imp = SimpleImputer(strategy="median")
            scaler = RobustScaler()
            X_raw = self.train_posters_[cols].to_numpy(dtype=np.float64)
            X_norm = scaler.fit_transform(imp.fit_transform(X_raw))
            # Divide by sqrt(d) so group distances are invariant to number of columns in group
            X_norm = X_norm / np.sqrt(max(1, len(cols)))
            self.scalers_[g_name] = (imp, scaler)
            self.train_group_matrices_[g_name] = X_norm

        occ_cols = [f"occ_grid_r{r}_c{c}" for r in range(4) for c in range(4)]
        self.train_occ_grids_ = self.train_posters_[occ_cols].to_numpy(dtype=np.float64)
        self.train_poster_dicts_ = self.train_posters_.to_dict(orient="records")
        return self

    def _transform_query_group(self, req_dict: dict[str, Any], g_name: str, cols: list[str]) -> np.ndarray:
        imp, scaler = self.scalers_[g_name]
        X_raw = np.array([[float(req_dict.get(c, 0.0)) for c in cols]], dtype=np.float64)
        X_norm = scaler.transform(imp.transform(X_raw))
        return X_norm / np.sqrt(max(1, len(cols)))

    def retrieve_for_request(
        self,
        request: PosterRequest | dict[str, Any],
        top_k: int = 5,
        diversify: bool = True,
        active_groups: tuple[str, ...] = ("canvas", "inventory", "capacity", "structure"),
        exclude_poster_id: int | None = None,
    ) -> pd.DataFrame:
        """Retrieve the top-k most suitable reference posters for a user content request."""
        if self.train_posters_ is None or self.train_occ_grids_ is None:
            raise RuntimeError("LayoutRetriever is not fitted.")

        req_dict = (
            request.to_summary_dict()
            if isinstance(request, PosterRequest)
            else dict(request)
        )

        group_col_map = {
            "canvas": SHARED_CANVAS_COLS,
            "inventory": SHARED_INVENTORY_COLS,
            "capacity": SHARED_CAPACITY_COLS,
            "structure": SHARED_STRUCTURE_COLS,
        }

        n_train = len(self.train_posters_)
        total_dist_sq = np.zeros(n_train, dtype=np.float64)

        for g_name in active_groups:
            if g_name not in group_col_map:
                continue
            w = float(self.group_weights.get(g_name, 1.0))
            cols = group_col_map[g_name]
            q_vec = self._transform_query_group(req_dict, g_name, cols)
            d_g = euclidean_distances(q_vec, self.train_group_matrices_[g_name]).ravel()
            total_dist_sq += w * (d_g ** 2)

        distances = np.sqrt(total_dist_sq)

        # Apply orientation constraint if requested (with fallback if fewer than top_k match)
        candidate_mask = np.ones(n_train, dtype=bool)
        if exclude_poster_id is not None:
            candidate_mask &= (self.train_posters_["poster_id"].to_numpy() != int(exclude_poster_id))

        if self.strict_orientation and "orientation" in req_dict:
            orient_mask = candidate_mask & (
                self.train_posters_["orientation"].to_numpy() == req_dict["orientation"]
            )
            if int(np.sum(orient_mask)) >= top_k:
                candidate_mask = orient_mask

        valid_indices = np.where(candidate_mask)[0]
        sub_dists = distances[valid_indices]

        # Shortlist top 30 candidates for compatibility scoring and MMR diversification
        pool_size = min(30, len(valid_indices))
        pool_order = np.argsort(sub_dists)[:pool_size]
        pool_indices = valid_indices[pool_order]

        # Compute explicit compatibility metrics on the shortlist
        compat_records = []
        for idx in pool_indices:
            cand_row = self.train_poster_dicts_[int(idx)]
            comp = compute_request_candidate_compatibility(req_dict, cand_row)
            comp["train_idx"] = int(idx)
            comp["feature_distance"] = float(distances[idx])
            # Combined relevance score blending group distance and explicit compatibility
            comp["relevance_score"] = float(
                0.55 * comp["compatibility_score"] + 0.45 * np.exp(-0.4 * distances[idx])
            )
            compat_records.append(comp)

        pool_df = pd.DataFrame(compat_records).sort_values(
            "relevance_score", ascending=False
        ).reset_index(drop=True)

        if not diversify or len(pool_df) <= top_k:
            chosen_df = pool_df.head(top_k).copy()
        else:
            # Maximal Marginal Relevance (MMR) diversification using 4x4 occupancy grids & column counts
            rel_scores = pool_df["relevance_score"].to_numpy()
            p_idxs = pool_df["train_idx"].to_numpy(dtype=int)
            pool_occ = self.train_occ_grids_[p_idxs]
            pool_cols = self.train_posters_.iloc[p_idxs]["estimated_column_count"].to_numpy()

            occ_sim = cosine_similarity(pool_occ)
            selected_pos: list[int] = [0]  # Always pick #1 most relevant candidate first

            while len(selected_pos) < top_k and len(selected_pos) < len(pool_df):
                best_pos = -1
                best_mmr = -np.inf
                for pos in range(len(pool_df)):
                    if pos in selected_pos:
                        continue
                    # Redundancy penalty: max spatial similarity to already selected templates
                    max_sim = max(
                        float(occ_sim[pos, s])
                        + (0.15 if pool_cols[pos] == pool_cols[s] else 0.0)
                        for s in selected_pos
                    )
                    mmr_val = self.mmr_lambda * rel_scores[pos] - (1.0 - self.mmr_lambda) * max_sim
                    if mmr_val > best_mmr:
                        best_mmr = mmr_val
                        best_pos = pos
                if best_pos < 0:
                    break
                selected_pos.append(best_pos)

            chosen_df = pool_df.iloc[selected_pos].copy().reset_index(drop=True)

        # Join candidate poster metadata
        chosen_meta = self.train_posters_.iloc[chosen_df["train_idx"].to_numpy()].reset_index(
            drop=True
        )
        result = pd.concat(
            [
                chosen_meta[
                    [
                        "poster_id",
                        "orientation",
                        "aspect_ratio",
                        "estimated_column_count",
                        "known_element_count",
                        "count_title",
                        "count_section",
                        "count_text",
                        "count_list",
                        "count_figure",
                        "count_table",
                        "count_caption",
                        "visual_area_sum",
                        "textual_area_sum",
                        "union_box_coverage",
                    ]
                    + (["archetype_cluster"] if "archetype_cluster" in chosen_meta.columns else [])
                ],
                chosen_df[
                    [
                        "relevance_score",
                        "compatibility_score",
                        "feature_distance",
                        "orientation_match",
                        "weighted_l1_count_mismatch",
                        "shortfall_element_count",
                        "surplus_element_count",
                        "column_count_diff",
                    ]
                ],
            ],
            axis=1,
        )
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result


def evaluate_retrieval_comprehensive(
    retriever: LayoutRetriever,
    eval_posters_df: pd.DataFrame,
    eval_elements_df: pd.DataFrame,
    top_k: int = 5,
    max_queries: int | None = None,
    diversify: bool = True,
    active_groups: tuple[str, ...] = ("canvas", "inventory", "capacity", "structure"),
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate retrieval quality on simulated user specifications built from held-out posters.

    For each held-out poster:
      1. Hides its bounding box positions and spatial grid.
      2. Constructs a `PosterRequest` from its content inventory and approximate block sizes.
      3. Retrieves top-k training layouts.
      4. Compares retrieved layouts against the held-out poster's ground-truth geometry.
    """
    if retriever.train_posters_ is None or retriever.train_occ_grids_ is None:
        raise RuntimeError("Retriever must be fitted before evaluation.")

    el_groups = {
        int(pid): grp for pid, grp in eval_elements_df.groupby("poster_id", sort=False)
    }
    occ_cols = [f"occ_grid_r{r}_c{c}" for r in range(4) for c in range(4)]
    inv_cols = [f"count_{s}" for s in KNOWN_CATEGORY_SLUGS]

    eval_sub = eval_posters_df.reset_index(drop=True)
    if max_queries is not None and len(eval_sub) > max_queries:
        rng = np.random.default_rng(random_state)
        pick = rng.choice(len(eval_sub), size=max_queries, replace=False)
        eval_sub = eval_sub.iloc[pick].reset_index(drop=True)

    train_occ_mat = retriever.train_posters_[occ_cols].to_numpy(dtype=float)
    train_inv_mat = retriever.train_posters_[inv_cols].to_numpy(dtype=float)
    train_cap_mat = retriever.train_posters_[["visual_area_sum", "textual_area_sum"]].to_numpy(dtype=float)
    pid_to_train_idx = {
        int(pid): idx
        for idx, pid in enumerate(retriever.train_posters_["poster_id"].to_numpy())
    }

    orient_acc_list: list[float] = []
    col_exact_top1_list: list[float] = []
    col_mae_topk_list: list[float] = []
    cat_l1_topk_list: list[float] = []
    inv_cos_topk_list: list[float] = []
    cap_l1_topk_list: list[float] = []
    spatial_occ_sim_top1_list: list[float] = []
    spatial_occ_sim_topk_list: list[float] = []
    diversity_topk_list: list[float] = []

    q_occ_all = eval_sub[occ_cols].to_numpy(dtype=float)
    q_inv_all = eval_sub[inv_cols].to_numpy(dtype=float)
    q_cap_all = eval_sub[["visual_area_sum", "textual_area_sum"]].to_numpy(dtype=float)

    for row_i, (_, q_row) in enumerate(eval_sub.iterrows()):
        qid = int(q_row["poster_id"])
        q_els = el_groups.get(qid)
        if q_els is None or len(q_els) == 0:
            continue

        sim_req = simulate_request_from_heldout_poster(
            q_row, q_els, random_state=random_state
        )
        retrieved = retriever.retrieve_for_request(
            sim_req,
            top_k=top_k,
            diversify=diversify,
            active_groups=active_groups,
            exclude_poster_id=qid,
        )
        ret_ids = retrieved["poster_id"].to_numpy(dtype=int)
        t_idxs = np.array([pid_to_train_idx[pid] for pid in ret_ids], dtype=int)

        # 1. Orientation match @ k
        orient_acc_list.append(float(np.mean(retrieved["orientation_match"].to_numpy())))

        # 2. Column structure match
        true_col = float(q_row["estimated_column_count"])
        ret_cols = retrieved["estimated_column_count"].to_numpy(dtype=float)
        col_exact_top1_list.append(float(ret_cols[0] == true_col))
        col_mae_topk_list.append(float(np.mean(np.abs(ret_cols - true_col))))

        # 3. Category inventory L1 & Cosine similarity
        q_inv = q_inv_all[row_i : row_i + 1]
        r_inv = train_inv_mat[t_idxs]
        cat_l1_topk_list.append(float(np.mean(np.sum(np.abs(r_inv - q_inv), axis=1))))
        inv_cos = cosine_similarity(q_inv, r_inv).ravel()
        inv_cos_topk_list.append(float(np.mean(inv_cos)))

        # 4. Capacity (Visual + Textual area) L1 error
        q_cap = q_cap_all[row_i : row_i + 1]
        r_cap = train_cap_mat[t_idxs]
        cap_l1_topk_list.append(float(np.mean(np.sum(np.abs(r_cap - q_cap), axis=1))))

        # 5. Spatial Arrangement Similarity (comparing HIDDEN ground-truth 4x4 grid to retrieved grids!)
        q_occ = q_occ_all[row_i : row_i + 1]
        r_occ = train_occ_mat[t_idxs]
        occ_sims = cosine_similarity(q_occ, r_occ).ravel()
        spatial_occ_sim_top1_list.append(float(occ_sims[0]))
        spatial_occ_sim_topk_list.append(float(np.mean(occ_sims)))

        # 6. Top-k Candidate Structural Diversity (1 - mean pairwise cosine sim of retrieved 4x4 grids)
        if len(r_occ) >= 2:
            pair_sim = cosine_similarity(r_occ)
            tri = np.triu_indices(len(r_occ), k=1)
            diversity_topk_list.append(float(1.0 - np.mean(pair_sim[tri])))

    return {
        "n_queries": len(orient_acc_list),
        "top_k": top_k,
        "active_groups": "+".join(active_groups),
        "diversify": diversify,
        "orientation_match_at_k": round(float(np.mean(orient_acc_list)), 4),
        "column_exact_match_at_1": round(float(np.mean(col_exact_top1_list)), 4),
        "column_mae_at_k": round(float(np.mean(col_mae_topk_list)), 4),
        "category_l1_error_at_k": round(float(np.mean(cat_l1_topk_list)), 4),
        "inventory_cosine_sim_at_k": round(float(np.mean(inv_cos_topk_list)), 4),
        "capacity_area_l1_at_k": round(float(np.mean(cap_l1_topk_list)), 4),
        "spatial_occ_sim_at_1": round(float(np.mean(spatial_occ_sim_top1_list)), 4),
        "spatial_occ_sim_at_k": round(float(np.mean(spatial_occ_sim_topk_list)), 4),
        "top_k_spatial_diversity": round(float(np.mean(diversity_topk_list)), 4),
    }
