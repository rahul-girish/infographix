"""Reconsidered Layout Clustering Models & Structural Evaluation for Infographix (v2.0).

Implements and benchmarks:
  1. LayoutFeaturePreprocessor: Leak-free train-only imputation, group-aware scaling,
     and interpretable PCA diagnostics (variance explained & feature loadings).
  2. OrientationConditionedHierarchical: Separates Portrait and Landscape/Square
     canvases first, then applies Ward-linkage hierarchical clustering with
     inductive centroid/medoid assignment.
  3. LayoutGMMClusterer: Gaussian Mixture Model with BIC/AIC model selection,
     soft probability assignments, and outlier log-likelihood thresholding.
  4. LayoutHDBSCANClusterer: Varying-density HDBSCAN with inductive nearest-core
     assignment.
  5. Comprehensive Evaluation (`evaluate_clustering_comprehensive`): Measures both
     mathematical cluster separation AND domain-specific structural quality
     (orientation purity, column-count purity, category inventory similarity,
     4x4 spatial occupancy grid consistency, and cluster size entropy).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.features.pipeline_v2 import EPS, KNOWN_CATEGORY_SLUGS

# Curated, non-redundant macro layout features for archetype discovery
# Avoids sparse category-absent quantiles so rare categories do not hijack clusters
CORE_MACRO_ARCHETYPE_FEATURES = [
    # 1. Canvas & Orientation (4)
    "log_aspect_ratio",
    "is_landscape",
    "is_portrait",
    "known_element_count",
    # 2. Structural Grid, Columns & Symmetry (10)
    "estimated_column_count",
    "column_assignment_confidence",
    "estimated_gutter_fraction",
    "estimated_row_band_count",
    "full_width_region_count",
    "multi_column_span_count",
    "horiz_profile_symmetry",
    "vert_profile_symmetry",
    "left_right_area_balance",
    "top_bottom_area_balance",
    # 3. Content Inventory & Capacity Balance (12)
    "visual_element_count",
    "textual_element_count",
    "heading_element_count",
    "media_to_text_count_ratio",
    "visual_area_sum",
    "textual_area_sum",
    "header_area_sum",
    "visual_to_body_area_ratio",
    "union_box_coverage",
    "unannotated_whitespace_fraction",
    "area_norm_p50",
    "area_norm_iqr",
    # 4. Alignment & Regional Occupancy (8)
    "frac_aligned_left",
    "frac_aligned_top",
    "frac_aligned_cx",
    "text_occ_top_half",
    "text_occ_bottom_half",
    "visual_occ_left_half",
    "visual_occ_right_half",
    "visual_occ_center",
]


class LayoutFeaturePreprocessor:
    """Leak-free feature preprocessor fitted strictly on the training split."""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        scaler_type: str = "standard",
        clip_z: float | None = 3.5,
        pca_components: int | float | None = None,
        structural_weights: dict[str, float] | None = None,
        random_state: int = 42,
    ):
        self.feature_columns = list(feature_columns or CORE_MACRO_ARCHETYPE_FEATURES)
        self.scaler_type = scaler_type
        self.clip_z = clip_z
        self.pca_components = pca_components
        self.structural_weights = structural_weights or {}
        self.random_state = random_state

        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | RobustScaler | None = None
        self.pca_: PCA | None = None
        self.weight_vector_: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame) -> LayoutFeaturePreprocessor:
        X_raw = train_df[self.feature_columns].to_numpy(dtype=np.float64)
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X_raw)

        if self.scaler_type == "robust":
            self.scaler_ = RobustScaler()
        else:
            self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imp)
        if self.clip_z is not None:
            X_scaled = np.clip(X_scaled, -self.clip_z, self.clip_z)

        # Apply explicit structural feature weights if configured
        w = np.ones(len(self.feature_columns), dtype=np.float64)
        for col, mult in self.structural_weights.items():
            if col in self.feature_columns:
                w[self.feature_columns.index(col)] = float(mult)
        self.weight_vector_ = w
        X_weighted = X_scaled * self.weight_vector_

        if self.pca_components is not None:
            self.pca_ = PCA(
                n_components=self.pca_components, random_state=self.random_state
            )
            self.pca_.fit(X_weighted)
        else:
            self.pca_ = None

        return self

    def transform(self, df: pd.DataFrame, apply_pca: bool = True) -> np.ndarray:
        if self.imputer_ is None or self.scaler_ is None or self.weight_vector_ is None:
            raise RuntimeError("LayoutFeaturePreprocessor is not fitted.")
        X_raw = df[self.feature_columns].to_numpy(dtype=np.float64)
        X_imp = self.imputer_.transform(X_raw)
        X_scaled = self.scaler_.transform(X_imp)
        if self.clip_z is not None:
            X_scaled = np.clip(X_scaled, -self.clip_z, self.clip_z)
        X_weighted = X_scaled * self.weight_vector_
        if apply_pca and self.pca_ is not None:
            return self.pca_.transform(X_weighted)
        return X_weighted

    def fit_transform(self, train_df: pd.DataFrame, apply_pca: bool = True) -> np.ndarray:
        self.fit(train_df)
        return self.transform(train_df, apply_pca=apply_pca)

    def pca_diagnostics(self, top_n_features: int = 5) -> pd.DataFrame:
        """Return interpretable diagnostics explaining what each PCA component captures."""
        if self.pca_ is None:
            return pd.DataFrame()

        rows = []
        cum_var = np.cumsum(self.pca_.explained_variance_ratio_)
        for idx, (var_r, c_var, comp) in enumerate(
            zip(self.pca_.explained_variance_ratio_, cum_var, self.pca_.components_)
        ):
            top_idx = np.argsort(np.abs(comp))[::-1][:top_n_features]
            loadings_str = ", ".join(
                f"{self.feature_columns[i]} ({comp[i]:+.2f})" for i in top_idx
            )
            rows.append(
                {
                    "component": f"PC{idx + 1}",
                    "explained_variance_ratio": round(float(var_r), 4),
                    "cumulative_variance_ratio": round(float(c_var), 4),
                    "top_feature_loadings": loadings_str,
                }
            )
        return pd.DataFrame(rows)


class OrientationConditionedHierarchical:
    """Two-Stage Orientation-Conditioned Ward Hierarchical Clusterer.

    Stage 1: Hard partition by poster orientation (Portrait vs Landscape/Square).
             Ensures a portrait poster is never grouped into a landscape template.
    Stage 2: Ward-linkage Agglomerative Clustering within each orientation regime.
             Provides inductive prediction via nearest cluster centroid in the
             matching orientation partition.
    """

    def __init__(
        self,
        n_clusters_landscape: int = 4,
        n_clusters_portrait: int = 3,
        linkage: str = "ward",
    ):
        self.n_clusters_landscape = n_clusters_landscape
        self.n_clusters_portrait = n_clusters_portrait
        self.linkage = linkage

        self.centroids_landscape_: np.ndarray | None = None
        self.centroids_portrait_: np.ndarray | None = None
        self.medoid_ids_landscape_: list[int] = []
        self.medoid_ids_portrait_: list[int] = []
        self.train_labels_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        is_portrait: np.ndarray,
        poster_ids: np.ndarray | None = None,
    ) -> OrientationConditionedHierarchical:
        port_mask = np.asarray(is_portrait, dtype=bool)
        land_mask = ~port_mask

        labels = np.full(len(X), -1, dtype=int)

        # 1. Fit Landscape / Square partition -> Clusters [0 .. n_land - 1]
        X_land = X[land_mask]
        if len(X_land) >= self.n_clusters_landscape:
            agg_l = AgglomerativeClustering(
                n_clusters=self.n_clusters_landscape, linkage=self.linkage
            )
            l_labs = agg_l.fit_predict(X_land)
            labels[land_mask] = l_labs
            self.centroids_landscape_ = np.vstack(
                [X_land[l_labs == c].mean(axis=0) for c in range(self.n_clusters_landscape)]
            )
            if poster_ids is not None:
                pids_l = poster_ids[land_mask]
                self.medoid_ids_landscape_ = []
                for c in range(self.n_clusters_landscape):
                    c_idx = np.where(l_labs == c)[0]
                    dists = np.linalg.norm(
                        X_land[c_idx] - self.centroids_landscape_[c], axis=1
                    )
                    self.medoid_ids_landscape_.append(int(pids_l[c_idx[np.argmin(dists)]]))

        # 2. Fit Portrait partition -> Clusters [n_land .. n_land + n_port - 1]
        X_port = X[port_mask]
        if len(X_port) >= self.n_clusters_portrait:
            agg_p = AgglomerativeClustering(
                n_clusters=self.n_clusters_portrait, linkage=self.linkage
            )
            p_labs = agg_p.fit_predict(X_port)
            labels[port_mask] = p_labs + self.n_clusters_landscape
            self.centroids_portrait_ = np.vstack(
                [X_port[p_labs == c].mean(axis=0) for c in range(self.n_clusters_portrait)]
            )
            if poster_ids is not None:
                pids_p = poster_ids[port_mask]
                self.medoid_ids_portrait_ = []
                for c in range(self.n_clusters_portrait):
                    c_idx = np.where(p_labs == c)[0]
                    dists = np.linalg.norm(
                        X_port[c_idx] - self.centroids_portrait_[c], axis=1
                    )
                    self.medoid_ids_portrait_.append(int(pids_p[c_idx[np.argmin(dists)]]))

        self.train_labels_ = labels
        return self

    def predict(self, X: np.ndarray, is_portrait: np.ndarray) -> np.ndarray:
        if self.centroids_landscape_ is None or self.centroids_portrait_ is None:
            raise RuntimeError("OrientationConditionedHierarchical is not fitted.")

        port_mask = np.asarray(is_portrait, dtype=bool)
        land_mask = ~port_mask
        preds = np.full(len(X), -1, dtype=int)

        if np.any(land_mask):
            dists_l = np.linalg.norm(
                X[land_mask][:, None, :] - self.centroids_landscape_[None, :, :], axis=2
            )
            preds[land_mask] = np.argmin(dists_l, axis=1)

        if np.any(port_mask):
            dists_p = np.linalg.norm(
                X[port_mask][:, None, :] - self.centroids_portrait_[None, :, :], axis=2
            )
            preds[port_mask] = np.argmin(dists_p, axis=1) + self.n_clusters_landscape

        return preds


class LayoutGMMClusterer:
    """Gaussian Mixture Model for soft layout archetype discovery with BIC/AIC selection."""

    def __init__(
        self,
        n_components: int = 6,
        covariance_type: str = "full",
        outlier_quantile: float | None = None,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.outlier_quantile = outlier_quantile
        self.random_state = random_state

        self.gmm_: GaussianMixture | None = None
        self.log_prob_thresh_: float | None = None
        self.bic_: float | None = None
        self.aic_: float | None = None

    def fit(self, X: np.ndarray) -> LayoutGMMClusterer:
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_init=5,
            reg_covar=1e-4,
            random_state=self.random_state,
        )
        self.gmm_.fit(X)
        self.bic_ = float(self.gmm_.bic(X))
        self.aic_ = float(self.gmm_.aic(X))

        if self.outlier_quantile is not None:
            log_probs = self.gmm_.score_samples(X)
            self.log_prob_thresh_ = float(np.quantile(log_probs, self.outlier_quantile))
        else:
            self.log_prob_thresh_ = None

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.gmm_ is None:
            raise RuntimeError("LayoutGMMClusterer is not fitted.")
        labels = self.gmm_.predict(X)
        if self.log_prob_thresh_ is not None:
            log_probs = self.gmm_.score_samples(X)
            labels = np.where(log_probs < self.log_prob_thresh_, -1, labels)
        return labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.gmm_ is None:
            raise RuntimeError("LayoutGMMClusterer is not fitted.")
        return self.gmm_.predict_proba(X)

    @staticmethod
    def search_bic_aic(
        X_train: np.ndarray,
        k_values: list[int] = [2, 3, 4, 5, 6, 7, 8, 10, 12],
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Evaluate GMM across component counts using BIC and AIC."""
        records = []
        for k in k_values:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                n_init=3,
                reg_covar=1e-4,
                random_state=random_state,
            ).fit(X_train)
            labs = gmm.predict(X_train)
            sil = float(
                silhouette_score(X_train, labs, sample_size=min(3000, len(X_train)), random_state=random_state)
            )
            counts = pd.Series(labs).value_counts()
            records.append(
                {
                    "n_components": k,
                    "bic": round(float(gmm.bic(X_train)), 1),
                    "aic": round(float(gmm.aic(X_train)), 1),
                    "train_silhouette": round(sil, 4),
                    "smallest_cluster": int(counts.min()),
                    "largest_cluster": int(counts.max()),
                    "largest_cluster_ratio": round(float(counts.max() / len(X_train)), 4),
                }
            )
        return pd.DataFrame(records)


class LayoutHDBSCANClusterer:
    """Varying-density HDBSCAN wrapper with inductive nearest-core prediction."""

    def __init__(
        self,
        min_cluster_size: int = 45,
        min_samples: int = 15,
        cluster_selection_method: str = "eom",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method

        self.hdbscan_: HDBSCAN | None = None
        self.train_labels_: np.ndarray | None = None
        self.nn_assigner_: NearestNeighbors | None = None
        self.clustered_points_: np.ndarray | None = None
        self.clustered_labels_: np.ndarray | None = None
        self.max_assign_dist_: float = np.inf

    def fit(self, X: np.ndarray) -> LayoutHDBSCANClusterer:
        self.hdbscan_ = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
        )
        self.train_labels_ = self.hdbscan_.fit_predict(X)
        valid_mask = self.train_labels_ != -1
        if np.any(valid_mask):
            self.clustered_points_ = X[valid_mask]
            self.clustered_labels_ = self.train_labels_[valid_mask]
            self.nn_assigner_ = NearestNeighbors(n_neighbors=3).fit(self.clustered_points_)
            dists, _ = self.nn_assigner_.kneighbors(self.clustered_points_)
            self.max_assign_dist_ = float(np.quantile(dists[:, -1], 0.95) * 1.35)
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        if self.nn_assigner_ is None or self.clustered_labels_ is None:
            return np.full(len(X_new), -1, dtype=int)
        dists, indices = self.nn_assigner_.kneighbors(X_new, n_neighbors=1)
        d = dists.ravel()
        nearest_labs = self.clustered_labels_[indices.ravel()]
        return np.where(d <= self.max_assign_dist_, nearest_labs, -1)


def evaluate_clustering_comprehensive(
    X: np.ndarray,
    labels: np.ndarray,
    posters_df: pd.DataFrame,
    sample_size: int = 3500,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate both mathematical clustering quality and domain-specific structural quality.

    Metrics computed:
      1. Clustering Quality:
         - silhouette_clean (excluding -1 noise)
         - davies_bouldin (lower is better)
         - calinski_harabasz (higher is better)
         - clusters_present, noise_ratio, smallest/largest cluster size, largest_cluster_ratio
         - cluster_size_entropy_norm (0=collapsed into 1 cluster, 1=perfectly balanced)
      2. Structural Quality (Domain-Specific for Infographix):
         - orientation_purity: Weighted fraction of posters matching cluster's dominant orientation
         - column_count_purity: Weighted fraction matching cluster's dominant column count
         - within_cluster_column_mae: Mean absolute deviation from cluster median column count
         - inventory_cosine_consistency: Mean within-cluster cosine similarity of category counts
         - occupancy_grid_consistency: Mean within-cluster cosine similarity of 4x4 occupancy grids
    """
    labels = np.asarray(labels, dtype=int)
    n_total = len(labels)
    is_noise = labels == -1
    n_noise = int(np.sum(is_noise))
    noise_ratio = float(n_noise / n_total) if n_total > 0 else 0.0

    valid_mask = ~is_noise
    valid_labels = labels[valid_mask]
    unique_clusters = sorted(list(set(valid_labels)))
    n_clusters = len(unique_clusters)

    counts = pd.Series(valid_labels).value_counts()
    n_valid = len(valid_labels)

    # Normalized Shannon entropy of cluster size distribution
    if n_clusters >= 2 and n_valid > 0:
        probs = counts.to_numpy(dtype=float) / n_valid
        entropy = float(-np.sum(probs * np.log(probs + EPS)) / np.log(n_clusters))
    else:
        entropy = 0.0

    metrics: dict[str, Any] = {
        "n_samples": n_total,
        "clusters_present": n_clusters,
        "noise_count": n_noise,
        "noise_ratio": round(noise_ratio, 4),
        "smallest_cluster": int(counts.min()) if n_clusters > 0 else 0,
        "largest_cluster": int(counts.max()) if n_clusters > 0 else 0,
        "largest_cluster_ratio": round(float(counts.max() / n_total), 4) if n_clusters > 0 else 0.0,
        "cluster_size_entropy_norm": round(entropy, 4),
        "silhouette_clean": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "orientation_purity": 0.0,
        "column_count_purity": 0.0,
        "within_cluster_column_mae": None,
        "inventory_cosine_consistency": 0.0,
        "occupancy_grid_consistency": 0.0,
    }

    if n_clusters >= 2 and n_valid > n_clusters:
        X_clean = X[valid_mask]
        sil_samp = min(sample_size, n_valid)
        metrics["silhouette_clean"] = round(
            float(silhouette_score(X_clean, valid_labels, sample_size=sil_samp, random_state=random_state)),
            4,
        )
        metrics["davies_bouldin"] = round(float(davies_bouldin_score(X_clean, valid_labels)), 4)
        metrics["calinski_harabasz"] = round(
            float(calinski_harabasz_score(X_clean, valid_labels)), 2
        )

    # Structural quality metrics on non-noise posters
    if n_clusters >= 1 and n_valid > 0:
        sub_df = posters_df.iloc[np.where(valid_mask)[0]].copy()
        sub_df["_cluster"] = valid_labels

        # 1. Orientation Purity
        orient_matches = 0
        # 2. Column Count Purity & MAE
        col_matches = 0
        col_abs_err = 0.0
        # 3. Inventory & 4x4 Occupancy Consistency
        inv_cols = [f"count_{s}" for s in KNOWN_CATEGORY_SLUGS]
        occ_cols = [f"occ_grid_r{r}_c{c}" for r in range(4) for c in range(4)]

        inv_sims: list[float] = []
        occ_sims: list[float] = []
        cluster_weights: list[int] = []

        rng = np.random.default_rng(random_state)

        for cid, c_grp in sub_df.groupby("_cluster"):
            c_len = len(c_grp)
            orient_mode_cnt = int(c_grp["orientation"].value_counts().iloc[0])
            orient_matches += orient_mode_cnt

            col_mode_cnt = int(c_grp["estimated_column_count"].value_counts().iloc[0])
            col_matches += col_mode_cnt
            med_col = float(c_grp["estimated_column_count"].median())
            col_abs_err += float(np.sum(np.abs(c_grp["estimated_column_count"] - med_col)))

            # Sample up to 200 posters per cluster for fast pairwise cosine consistency
            if c_len >= 2:
                idx_s = (
                    rng.choice(c_len, size=200, replace=False)
                    if c_len > 200
                    else np.arange(c_len)
                )
                inv_mat = c_grp[inv_cols].to_numpy(dtype=float)[idx_s]
                occ_mat = c_grp[occ_cols].to_numpy(dtype=float)[idx_s]

                inv_cs = cosine_similarity(inv_mat)
                occ_cs = cosine_similarity(occ_mat)
                tri = np.triu_indices(len(idx_s), k=1)
                inv_sims.append(float(np.mean(inv_cs[tri])))
                occ_sims.append(float(np.mean(occ_cs[tri])))
                cluster_weights.append(c_len)

        metrics["orientation_purity"] = round(float(orient_matches / n_valid), 4)
        metrics["column_count_purity"] = round(float(col_matches / n_valid), 4)
        metrics["within_cluster_column_mae"] = round(float(col_abs_err / n_valid), 4)
        if cluster_weights:
            metrics["inventory_cosine_consistency"] = round(
                float(np.average(inv_sims, weights=cluster_weights)), 4
            )
            metrics["occupancy_grid_consistency"] = round(
                float(np.average(occ_sims, weights=cluster_weights)), 4
            )

    return metrics
