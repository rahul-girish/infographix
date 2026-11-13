"""DBSCAN clustering implementations for Infographix.

Includes:
1. PosterLayoutDBSCAN: Macro-level poster layout clustering with PCA,
   automated k-distance elbow heuristic, and inductive out-of-sample prediction.
2. SpatialBlockDBSCAN: Micro-level 2D spatial clustering for grouping bounding
   boxes into columns, sections, and pairing figures with captions.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


class PosterLayoutDBSCAN:
    """Macro-level poster layout clustering with PCA and inductive prediction.

    Solves the curse of dimensionality on 92 features by applying PCA first,
    then discovers dense layout archetypes while isolating abnormal posters
    as noise (label = -1).
    """

    def __init__(
        self,
        pca_components: int | float = 15,
        eps: float | None = None,
        min_samples: int = 15,
        metric: str = "euclidean",
        random_state: int = 42,
    ):
        self.pca_components = pca_components
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.random_state = random_state

        self.pca_: PCA | None = None
        self.dbscan_: DBSCAN | None = None
        self.core_points_: np.ndarray | None = None
        self.core_labels_: np.ndarray | None = None
        self.core_nn_: NearestNeighbors | None = None
        self.train_labels_: np.ndarray | None = None

    def find_elbow_eps(
        self,
        X_pca: np.ndarray,
        k: int | None = None,
        quantile: float = 0.85,
    ) -> float:
        """Estimate candidate eps using the k-distance distribution."""
        k = k or self.min_samples
        k_eff = min(k, max(1, len(X_pca) - 1))
        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=self.metric).fit(X_pca)
        distances, _ = nn.kneighbors(X_pca)
        k_distances = np.sort(distances[:, -1])
        # Return the specified quantile as a balanced density threshold
        return float(np.quantile(k_distances, quantile))

    def fit(self, X: np.ndarray) -> PosterLayoutDBSCAN:
        """Fit PCA and DBSCAN on training feature matrix."""
        # 1. Fit PCA if requested
        if self.pca_components is not None:
            n_components = min(self.pca_components, X.shape[1], X.shape[0])
            self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
            X_reduced = self.pca_.fit_transform(X)
        else:
            self.pca_ = None
            X_reduced = X

        # 2. Determine eps if not provided
        if self.eps is None:
            self.eps = self.find_elbow_eps(X_reduced, self.min_samples)

        # 3. Fit DBSCAN
        self.dbscan_ = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
        )
        self.train_labels_ = self.dbscan_.fit_predict(X_reduced)

        # 4. Prepare inductive nearest-core-sample assigner
        core_indices = self.dbscan_.core_sample_indices_
        if len(core_indices) > 0:
            self.core_points_ = X_reduced[core_indices]
            self.core_labels_ = self.train_labels_[core_indices]
            self.core_nn_ = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(
                self.core_points_
            )
        else:
            self.core_points_ = None
            self.core_labels_ = None
            self.core_nn_ = None

        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Inductive prediction: assign new samples to the nearest core cluster if within eps."""
        if self.dbscan_ is None:
            raise RuntimeError("PosterLayoutDBSCAN is not fitted yet.")

        X_reduced = self.pca_.transform(X_new) if self.pca_ is not None else X_new

        if self.core_nn_ is None or self.core_labels_ is None:
            # If no core points exist, everything is noise
            return np.full(len(X_reduced), -1, dtype=int)

        distances, indices = self.core_nn_.kneighbors(X_reduced)
        distances = distances.ravel()
        nearest_labels = self.core_labels_[indices.ravel()]

        # Within eps -> assigned to core cluster; beyond eps -> noise (-1)
        assigned = np.where(distances <= self.eps, nearest_labels, -1)
        return assigned

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.train_labels_

    def evaluate_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        sample_size: int = 5000,
    ) -> dict[str, Any]:
        """Compute comprehensive clustering metrics, handling noise points properly."""
        labels = np.asarray(labels)
        n_samples = len(labels)
        is_noise = labels == -1
        n_noise = int(np.sum(is_noise))
        noise_ratio = float(n_noise / n_samples) if n_samples > 0 else 0.0

        unique_non_noise = sorted(list(set(labels) - {-1}))
        n_clusters = len(unique_non_noise)

        cluster_counts = pd.Series(labels[~is_noise]).value_counts().to_dict()

        metrics: dict[str, Any] = {
            "n_samples": n_samples,
            "clusters_present": n_clusters,
            "noise_count": n_noise,
            "noise_ratio": noise_ratio,
            "cluster_sizes": {int(k): int(v) for k, v in cluster_counts.items()},
            "smallest_cluster_size": int(min(cluster_counts.values())) if cluster_counts else 0,
            "largest_cluster_size": int(max(cluster_counts.values())) if cluster_counts else 0,
            "silhouette_clean": None,
            "silhouette_with_noise": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }

        X_eval = self.pca_.transform(X) if self.pca_ is not None else X

        # Calculate metrics on valid clustered data (excluding noise)
        if n_clusters >= 2 and (n_samples - n_noise) > n_clusters:
            X_clean = X_eval[~is_noise]
            labels_clean = labels[~is_noise]
            sil_sample = min(sample_size, len(labels_clean)) if sample_size else None
            metrics["silhouette_clean"] = float(
                silhouette_score(
                    X_clean,
                    labels_clean,
                    sample_size=sil_sample,
                    random_state=self.random_state,
                )
            )
            metrics["davies_bouldin"] = float(davies_bouldin_score(X_clean, labels_clean))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_clean, labels_clean))

        # Also calculate silhouette treating noise as its own cluster (for full-space comparison)
        if len(np.unique(labels)) >= 2:
            sil_sample = min(sample_size, len(labels)) if sample_size else None
            metrics["silhouette_with_noise"] = float(
                silhouette_score(
                    X_eval,
                    labels,
                    sample_size=sil_sample,
                    random_state=self.random_state,
                )
            )

        return metrics


class SpatialBlockDBSCAN:
    """Micro-level 2D spatial clustering for raw document bounding boxes.

    Groups fragmented word/line boxes into columns or paragraphs, and
    pairs figure boxes with adjacent captions.
    """

    def __init__(
        self,
        horizontal_weight: float = 1.0,
        vertical_weight: float = 0.5,
        eps: float = 35.0,
        min_samples: int = 1,
    ):
        self.horizontal_weight = horizontal_weight
        self.vertical_weight = vertical_weight
        self.eps = eps
        self.min_samples = min_samples

    def cluster_boxes(
        self,
        bboxes: list[list[float]] | np.ndarray,
    ) -> np.ndarray:
        """Cluster [x, y, w, h] boxes into spatial groups (e.g. columns/paragraphs)."""
        boxes = np.asarray(bboxes)
        if len(boxes) == 0:
            return np.array([], dtype=int)

        # Compute center coordinates: center_x, center_y
        centers = np.column_stack([
            boxes[:, 0] + boxes[:, 2] / 2.0,
            boxes[:, 1] + boxes[:, 3] / 2.0,
        ])

        # Apply anisotropic scaling: vertical proximity vs horizontal proximity
        scaled_coords = np.column_stack([
            centers[:, 0] * self.horizontal_weight,
            centers[:, 1] * self.vertical_weight,
        ])

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean")
        labels = db.fit_predict(scaled_coords)
        return labels

    def pair_figures_with_captions(
        self,
        figure_boxes: list[list[float]] | np.ndarray,
        caption_boxes: list[list[float]] | np.ndarray,
        max_distance: float = 120.0,
    ) -> list[tuple[int, int | None]]:
        """Pair each figure box with the closest caption box within max_distance."""
        figs = np.asarray(figure_boxes)
        caps = np.asarray(caption_boxes)

        if len(figs) == 0:
            return []
        if len(caps) == 0:
            return [(i, None) for i in range(len(figs))]

        # Figure centers and Caption centers
        fig_centers = np.column_stack([
            figs[:, 0] + figs[:, 2] / 2.0,
            figs[:, 1] + figs[:, 3] / 2.0,
        ])
        cap_centers = np.column_stack([
            caps[:, 0] + caps[:, 2] / 2.0,
            caps[:, 1] + caps[:, 3] / 2.0,
        ])

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(cap_centers)
        distances, indices = nn.kneighbors(fig_centers)

        pairs: list[tuple[int, int | None]] = []
        for fig_idx, (dist, cap_idx) in enumerate(zip(distances.ravel(), indices.ravel())):
            if dist <= max_distance:
                pairs.append((fig_idx, int(cap_idx)))
            else:
                pairs.append((fig_idx, None))

        return pairs
