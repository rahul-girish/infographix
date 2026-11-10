"""Benchmarking, Ablation, and Model Registry script for Infographix v2.0.

Executes:
  1. Group E visual appearance feature extraction on a stratified sample (`appearance_sample.parquet`).
  2. Comprehensive clustering & structural evaluation across K-Means, DBSCAN (v1),
     GMM (with BIC/AIC curve), HDBSCAN, and Orientation-Conditioned Ward Hierarchical.
  3. Retrieval evaluation & feature-group ablation on held-out dev and test splits
     using simulated user content requests (`PosterRequest` with hidden coordinates).
  4. Registration of `poster_layout_retriever` (v2.0.0) and `poster_layout_archetypes` (v2.0.0)
     in `ModelRegistry` alongside the preserved v1 `poster_layout_clusterer`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.clustering.dbscan_clustering import PosterLayoutDBSCAN
from src.features.visual_features import extract_visual_features_for_sample
from src.models.layout_clustering import (
    CORE_MACRO_ARCHETYPE_FEATURES,
    LayoutFeaturePreprocessor,
    LayoutGMMClusterer,
    LayoutHDBSCANClusterer,
    OrientationConditionedHierarchical,
    evaluate_clustering_comprehensive,
)
from src.models.layout_retrieval import (
    LayoutRetriever,
    evaluate_retrieval_comprehensive,
)
from src.registry.model_registry import ModelRegistry
from src.tracking.tracker import ExperimentTracker

V2_DIR = PROJECT_ROOT / "data" / "processed" / "features" / "v2"
LEGACY_DIR = PROJECT_ROOT / "data" / "processed" / "features"
PNG_ROOT = PROJECT_ROOT / "data" / "interim" / "scipostlayout" / "poster" / "png"
REPORT_DIR = PROJECT_ROOT / "reports" / "comparisons"


def get_v2_clustering_feature_cols(posters_df: pd.DataFrame) -> list[str]:
    """Return the curated, non-redundant macro layout features for archetype clustering."""
    return [c for c in CORE_MACRO_ARCHETYPE_FEATURES if c in posters_df.columns]


def run_v2_benchmarks_and_register():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    posters_df = pd.read_parquet(V2_DIR / "posters.parquet")
    elements_df = pd.read_parquet(V2_DIR / "elements.parquet")

    # 1. Extract Group E Visual Features Sample if not yet saved or empty
    app_path = V2_DIR / "appearance_sample.parquet"
    need_app = not app_path.exists()
    if not need_app:
        try:
            need_app = len(pd.read_parquet(app_path)) == 0
        except Exception:
            need_app = True
    if need_app:
        print("[1/5] Extracting Group E visual appearance features for stratified sample...")
        app_df = extract_visual_features_for_sample(
            PNG_ROOT, posters_df, elements_df, max_per_split=60, random_state=42
        )
        app_df.to_parquet(app_path, index=False)
        print(f"  Saved {len(app_df)} visual feature rows to {app_path}")

    train_p = posters_df[posters_df["split"] == "train"].reset_index(drop=True)
    dev_p = posters_df[posters_df["split"] == "dev"].reset_index(drop=True)
    test_p = posters_df[posters_df["split"] == "test"].reset_index(drop=True)

    dev_e = elements_df[elements_df["split"] == "dev"].reset_index(drop=True)
    test_e = elements_df[elements_df["split"] == "test"].reset_index(drop=True)

    # 2. Fit Leak-Free v2 Preprocessor on Train Split Only
    print("[2/5] Fitting leak-free v2 LayoutFeaturePreprocessor on train split...")
    v2_cols = get_v2_clustering_feature_cols(posters_df)
    structural_weights = {
        "log_aspect_ratio": 2.2,
        "is_landscape": 2.2,
        "is_portrait": 2.2,
        "estimated_column_count": 2.5,
        "visual_to_body_area_ratio": 2.0,
        "union_box_coverage": 1.6,
        "horiz_profile_symmetry": 1.4,
    }

    preproc_pca = LayoutFeaturePreprocessor(
        feature_columns=v2_cols,
        scaler_type="standard",
        clip_z=3.5,
        pca_components=12,
        structural_weights=structural_weights,
        random_state=42,
    )
    X_train_v2 = preproc_pca.fit_transform(train_p, apply_pca=True)
    X_dev_v2 = preproc_pca.transform(dev_p, apply_pca=True)
    X_test_v2 = preproc_pca.transform(test_p, apply_pca=True)

    pca_diag_df = preproc_pca.pca_diagnostics(top_n_features=4)
    pca_diag_df.to_csv(REPORT_DIR / "v2_pca_diagnostics.csv", index=False)

    # 3. Evaluate Clustering Models (Legacy v1 vs New v2 Models)
    print("[3/5] Benchmarking clustering algorithms on Train / Dev / Test...")
    comparison_rows = []

    # (a) Legacy v1 Models via Registered v1 Model
    reg = ModelRegistry()
    v1_dbscan, v1_preproc, v1_manifest = reg.load_production_model("poster_layout_clusterer")
    v1_cols = v1_manifest.feature_schema_columns

    train_v1_df = pd.read_parquet(LEGACY_DIR / "train_features.parquet")
    dev_v1_df = pd.read_parquet(LEGACY_DIR / "dev_features.parquet")
    test_v1_df = pd.read_parquet(LEGACY_DIR / "test_features.parquet")

    X_tr_v1 = v1_preproc.transform(train_v1_df[v1_cols].to_numpy())
    X_dv_v1 = v1_preproc.transform(dev_v1_df[v1_cols].to_numpy())
    X_te_v1 = v1_preproc.transform(test_v1_df[v1_cols].to_numpy())

    # Legacy K-Means (k=3)
    km_v1 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_tr_v1)
    m_km_v1_tr = evaluate_clustering_comprehensive(X_tr_v1, km_v1.labels_, train_p)
    m_km_v1_te = evaluate_clustering_comprehensive(X_te_v1, km_v1.predict(X_te_v1), test_p)
    comparison_rows.append({
        "model_name": "K-Means (k=3, v1 features)",
        "feature_set": "v1 (92 summary stats)",
        "train_silhouette": m_km_v1_tr["silhouette_clean"],
        "test_silhouette": m_km_v1_te["silhouette_clean"],
        "test_davies_bouldin": m_km_v1_te["davies_bouldin"],
        **{f"test_{k}": m_km_v1_te[k] for k in (
            "clusters_present", "noise_ratio", "largest_cluster_ratio",
            "cluster_size_entropy_norm", "orientation_purity", "column_count_purity",
            "within_cluster_column_mae", "inventory_cosine_consistency", "occupancy_grid_consistency"
        )},
    })

    # Legacy Production DBSCAN (v1.0.0)
    m_db_v1_tr = evaluate_clustering_comprehensive(
        v1_dbscan.pca_.transform(X_tr_v1), v1_dbscan.predict(X_tr_v1), train_p
    )
    m_db_v1_te = evaluate_clustering_comprehensive(
        v1_dbscan.pca_.transform(X_te_v1), v1_dbscan.predict(X_te_v1), test_p
    )
    comparison_rows.append({
        "model_name": "DBSCAN (PCA-15, v1.0.0 Prod)",
        "feature_set": "v1 (92 summary stats)",
        "train_silhouette": m_db_v1_tr["silhouette_clean"],
        "test_silhouette": m_db_v1_te["silhouette_clean"],
        "test_davies_bouldin": m_db_v1_te["davies_bouldin"],
        **{f"test_{k}": m_db_v1_te[k] for k in (
            "clusters_present", "noise_ratio", "largest_cluster_ratio",
            "cluster_size_entropy_norm", "orientation_purity", "column_count_purity",
            "within_cluster_column_mae", "inventory_cosine_consistency", "occupancy_grid_consistency"
        )},
    })

    # (b) HDBSCAN on v2 features
    hdb_v2 = LayoutHDBSCANClusterer(min_cluster_size=40, min_samples=12).fit(X_train_v2)
    m_hdb_tr = evaluate_clustering_comprehensive(X_train_v2, hdb_v2.train_labels_, train_p)
    m_hdb_te = evaluate_clustering_comprehensive(X_test_v2, hdb_v2.predict(X_test_v2), test_p)
    comparison_rows.append({
        "model_name": "HDBSCAN (Varying Density, v2)",
        "feature_set": "v2 (Structural + Occupancy)",
        "train_silhouette": m_hdb_tr["silhouette_clean"],
        "test_silhouette": m_hdb_te["silhouette_clean"],
        "test_davies_bouldin": m_hdb_te["davies_bouldin"],
        **{f"test_{k}": m_hdb_te[k] for k in (
            "clusters_present", "noise_ratio", "largest_cluster_ratio",
            "cluster_size_entropy_norm", "orientation_purity", "column_count_purity",
            "within_cluster_column_mae", "inventory_cosine_consistency", "occupancy_grid_consistency"
        )},
    })

    # (c) GMM with BIC/AIC search on v2 features
    gmm_search_df = LayoutGMMClusterer.search_bic_aic(
        X_train_v2, k_values=[2, 3, 4, 5, 6, 7, 8, 10], covariance_type="diag"
    )
    gmm_search_df.to_csv(REPORT_DIR / "v2_gmm_bic_aic_search.csv", index=False)

    gmm_v2 = LayoutGMMClusterer(
        n_components=7, covariance_type="diag", outlier_quantile=0.03, random_state=42
    ).fit(X_train_v2)
    m_gmm_tr = evaluate_clustering_comprehensive(X_train_v2, gmm_v2.predict(X_train_v2), train_p)
    m_gmm_te = evaluate_clustering_comprehensive(X_test_v2, gmm_v2.predict(X_test_v2), test_p)
    comparison_rows.append({
        "model_name": "GMM (k=7, 3% outlier filter, v2)",
        "feature_set": "v2 (Structural + Occupancy)",
        "train_silhouette": m_gmm_tr["silhouette_clean"],
        "test_silhouette": m_gmm_te["silhouette_clean"],
        "test_davies_bouldin": m_gmm_te["davies_bouldin"],
        **{f"test_{k}": m_gmm_te[k] for k in (
            "clusters_present", "noise_ratio", "largest_cluster_ratio",
            "cluster_size_entropy_norm", "orientation_purity", "column_count_purity",
            "within_cluster_column_mae", "inventory_cosine_consistency", "occupancy_grid_consistency"
        )},
    })

    # (d) Orientation-Conditioned Ward Hierarchical (4 Landscape + 3 Portrait = 7 archetypes)
    hier_v2 = OrientationConditionedHierarchical(
        n_clusters_landscape=4, n_clusters_portrait=3, linkage="ward"
    ).fit(
        X_train_v2,
        train_p["is_portrait"].to_numpy(),
        poster_ids=train_p["poster_id"].to_numpy(),
    )
    m_hier_tr = evaluate_clustering_comprehensive(X_train_v2, hier_v2.train_labels_, train_p)
    m_hier_dv = evaluate_clustering_comprehensive(
        X_dev_v2, hier_v2.predict(X_dev_v2, dev_p["is_portrait"].to_numpy()), dev_p
    )
    m_hier_te = evaluate_clustering_comprehensive(
        X_test_v2, hier_v2.predict(X_test_v2, test_p["is_portrait"].to_numpy()), test_p
    )
    comparison_rows.append({
        "model_name": "Hierarchical Ward (Orientation-Conditioned k=4L+3P, v2)",
        "feature_set": "v2 (Structural + Occupancy)",
        "train_silhouette": m_hier_tr["silhouette_clean"],
        "test_silhouette": m_hier_te["silhouette_clean"],
        "test_davies_bouldin": m_hier_te["davies_bouldin"],
        **{f"test_{k}": m_hier_te[k] for k in (
            "clusters_present", "noise_ratio", "largest_cluster_ratio",
            "cluster_size_entropy_norm", "orientation_purity", "column_count_purity",
            "within_cluster_column_mae", "inventory_cosine_consistency", "occupancy_grid_consistency"
        )},
    })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(REPORT_DIR / "v2_clustering_comparison.csv", index=False)
    print("\n=== CLUSTERING & STRUCTURAL QUALITY COMPARISON ===")
    print(comp_df.to_string(index=False))

    # 4. Evaluate Direct Layout Retrieval & Feature-Group Ablation on Test Set
    print("\n[4/5] Evaluating Direct Nearest-Neighbour Layout Retriever & Feature Ablation...")
    retriever = LayoutRetriever(mmr_lambda=0.78, strict_orientation=True).fit(
        train_p, cluster_labels=hier_v2.train_labels_
    )

    ablation_configs = [
        ("1. Canvas Only", ("canvas",), False),
        ("2. Canvas + Category Inventory", ("canvas", "inventory"), False),
        ("3. Canvas + Inventory + Area Capacity", ("canvas", "inventory", "capacity"), False),
        ("4. Full Structural (+Columns, No MMR)", ("canvas", "inventory", "capacity", "structure"), False),
        ("5. Full Structural + MMR Diversity (Recommended)", ("canvas", "inventory", "capacity", "structure"), True),
    ]

    ablation_rows = []
    for label, active_grps, use_mmr in ablation_configs:
        ret_metrics = evaluate_retrieval_comprehensive(
            retriever,
            test_p,
            test_e,
            top_k=5,
            diversify=use_mmr,
            active_groups=active_grps,
            random_state=42,
        )
        ret_metrics["configuration"] = label
        ablation_rows.append(ret_metrics)

    ablation_df = pd.DataFrame(ablation_rows)
    cols_order = [
        "configuration",
        "orientation_match_at_k",
        "column_exact_match_at_1",
        "column_mae_at_k",
        "category_l1_error_at_k",
        "inventory_cosine_sim_at_k",
        "capacity_area_l1_at_k",
        "spatial_occ_sim_at_1",
        "spatial_occ_sim_at_k",
        "top_k_spatial_diversity",
    ]
    ablation_df = ablation_df[cols_order]
    ablation_df.to_csv(REPORT_DIR / "v2_retrieval_ablation.csv", index=False)
    print("\n=== RETRIEVAL ABLATION ON HELD-OUT TEST POSTERS (N=500) ===")
    print(ablation_df.to_string(index=False))

    # 5. Track in Aim / ExperimentTracker and Register in ModelRegistry
    print("\n[5/5] Registering v2.0.0 models in ModelRegistry...")
    tracker = ExperimentTracker(
        experiment_name="infographix-v2-redesign",
        run_name="v2_hierarchical_and_retriever",
        output_dir=REPORT_DIR,
    )
    tracker.set_params({
        "schema_version": "2.0.0",
        "n_clustering_features": len(v2_cols),
        "pca_components": 20,
        "hierarchical_clusters": "4_landscape_3_portrait",
        "retriever_mmr_lambda": 0.78,
    })
    tracker.log_metrics(m_hier_te, prefix="hier_test_")
    tracker.log_metrics(ablation_rows[-1], prefix="retriever_test_")
    tracker.close()

    # Register Orientation-Conditioned Archetype Clusterer (v2.0.0)
    reg.register_model(
        model_name="poster_layout_archetypes",
        version="v2.0.0",
        model_object=hier_v2,
        preprocessor=preproc_pca,
        feature_schema_columns=v2_cols,
        hyperparameters={
            "n_clusters_landscape": 4,
            "n_clusters_portrait": 3,
            "linkage": "ward",
            "pca_components": 20,
        },
        metrics={
            "train_silhouette": m_hier_tr["silhouette_clean"],
            "dev_silhouette": m_hier_dv["silhouette_clean"],
            "test_silhouette": m_hier_te["silhouette_clean"],
            "test_orientation_purity": m_hier_te["orientation_purity"],
            "test_column_purity": m_hier_te["column_count_purity"],
            "test_cluster_size_entropy": m_hier_te["cluster_size_entropy_norm"],
            "test_occupancy_consistency": m_hier_te["occupancy_grid_consistency"],
        },
        stage="production",
        description="Orientation-conditioned Ward hierarchical archetype discovery (4 Landscape + 3 Portrait templates) on v2 structural features.",
    )

    # Register Direct Constraint-Aware Layout Retriever (v2.0.0)
    best_ret = ablation_rows[-1]
    reg.register_model(
        model_name="poster_layout_retriever",
        version="v2.0.0",
        model_object=retriever,
        preprocessor=None,
        feature_schema_columns=list(retriever.group_weights.keys()),
        hyperparameters={
            "group_weights": retriever.group_weights,
            "mmr_lambda": retriever.mmr_lambda,
            "strict_orientation": retriever.strict_orientation,
        },
        metrics=best_ret,
        stage="production",
        description="Direct constraint-aware, group-normalized nearest-neighbour layout retriever with MMR diversity reranking.",
    )

    full_summary = {
        "clustering_comparison": comparison_rows,
        "retrieval_ablation": ablation_rows,
        "gmm_bic_aic_search": gmm_search_df.to_dict(orient="records"),
        "pca_diagnostics": pca_diag_df.to_dict(orient="records"),
    }
    (REPORT_DIR / "v2_model_comparison.json").write_text(
        json.dumps(full_summary, indent=2), encoding="utf-8"
    )
    print("Successfully registered v2.0.0 models and saved comparison reports!")


if __name__ == "__main__":
    run_v2_benchmarks_and_register()
