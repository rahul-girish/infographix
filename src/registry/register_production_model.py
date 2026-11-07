"""Production registration script for Infographix layout clusterer.

Trains the validated DBSCAN (PCA-15) pipeline on training data, evaluates metrics
on dev/test splits, packages all artifacts, and registers version v1.0.0 as 'production'.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.clustering.comparison import load_and_preprocess_data
from src.clustering.dbscan_clustering import PosterLayoutDBSCAN
from src.registry.model_registry import ModelRegistry


def register_v1_production():
    print("--- Infographix: Registering Production Model v1.0.0 ---")

    # 1. Load data
    data_dir = PROJECT_ROOT / "data" / "processed" / "features"
    train_df = pd.read_parquet(data_dir / "train_features.parquet")
    dev_df = pd.read_parquet(data_dir / "dev_features.parquet")
    test_df = pd.read_parquet(data_dir / "test_features.parquet")

    excluded = {
        "poster_id", "image_id", "id", "file_name", "filename",
        "split", "pixel_width", "pixel_height", "width", "height",
        "image_width", "image_height",
    }
    feature_cols = [
        c for c in train_df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    # 2. Build preprocessor
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    X_train_raw = train_df[feature_cols].to_numpy()
    X_dev_raw = dev_df[feature_cols].to_numpy()
    X_test_raw = test_df[feature_cols].to_numpy()

    X_train = preprocessor.fit_transform(X_train_raw)
    X_dev = preprocessor.transform(X_dev_raw)
    X_test = preprocessor.transform(X_test_raw)

    # 3. Train validated DBSCAN + PCA-15 model
    hparams = {
        "pca_components": 15,
        "eps": 4.8,
        "min_samples": 15,
        "metric": "euclidean",
        "random_state": 42,
    }
    model = PosterLayoutDBSCAN(**hparams)
    train_labels = model.fit_predict(X_train)

    # 4. Evaluate metrics across all splits
    train_metrics = model.evaluate_clusters(X_train, train_labels)
    dev_labels = model.predict(X_dev)
    dev_metrics = model.evaluate_clusters(X_dev, dev_labels)
    test_labels = model.predict(X_test)
    test_metrics = model.evaluate_clusters(X_test, test_labels)

    consolidated_metrics = {
        "train_silhouette": train_metrics.get("silhouette_clean"),
        "dev_silhouette": dev_metrics.get("silhouette_clean"),
        "test_silhouette": test_metrics.get("silhouette_clean"),
        "train_noise_ratio": train_metrics.get("noise_ratio"),
        "dev_noise_ratio": dev_metrics.get("noise_ratio"),
        "test_noise_ratio": test_metrics.get("noise_ratio"),
        "train_clusters_present": train_metrics.get("clusters_present"),
        "davies_bouldin": train_metrics.get("davies_bouldin"),
        "calinski_harabasz": train_metrics.get("calinski_harabasz"),
    }

    print("\nModel Performance Metrics:")
    for k, v in consolidated_metrics.items():
        print(f"  {k}: {v}")

    # 5. Register in Model Registry
    registry = ModelRegistry()
    manifest = registry.register_model(
        model_name="poster_layout_clusterer",
        version="v1.0.0",
        model_object=model,
        preprocessor=preprocessor,
        feature_schema_columns=feature_cols,
        hyperparameters=hparams,
        metrics=consolidated_metrics,
        stage="production",
        description="Production DBSCAN (PCA-15) model for poster layout archetype recommendation and outlier isolation.",
        git_commit="3685363",
    )

    print("\nSuccessfully registered model in registry!")
    print(f"Model: {manifest.model_name} ({manifest.version})")
    print(f"Stage: {manifest.stage}")
    print(f"Registry Root: {registry.registry_root}")

    # 6. Verify loading
    loaded_model, loaded_prep, loaded_manifest = registry.load_production_model("poster_layout_clusterer")
    test_pred = loaded_model.predict(loaded_prep.transform(X_test_raw[:3]))
    print(f"\nVerification test inference on 3 samples: predicted cluster IDs = {test_pred}")

    return manifest


if __name__ == "__main__":
    register_v1_production()
