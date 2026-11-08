"""Live end-to-end interactive demo script for Infographix ML Core."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.clustering.dbscan_clustering import SpatialBlockDBSCAN
from src.registry.model_registry import ModelRegistry
from src.schemas.coco import (
    CATEGORY_NAME_TO_ID,
    COCODataset,
    convert_canvas_to_coco,
    convert_extracted_document_to_coco,
    validate_coco_structure,
)


def run_live_demo():
    print("=" * 70)
    print("🚀 INFOGRAPHIX LIVE DEMO: END-TO-END PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------
    # STAGE 1: LOAD MODEL FROM PRODUCTION REGISTRY
    # -------------------------------------------------------------
    print("\n[STAGE 1] Loading Production Model from Model Registry...")
    registry = ModelRegistry()
    model, preprocessor, manifest = registry.load_production_model("poster_layout_clusterer")

    print(f"  ✓ Loaded Model: {manifest.model_name}")
    print(f"  ✓ Version:      {manifest.version} (Active Stage: {manifest.stage})")
    print(f"  ✓ Algorithm:    {manifest.algorithm}")
    print(f"  ✓ Train Metrics: Silhouette={manifest.metrics.get('train_silhouette'):.4f}, Noise Ratio={manifest.metrics.get('train_noise_ratio')*100:.1f}%")

    # -------------------------------------------------------------
    # STAGE 2: SIMULATE INGESTION & MICRO-DBSCAN SPATIAL GROUPING
    # -------------------------------------------------------------
    print("\n[STAGE 2] Ingesting Research Paper & Running Micro-DBSCAN...")
    print("  Simulating extracted raw text & figure coordinates from a 2-column paper:")

    raw_boxes = [
        # Column 1 text lines
        [100.0, 200.0, 450.0, 25.0],
        [100.0, 235.0, 450.0, 25.0],
        [100.0, 270.0, 450.0, 25.0],
        # Column 2 text lines
        [650.0, 200.0, 450.0, 25.0],
        [650.0, 235.0, 450.0, 25.0],
        [650.0, 270.0, 450.0, 25.0],
    ]
    figures = [[650.0, 350.0, 450.0, 300.0]]
    captions = [[650.0, 660.0, 450.0, 30.0]]

    # Run Micro-DBSCAN for reading columns
    spatial_grouper = SpatialBlockDBSCAN(horizontal_weight=1.0, vertical_weight=0.5, eps=60.0)
    column_clusters = spatial_grouper.cluster_boxes(raw_boxes)
    print(f"  ✓ Text Line Clustering (Detected Columns): {list(column_clusters)}")
    print(f"    -> Lines 0,1,2 clustered into Column {column_clusters[0]}")
    print(f"    -> Lines 3,4,5 clustered into Column {column_clusters[3]}")

    # Pair figure with caption
    pairs = spatial_grouper.pair_figures_with_captions(figures, captions, max_distance=150.0)
    print(f"  ✓ Spatial Figure-Caption Pairing: Figure 0 paired with Caption {pairs[0][1]}")

    # -------------------------------------------------------------
    # STAGE 3: CONVERT EXTRACTED DATA TO UNIVERSAL COCO JSON
    # -------------------------------------------------------------
    print("\n[STAGE 3] Serializing Document to Standard COCO JSON...")
    extracted_elements = [
        {"category_name": "title", "bbox": [100.0, 50.0, 1720.0, 100.0], "text_content": "Deep Learning for Scientific Layouts"},
        {"category_name": "author_info", "bbox": [100.0, 160.0, 1720.0, 35.0], "text_content": "A. Researcher et al. - Stanford University"},
        {"category_name": "section", "bbox": [100.0, 220.0, 450.0, 400.0], "text_content": "Methodology Column"},
        {"category_name": "figure", "bbox": [650.0, 220.0, 450.0, 300.0]},
        {"category_name": "caption", "bbox": [650.0, 530.0, 450.0, 30.0], "text_content": "Figure 1: Neural Architecture"},
        {"category_name": "text", "bbox": [1150.0, 220.0, 450.0, 400.0], "text_content": "Results & Discussion"},
    ]

    coco_dataset = convert_extracted_document_to_coco("paper_project_42", extracted_elements, 1920, 1080)
    coco_json = coco_dataset.to_dict()
    is_valid, msg = validate_coco_structure(coco_json)

    print(f"  ✓ COCO Schema Valid: {is_valid} ({msg})")
    print(f"  ✓ COCO Images:      {len(coco_json['images'])} (Canvas: 1920x1080)")
    print(f"  ✓ COCO Annotations: {len(coco_json['annotations'])} element bounding boxes")

    # -------------------------------------------------------------
    # STAGE 4: MACRO CLUSTERING & TEMPLATE RECOMMENDATION
    # -------------------------------------------------------------
    print("\n[STAGE 4] Predicting Layout Archetype on Test Dataset...")
    test_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features" / "test_features.parquet")
    feature_cols = manifest.feature_schema_columns
    X_test_raw = test_df[feature_cols].to_numpy()

    # Run inference on real test samples using the loaded production pipeline
    sample_indices = [0, 15, 42, 88]
    X_samples = X_test_raw[sample_indices]
    X_scaled = preprocessor.transform(X_samples)
    predictions = model.predict(X_scaled)

    for idx, sample_id in enumerate(sample_indices):
        pred_cluster = predictions[idx]
        poster_id = test_df.iloc[sample_id].get("poster_id", f"sample_{sample_id}")
        elements = test_df.iloc[sample_id].get("element_count", 0)
        status = "Noise/Outlier (Excluded)" if pred_cluster == -1 else f"Layout Archetype {pred_cluster} (Recommended)"
        print(f"  Poster #{poster_id} ({elements} elements): -> Predicted: {status}")

    # -------------------------------------------------------------
    # STAGE 5: SIMULATING CANVAS EDIT & RE-EXPORT
    # -------------------------------------------------------------
    print("\n[STAGE 5] Simulating Frontend Canvas Interaction (Move & Resize)...")
    # User drags title and enlarges figure on canvas
    extracted_elements[0]["bbox"] = [120.0, 60.0, 1680.0, 110.0]  # Moved title
    canvas_coco = convert_canvas_to_coco("paper_project_42", extracted_elements, 1920, 1080)
    print("  ✓ Canvas updated and saved directly back to COCO JSON.")
    print(f"  ✓ Updated Title BBox: {canvas_coco['annotations'][0]['bbox']}")

    print("\n" + "=" * 70)
    print("🎉 LIVE DEMO COMPLETED: ALL ML CORE MODULES OPERATIONAL!")
    print("=" * 70)


if __name__ == "__main__":
    run_live_demo()
