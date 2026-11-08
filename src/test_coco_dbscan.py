"""Unit tests for COCO schemas and DBSCAN clustering modules."""

import json
from pathlib import Path

import numpy as np

from src.clustering.dbscan_clustering import PosterLayoutDBSCAN, SpatialBlockDBSCAN
from src.schemas.coco import (
    CATEGORY_NAME_TO_ID,
    COCOAnnotation,
    COCOCategory,
    COCODataset,
    COCOImage,
    convert_canvas_to_coco,
    convert_extracted_document_to_coco,
    validate_coco_structure,
)


def test_coco_data_models_and_validation():
    """Verify COCO data models create valid COCO JSON structures."""
    image = COCOImage(id=1, width=1920, height=1080, file_name="poster_1.png")
    cat = COCOCategory(id=1, name="title", supercategory="header")
    ann = COCOAnnotation(
        id=101,
        image_id=1,
        category_id=1,
        bbox=[100.0, 50.0, 800.0, 120.0],
        area=96000.0,
        text_content="Scientific Poster Title",
    )

    dataset = COCODataset(images=[image], annotations=[ann], categories=[cat])
    data_dict = dataset.to_dict()

    is_valid, msg = validate_coco_structure(data_dict)
    assert is_valid, f"COCO validation failed: {msg}"
    assert data_dict["images"][0]["width"] == 1920
    assert data_dict["annotations"][0]["bbox"] == [100.0, 50.0, 800.0, 120.0]


def test_convert_extracted_document_to_coco():
    """Verify converting extracted document sections into COCO dataset."""
    sections = [
        {"category_name": "title", "bbox": [100.0, 50.0, 1720.0, 100.0], "text_content": "Title"},
        {"category_name": "figure", "bbox": [100.0, 200.0, 500.0, 400.0]},
        {"category_name": "caption", "bbox": [100.0, 610.0, 500.0, 50.0], "text_content": "Figure 1"},
    ]
    coco = convert_extracted_document_to_coco(doc_id="test_doc", extracted_sections=sections)
    data = coco.to_dict()

    is_valid, msg = validate_coco_structure(data)
    assert is_valid
    assert len(data["annotations"]) == 3
    assert data["annotations"][0]["category_id"] == CATEGORY_NAME_TO_ID["title"]
    assert data["annotations"][1]["category_id"] == CATEGORY_NAME_TO_ID["figure"]


def test_canvas_to_coco_conversion():
    """Verify frontend canvas element dictionary converts to COCO format."""
    elements = [
        {"category_name": "section", "bbox": [50.0, 50.0, 400.0, 300.0]},
        {"category_name": "text", "bbox": [60.0, 100.0, 380.0, 240.0]},
    ]
    coco_dict = convert_canvas_to_coco("canvas_proj_1", elements, 1920, 1080)
    is_valid, _ = validate_coco_structure(coco_dict)
    assert is_valid
    assert coco_dict["images"][0]["id"] == "canvas_proj_1"


def test_poster_layout_dbscan_clustering():
    """Verify PosterLayoutDBSCAN with PCA, inductive prediction, and metrics."""
    np.random.seed(42)
    # Create two synthetic clusters and some outliers
    c1 = np.random.normal(loc=0.0, scale=0.5, size=(40, 20))
    c2 = np.random.normal(loc=5.0, scale=0.5, size=(40, 20))
    outliers = np.random.uniform(low=20.0, high=30.0, size=(5, 20))
    X_train = np.vstack([c1, c2, outliers])

    model = PosterLayoutDBSCAN(pca_components=5, eps=2.0, min_samples=5)
    train_labels = model.fit_predict(X_train)

    assert len(train_labels) == 85
    # Outliers should be detected as noise (-1)
    assert -1 in train_labels

    # Test out-of-sample inductive prediction
    test_points = np.vstack([
        np.random.normal(loc=0.0, scale=0.5, size=(2, 20)),
        np.random.normal(loc=5.0, scale=0.5, size=(2, 20)),
        np.array([[50.0] * 20]),  # Extreme outlier
    ])
    pred_labels = model.predict(test_points)
    assert len(pred_labels) == 5
    assert pred_labels[-1] == -1  # Extreme point is noise

    # Verify metric evaluation
    metrics = model.evaluate_clusters(X_train, train_labels)
    assert metrics["clusters_present"] >= 2
    assert metrics["noise_count"] >= 1
    assert metrics["silhouette_clean"] is not None


def test_spatial_block_dbscan_and_pairing():
    """Verify micro-level 2D spatial clustering and figure-caption pairing."""
    grouper = SpatialBlockDBSCAN(eps=50.0, min_samples=1)

    # 3 boxes close to each other, 1 box far away
    boxes = [
        [10.0, 10.0, 100.0, 20.0],
        [12.0, 35.0, 100.0, 20.0],
        [11.0, 60.0, 100.0, 20.0],
        [500.0, 500.0, 100.0, 20.0],
    ]
    labels = grouper.cluster_boxes(boxes)
    assert len(labels) == 4
    # The first 3 should share a cluster, the 4th should be in a separate cluster
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]

    # Test figure-caption pairing
    figs = [[100.0, 100.0, 300.0, 200.0]]  # center at (250, 200)
    caps = [
        [100.0, 310.0, 300.0, 30.0],   # center at (250, 325) -> distance 125
        [800.0, 800.0, 300.0, 30.0],   # far away
    ]
    pairs = grouper.pair_figures_with_captions(figs, caps, max_distance=150.0)
    assert len(pairs) == 1
    fig_idx, cap_idx = pairs[0]
    assert fig_idx == 0
    assert cap_idx == 0
