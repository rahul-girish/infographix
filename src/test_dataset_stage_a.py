"""Rigorous Anti-Leakage and Provenance Test Suite for Stage A Dataset Construction."""

import os
import unittest
from pathlib import Path

# Ensure writable matplotlib temp directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch

from src.models.layout_retrieval import LayoutRetriever
from src.neural_layout.dataset import (
    CONTENT_FEAT_DIM,
    TEMPLATE_FEAT_DIM,
    PosterLayoutDataset,
    build_synthetic_request_and_sample,
    collate_layout_batch,
    extract_intrinsic_image_aspect_ratios,
    extract_pdf_blocks_via_pdftotext,
)
from src.registry.model_registry import ModelRegistry


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDatasetStageA(unittest.TestCase):
    """Rigorous anti-leakage test suite verifying zero target spatial leakage into content features."""

    @classmethod
    def setUpClass(cls):
        features_dir = PROJECT_ROOT / "data" / "processed" / "features" / "v2"
        cls.posters_df = pd.read_parquet(features_dir / "posters.parquet")
        cls.elements_df = pd.read_parquet(features_dir / "elements.parquet")

        reg = ModelRegistry(registry_root=PROJECT_ROOT / "models" / "registry")
        cls.retriever_model, _, _ = reg.load_production_model("poster_layout_retriever")
        train_posters = cls.posters_df[cls.posters_df["split"] == "train"].copy().reset_index(drop=True)
        cls.retriever_model.fit(train_posters)

        cls.pdf_dir = PROJECT_ROOT / "data" / "interim" / "scipostlayout" / "poster" / "pdf"

    def test_pdf_extraction_and_intrinsic_images(self):
        """Test pdftotext text extraction and pdfimages intrinsic dimension extraction."""
        sample_pdf = self.pdf_dir / "train" / "10000.pdf"
        if sample_pdf.exists():
            blocks = extract_pdf_blocks_via_pdftotext(sample_pdf)
            self.assertGreater(len(blocks), 0)
            self.assertIn("words", blocks[0])
            self.assertGreater(blocks[0]["word_count"], 0)

            # Test intrinsic image extraction from PDF asset
            images = extract_intrinsic_image_aspect_ratios(sample_pdf)
            self.assertGreater(len(images), 0)
            self.assertIn("intrinsic_width", images[0])
            self.assertIn("aspect_ratio", images[0])
            self.assertGreater(images[0]["aspect_ratio"], 0.1)

    def test_anti_leakage_candidate_exclusion(self):
        """Verify conditioning template strictly excludes the target poster ID."""
        target_row = self.posters_df[self.posters_df["split"] == "train"].iloc[0]
        target_pid = int(target_row["poster_id"])

        sample = build_synthetic_request_and_sample(
            target_poster_row=target_row,
            target_elements_df=self.elements_df,
            retriever_model=self.retriever_model,
            pdf_base_dir=self.pdf_dir,
        )

        self.assertNotEqual(sample.template_poster_id, target_pid)
        self.assertEqual(sample.target_poster_id, target_pid)

    def test_perturbing_target_bboxes_does_not_affect_content_features(self):
        """Critical: Modifying target bounding box dimensions/areas MUST NOT leak into content features.

        Tests that without PDF extraction (fallback state), zero area-based or box-derived
        estimation is performed: word_count, char_count, and aspect_ratio remain invariant.
        """
        target_row = self.posters_df[self.posters_df["split"] == "train"].iloc[1]
        pid = int(target_row["poster_id"])

        # 1. Base sample without PDF extraction
        sample_orig = build_synthetic_request_and_sample(
            target_poster_row=target_row,
            target_elements_df=self.elements_df,
            retriever_model=self.retriever_model,
            pdf_base_dir=None,
        )
        total_items = len(sample_orig.content_items)
        feats_orig = np.stack([it.to_feature_vector(total_items) for it in sample_orig.content_items])

        # 2. Corrupt target bounding boxes in elements_df (e.g. scale coordinates and areas by 3x)
        corrupted_elements = self.elements_df.copy()
        mask = corrupted_elements["poster_id"] == pid
        corrupted_elements.loc[mask, "x0"] = (corrupted_elements.loc[mask, "x0"] * 0.2) + 0.1
        corrupted_elements.loc[mask, "y0"] = (corrupted_elements.loc[mask, "y0"] * 0.3) + 0.05
        corrupted_elements.loc[mask, "w_norm"] = corrupted_elements.loc[mask, "w_norm"] * 0.5
        corrupted_elements.loc[mask, "h_norm"] = corrupted_elements.loc[mask, "h_norm"] * 2.5
        corrupted_elements.loc[mask, "area_norm"] = corrupted_elements.loc[mask, "area_norm"] * 4.0

        sample_corrupt = build_synthetic_request_and_sample(
            target_poster_row=target_row,
            target_elements_df=corrupted_elements,
            retriever_model=self.retriever_model,
            pdf_base_dir=None,
        )
        feats_corrupt = np.stack([it.to_feature_vector(total_items) for it in sample_corrupt.content_items])

        # Ensure content input features are identical despite extreme target geometry distortion
        np.testing.assert_allclose(feats_orig, feats_corrupt, rtol=1e-5, atol=1e-5)
        # But target supervision boxes are indeed different
        self.assertFalse(np.allclose(sample_orig.target_boxes, sample_corrupt.target_boxes))

    def test_perturbing_target_columns_does_not_affect_content_features(self):
        """Critical: Modifying target column count/assignment MUST NOT change content input features."""
        target_row = self.posters_df[self.posters_df["split"] == "train"].iloc[2]
        pid = int(target_row["poster_id"])

        sample_orig = build_synthetic_request_and_sample(
            target_poster_row=target_row,
            target_elements_df=self.elements_df,
            retriever_model=self.retriever_model,
            pdf_base_dir=self.pdf_dir,
        )
        total_items = len(sample_orig.content_items)
        feats_orig = np.stack([it.to_feature_vector(total_items) for it in sample_orig.content_items])

        # Corrupt column count in poster row
        corrupt_row = target_row.copy()
        corrupt_row["estimated_column_count"] = 5 if target_row["estimated_column_count"] != 5 else 1

        sample_corrupt = build_synthetic_request_and_sample(
            target_poster_row=corrupt_row,
            target_elements_df=self.elements_df,
            retriever_model=self.retriever_model,
            pdf_base_dir=self.pdf_dir,
        )
        feats_corrupt = np.stack([it.to_feature_vector(total_items) for it in sample_corrupt.content_items])

        # Content features must remain strictly invariant
        np.testing.assert_allclose(feats_orig, feats_corrupt, rtol=1e-5, atol=1e-5)
        # But target column supervision changed
        self.assertFalse(np.array_equal(sample_orig.target_columns, sample_corrupt.target_columns))

    def test_figure_aspect_ratios_independent_of_annotation_boxes(self):
        """Critical: Stretching figure annotation bounding boxes MUST NOT alter figure aspect ratio feature."""
        # Find a poster containing a figure
        fig_posters = self.elements_df[self.elements_df["category_slug"] == "figure"]["poster_id"].unique()
        target_pid = int(fig_posters[0])
        target_row = self.posters_df[self.posters_df["poster_id"] == target_pid].iloc[0]

        sample_orig = build_synthetic_request_and_sample(
            target_poster_row=target_row,
            target_elements_df=self.elements_df,
            retriever_model=self.retriever_model,
            pdf_base_dir=self.pdf_dir,
        )

        fig_items_orig = [it for it in sample_orig.content_items if it.category_slug == "figure"]
        if fig_items_orig:
            orig_asp = fig_items_orig[0].aspect_ratio
            orig_has_asp = fig_items_orig[0].has_aspect_ratio

            # Corrupt the figure's annotation bounding box: stretch height by 10x
            corrupted_elements = self.elements_df.copy()
            f_mask = (corrupted_elements["poster_id"] == target_pid) & (corrupted_elements["category_slug"] == "figure")
            corrupted_elements.loc[f_mask, "h_norm"] = corrupted_elements.loc[f_mask, "h_norm"] * 10.0
            corrupted_elements.loc[f_mask, "w_norm"] = corrupted_elements.loc[f_mask, "w_norm"] * 0.1

            sample_corrupt = build_synthetic_request_and_sample(
                target_poster_row=target_row,
                target_elements_df=corrupted_elements,
                retriever_model=self.retriever_model,
                pdf_base_dir=self.pdf_dir,
            )
            fig_items_corrupt = [it for it in sample_corrupt.content_items if it.category_slug == "figure"]

            # Aspect ratio feature MUST be identical (comes from intrinsic asset, not annotation box!)
            self.assertEqual(fig_items_orig[0].aspect_ratio, fig_items_corrupt[0].aspect_ratio)
            self.assertEqual(fig_items_orig[0].has_aspect_ratio, fig_items_corrupt[0].has_aspect_ratio)

    def test_explicit_missingness_representation(self):
        """Verify that elements without text extraction have explicit missing-value flags."""
        target_row = self.posters_df[self.posters_df["split"] == "train"].iloc[5]
        # Simulate missing PDF directory
        sample = build_synthetic_request_and_sample(
            target_poster_row=target_row,
            target_elements_df=self.elements_df,
            retriever_model=self.retriever_model,
            pdf_base_dir=Path("/non_existent_pdf_dir"),
        )

        for item in sample.content_items:
            self.assertFalse(item.has_text_length)
            self.assertEqual(item.text_source, "unavailable")
            self.assertEqual(item.word_count, 0.0)
            self.assertEqual(item.char_count, 0.0)

            # Check feature vector encoding: Dims 9 and 10 must be 0.0, Dim 11 (has_txt) must be 0.0
            feat_vec = item.to_feature_vector(len(sample.content_items))
            self.assertEqual(feat_vec[9], 0.0)
            self.assertEqual(feat_vec[10], 0.0)
            self.assertEqual(feat_vec[11], 0.0)

    def test_input_sequence_order_is_decoupled_from_reading_order(self):
        """Verify that input sequence index is shuffled and not trivially equal to reading order."""
        target_row = self.posters_df[self.posters_df["split"] == "train"].iloc[3]
        sample = build_synthetic_request_and_sample(
            target_poster_row=target_row,
            target_elements_df=self.elements_df,
            retriever_model=self.retriever_model,
            pdf_base_dir=self.pdf_dir,
        )

        shuffled_indices = [it.shuffled_order_idx for it in sample.content_items]
        target_orders = sample.target_reading_orders.tolist()

        self.assertEqual(shuffled_indices, list(range(len(sample.content_items))))
        if len(sample.content_items) > 3:
            # Shuffled order must not be trivially identical to true reading order
            self.assertNotEqual(shuffled_indices, target_orders)

    def test_dataset_batch_collation_shapes(self):
        """Test PyTorch Dataset and batch collation tensor dimensions."""
        ds = PosterLayoutDataset(
            posters_df=self.posters_df,
            elements_df=self.elements_df,
            retriever_model=self.retriever_model,
            split="train",
            pdf_base_dir=self.pdf_dir,
            max_samples=4,
        )
        self.assertEqual(len(ds), 4)

        item = ds[0]
        self.assertEqual(item["content_feats"].shape[1], CONTENT_FEAT_DIM)
        self.assertEqual(item["template_feats"].shape[1], TEMPLATE_FEAT_DIM)
        self.assertEqual(item["target_boxes"].shape[1], 4)

        batch = [ds[0], ds[1], ds[2]]
        collated = collate_layout_batch(batch)

        self.assertEqual(collated["content_feats"].ndim, 3)
        self.assertEqual(collated["content_feats"].shape[2], CONTENT_FEAT_DIM)
        self.assertEqual(collated["template_feats"].shape[2], TEMPLATE_FEAT_DIM)
        self.assertEqual(collated["content_mask"].shape, collated["target_boxes"].shape[:2])
        self.assertEqual(collated["macro_targets"].shape[0], 3)


if __name__ == "__main__":
    unittest.main()
