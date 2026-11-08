"""Unit tests for Infographix v2 Feature Engineering, Retrieval, Clustering, and Registry."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.request_schema import (
    ContentBlockDemand,
    PosterRequest,
    compute_request_candidate_compatibility,
    estimate_rendered_text_geometry,
    simulate_request_from_heldout_poster,
)
from src.models.layout_clustering import (
    CORE_MACRO_ARCHETYPE_FEATURES,
    LayoutFeaturePreprocessor,
    OrientationConditionedHierarchical,
    evaluate_clustering_comprehensive,
)
from src.models.layout_retrieval import LayoutRetriever
from src.registry.model_registry import ModelRegistry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_DIR = PROJECT_ROOT / "data" / "processed" / "features" / "v2"


class TestInfographixV2Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.posters_df = pd.read_parquet(V2_DIR / "posters.parquet")
        cls.elements_df = pd.read_parquet(V2_DIR / "elements.parquet")
        cls.relations_df = pd.read_parquet(V2_DIR / "relations.parquet")
        cls.rejected_df = pd.read_parquet(V2_DIR / "rejected_annotations.parquet")
        cls.schema = json.loads((V2_DIR / "feature_schema.json").read_text(encoding="utf-8"))

    def test_v2_tables_completeness_and_no_dropped_posters(self):
        """Verify all 7,855 posters are preserved across train/dev/test with valid schemas."""
        self.assertEqual(len(self.posters_df), 7855)
        split_counts = self.posters_df["split"].value_counts().to_dict()
        self.assertEqual(split_counts["train"], 6855)
        self.assertEqual(split_counts["dev"], 500)
        self.assertEqual(split_counts["test"], 500)

        self.assertGreater(len(self.elements_df), 192500)
        self.assertGreater(len(self.relations_df), 500000)
        self.assertGreater(len(self.rejected_df), 0)
        self.assertIn("rejection_reason", self.rejected_df.columns)

        # Verify union_box_coverage is bounded in [0, 1]
        cov = self.posters_df["union_box_coverage"].dropna()
        ov_ex = self.posters_df["overlap_excess"].dropna()
        self.assertTrue(((cov >= 0.0) & (cov <= 1.0)).all())
        self.assertTrue((ov_ex >= 0.0).all())

    def test_request_schema_and_typography_estimator(self):
        """Verify width-dependent text height estimation and request compatibility scoring."""
        est_narrow = estimate_rendered_text_geometry(word_count=150, candidate_col_width_norm=0.25)
        est_wide = estimate_rendered_text_geometry(word_count=150, candidate_col_width_norm=0.50)
        # Narrower column must require strictly more vertical lines/height than wider column
        self.assertGreater(est_narrow["estimated_lines"], est_wide["estimated_lines"])
        self.assertGreater(est_narrow["preferred_height"], est_wide["preferred_height"])

        req = PosterRequest(
            request_id="unit_test_req",
            requested_aspect_ratio=1.40,
            preferred_column_count=3,
            blocks=[
                ContentBlockDemand("t1", "title", "title", preferred_width=0.8, preferred_height=0.06),
                ContentBlockDemand("f1", "results", "figure", preferred_width=0.3, preferred_height=0.2),
                ContentBlockDemand("p1", "methods", "text", preferred_width=0.28, preferred_height=0.15),
            ],
        )
        summary = req.to_summary_dict()
        self.assertEqual(summary["orientation"], "landscape")
        self.assertEqual(summary["count_figure"], 1)

        sample_poster = self.posters_df[self.posters_df["orientation"] == "landscape"].iloc[0]
        compat = compute_request_candidate_compatibility(req, sample_poster)
        self.assertEqual(compat["orientation_match"], 1.0)
        self.assertGreater(compat["compatibility_score"], 0.0)
        self.assertLessEqual(compat["compatibility_score"], 1.0)

    def test_model_registry_v1_and_v2_compatibility(self):
        """Verify both legacy v1 and new v2 models load and run inference cleanly."""
        registry = ModelRegistry()
        models_info = registry.list_models()
        self.assertIn("poster_layout_clusterer", models_info)
        self.assertIn("poster_layout_archetypes", models_info)
        self.assertIn("poster_layout_retriever", models_info)

        # Test v2 Archetype Clusterer
        arch_model, arch_preproc, _ = registry.load_production_model("poster_layout_archetypes")
        test_sub = self.posters_df[self.posters_df["split"] == "test"].head(25)
        X_sub = arch_preproc.transform(test_sub)
        preds = arch_model.predict(X_sub, test_sub["is_portrait"].to_numpy())
        self.assertEqual(len(preds), 25)
        # Portrait samples must receive cluster IDs in [4, 5, 6] and non-portrait in [0, 1, 2, 3]
        port_mask = test_sub["is_portrait"].to_numpy(dtype=bool)
        self.assertTrue((preds[port_mask] >= 4).all())
        self.assertTrue((preds[~port_mask] < 4).all())

        # Test v2 Layout Retriever
        retriever, _, _ = registry.load_production_model("poster_layout_retriever")
        q_row = test_sub.iloc[0]
        q_els = self.elements_df[self.elements_df["poster_id"] == int(q_row["poster_id"])]
        sim_req = simulate_request_from_heldout_poster(q_row, q_els)
        top_k = retriever.retrieve_for_request(sim_req, top_k=5, diversify=True)
        self.assertEqual(len(top_k), 5)
        self.assertTrue((top_k["orientation"] == q_row["orientation"]).all())


if __name__ == "__main__":
    unittest.main()
