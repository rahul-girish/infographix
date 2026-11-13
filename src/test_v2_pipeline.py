"""Unit tests for Infographix Feature Pipeline, Request Schema, Retrieval, and Registry."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.features.pipeline import run_feature_pipeline_v2
from src.features.request_schema import (
    ContentBlockDemand,
    PosterRequest,
    compute_request_candidate_compatibility,
    estimate_rendered_text_geometry,
    simulate_request_from_heldout_poster,
)
from src.registry.model_registry import ModelRegistry

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_features_dir() -> Path:
    for candidate in ("v2", "v1"):
        p = PROJECT_ROOT / "data" / "processed" / "features" / candidate
        if (p / "posters.parquet").exists():
            return p
    raise FileNotFoundError("Could not find posters.parquet under data/processed/features/{v2,v1}")


class TestInfographixPipelineAndRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        features_dir = _resolve_features_dir()
        cls.posters_df = pd.read_parquet(features_dir / "posters.parquet")
        cls.elements_df = pd.read_parquet(features_dir / "elements.parquet")
        cls.relations_df = pd.read_parquet(features_dir / "relations.parquet")
        cls.rejected_df = pd.read_parquet(features_dir / "rejected_annotations.parquet")
        cls.schema = json.loads((features_dir / "feature_schema.json").read_text(encoding="utf-8"))

    def test_tables_completeness_and_no_dropped_posters(self):
        """Verify all 7,855 posters are preserved across the pooled 70/15/15 train/dev/test splits."""
        self.assertEqual(len(self.posters_df), 7855)
        split_counts = self.posters_df["split"].value_counts().to_dict()
        self.assertEqual(split_counts["train"], 5498)
        self.assertEqual(split_counts["dev"], 1178)
        self.assertEqual(split_counts["test"], 1179)

        train_pids = set(self.posters_df.loc[self.posters_df["split"] == "train", "poster_id"])
        dev_pids = set(self.posters_df.loc[self.posters_df["split"] == "dev", "poster_id"])
        test_pids = set(self.posters_df.loc[self.posters_df["split"] == "test", "poster_id"])
        self.assertEqual(len(train_pids & dev_pids), 0)
        self.assertEqual(len(train_pids & test_pids), 0)
        self.assertEqual(len(dev_pids & test_pids), 0)

        self.assertGreater(len(self.elements_df), 192500)
        self.assertGreater(len(self.relations_df), 500000)
        self.assertGreater(len(self.rejected_df), 0)
        self.assertIn("rejection_reason", self.rejected_df.columns)

        cov = self.posters_df["union_box_coverage"].dropna()
        ov_ex = self.posters_df["overlap_excess"].dropna()
        self.assertTrue(((cov >= 0.0) & (cov <= 1.0)).all())
        self.assertTrue((ov_ex >= 0.0).all())

    def test_pooled_resplit_70_15_15_no_leakage(self):
        """Verify pooling source JSONs and resplitting 70/15/15 is deterministic and poster-disjoint."""
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "out" / "v2"
            raw.mkdir(parents=True)
            cats = [{"id": 487601213, "name": "Title"}, {"id": 2871247081, "name": "Text"}]
            aid, pid = 1, 100
            for src_split, count in [("train", 10), ("dev", 5), ("test", 5)]:
                imgs, anns = [], []
                for i in range(count):
                    pid += 1
                    w, h = (800, 1200) if i % 2 == 0 else (1200, 800)
                    imgs.append({"id": pid, "width": w, "height": h, "file_name": f"{pid}.png"})
                    anns.append({"id": aid, "image_id": pid, "category_id": 487601213, "bbox": [50, 50, 400, 80], "area": 32000})
                    aid += 1
                    anns.append({"id": aid, "image_id": pid, "category_id": 2871247081, "bbox": [50, 200, 300, 400], "area": 120000})
                    aid += 1
                (raw / f"{src_split}.json").write_text(
                    json.dumps({"images": imgs, "annotations": anns, "categories": cats}),
                    encoding="utf-8",
                )

            res = run_feature_pipeline_v2(raw, out, export_legacy=True)
            p_df = res["posters"]
            e_df = res["elements"]
            self.assertEqual(p_df["split"].value_counts().to_dict(), {"train": 14, "dev": 3, "test": 3})
            merged = e_df[["poster_id", "split"]].merge(
                p_df[["poster_id", "split"]], on="poster_id", suffixes=("_el", "_p")
            )
            self.assertTrue((merged["split_el"] == merged["split_p"]).all())

    def test_request_schema_and_typography_estimator(self):
        """Verify width-dependent text height estimation and request compatibility scoring."""
        est_narrow = estimate_rendered_text_geometry(word_count=150, candidate_col_width_norm=0.25)
        est_wide = estimate_rendered_text_geometry(word_count=150, candidate_col_width_norm=0.50)
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

    def test_layout_retriever_registry(self):
        """Verify the registered poster_layout_retriever loads and retrieves matching layouts."""
        registry = ModelRegistry()
        models_info = registry.list_models()
        self.assertIn("poster_layout_retriever", models_info)

        retriever, _, _ = registry.load_production_model("poster_layout_retriever")
        test_sub = self.posters_df[self.posters_df["split"] == "test"].head(25)
        q_row = test_sub.iloc[0]
        q_els = self.elements_df[self.elements_df["poster_id"] == int(q_row["poster_id"])]
        sim_req = simulate_request_from_heldout_poster(q_row, q_els)
        top_k = retriever.retrieve_for_request(sim_req, top_k=5, diversify=True)
        self.assertEqual(len(top_k), 5)
        self.assertTrue((top_k["orientation"] == q_row["orientation"]).all())


if __name__ == "__main__":
    unittest.main()
