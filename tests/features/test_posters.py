"""Unit tests for src.features.posters."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.posters import build_posters_table


class TestPosters(unittest.TestCase):
    def test_build_posters_table_retains_degenerate_and_populated_posters(self):
        images_by_split = {
            "train": pd.DataFrame(
                [
                    {"id": 1, "width": 1000, "height": 1500, "split": "train", "source_split": "train"},
                    {"id": 2, "width": 1600, "height": 1000, "split": "train", "source_split": "dev"},
                ]
            ),
            "dev": pd.DataFrame(columns=["id", "width", "height", "split", "source_split"]),
            "test": pd.DataFrame(columns=["id", "width", "height", "split", "source_split"]),
        }

        elements_df = pd.DataFrame(
            [
                {
                    "poster_id": 1,
                    "element_id": 101,
                    "split": "train",
                    "category_id": 487601213,
                    "category_slug": "title",
                    "is_unknown": False,
                    "x0": 0.1,
                    "y0": 0.05,
                    "x1": 0.9,
                    "y1": 0.15,
                    "w_norm": 0.8,
                    "h_norm": 0.1,
                    "cx": 0.5,
                    "cy": 0.1,
                    "area_norm": 0.08,
                    "log_area_norm": float(np.log(0.08 + 1e-6)),
                },
                {
                    "poster_id": 1,
                    "element_id": 102,
                    "split": "train",
                    "category_id": 2871247081,
                    "category_slug": "text",
                    "is_unknown": False,
                    "x0": 0.1,
                    "y0": 0.2,
                    "x1": 0.45,
                    "y1": 0.8,
                    "w_norm": 0.35,
                    "h_norm": 0.6,
                    "cx": 0.275,
                    "cy": 0.5,
                    "area_norm": 0.21,
                    "log_area_norm": float(np.log(0.21 + 1e-6)),
                },
                {
                    "poster_id": 1,
                    "element_id": 103,
                    "split": "train",
                    "category_id": 3731383018,
                    "category_slug": "figure",
                    "is_unknown": False,
                    "x0": 0.55,
                    "y0": 0.2,
                    "x1": 0.9,
                    "y1": 0.8,
                    "w_norm": 0.35,
                    "h_norm": 0.6,
                    "cx": 0.725,
                    "cy": 0.5,
                    "area_norm": 0.21,
                    "log_area_norm": float(np.log(0.21 + 1e-6)),
                },
            ]
        )

        rel_stats = {1: {"overlap_pair_count": 0.0}}
        posters_df = build_posters_table(images_by_split, elements_df, rel_stats, grid_res=32)

        self.assertEqual(len(posters_df), 2)
        p1 = posters_df[posters_df["poster_id"] == 1].iloc[0]
        p2 = posters_df[posters_df["poster_id"] == 2].iloc[0]

        self.assertEqual(p1["orientation"], "portrait")
        self.assertEqual(p1["has_title"], 1.0)
        self.assertEqual(p1["has_table"], 0.0)
        self.assertTrue(np.isnan(p1["w_norm_p50_table"]))
        self.assertGreater(p1["union_box_coverage"], 0.0)
        self.assertLessEqual(p1["union_box_coverage"], 1.0)
        self.assertEqual(p1["estimated_column_count"], 2)

        # Degenerate poster 2 is retained with 0 elements and preserved source_split
        self.assertEqual(p2["orientation"], "landscape")
        self.assertEqual(p2["source_split"], "dev")
        self.assertEqual(p2["total_element_count"], 0)


if __name__ == "__main__":
    unittest.main()
