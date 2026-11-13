"""Unit tests for src.features.relations."""

from __future__ import annotations

import unittest

import pandas as pd

from src.features.relations import build_relations_and_spatial_summaries


class TestRelations(unittest.TestCase):
    def test_single_and_multi_element_relations(self):
        elements_df = pd.DataFrame(
            [
                # Poster 1: single element
                {
                    "poster_id": 1,
                    "element_id": 10,
                    "split": "train",
                    "category_slug": "title",
                    "x0": 0.1,
                    "y0": 0.05,
                    "x1": 0.9,
                    "y1": 0.15,
                    "w_norm": 0.8,
                    "h_norm": 0.1,
                    "cx": 0.5,
                    "cy": 0.1,
                    "area_norm": 0.08,
                },
                # Poster 2: figure + caption below + overlapping text
                {
                    "poster_id": 2,
                    "element_id": 20,
                    "split": "dev",
                    "category_slug": "figure",
                    "x0": 0.1,
                    "y0": 0.2,
                    "x1": 0.5,
                    "y1": 0.5,
                    "w_norm": 0.4,
                    "h_norm": 0.3,
                    "cx": 0.3,
                    "cy": 0.35,
                    "area_norm": 0.12,
                },
                {
                    "poster_id": 2,
                    "element_id": 21,
                    "split": "dev",
                    "category_slug": "caption",
                    "x0": 0.1,
                    "y0": 0.52,
                    "x1": 0.5,
                    "y1": 0.58,
                    "w_norm": 0.4,
                    "h_norm": 0.06,
                    "cx": 0.3,
                    "cy": 0.55,
                    "area_norm": 0.024,
                },
                {
                    "poster_id": 2,
                    "element_id": 22,
                    "split": "dev",
                    "category_slug": "text",
                    "x0": 0.55,
                    "y0": 0.2,
                    "x1": 0.90,
                    "y1": 0.5,
                    "w_norm": 0.35,
                    "h_norm": 0.3,
                    "cx": 0.725,
                    "cy": 0.35,
                    "area_norm": 0.105,
                },
            ]
        )

        relations_df, rel_stats = build_relations_and_spatial_summaries(elements_df)

        self.assertEqual(rel_stats[1]["overlap_pair_count"], 0.0)
        self.assertEqual(rel_stats[2]["fig_caption_paired_fraction"], 1.0)
        self.assertGreater(len(relations_df), 0)

        fig_to_cap = relations_df[
            (relations_df["poster_id"] == 2)
            & (relations_df["source_element_id"] == 20)
            & (relations_df["target_element_id"] == 21)
        ]
        self.assertEqual(len(fig_to_cap), 1)
        self.assertIn("figure_caption_hypothesis", fig_to_cap.iloc[0]["edge_types"])
        self.assertEqual(fig_to_cap.iloc[0]["direction"], "below")


if __name__ == "__main__":
    unittest.main()
