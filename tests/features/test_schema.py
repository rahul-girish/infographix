"""Unit tests for src.features.schema."""

from __future__ import annotations

import unittest

import pandas as pd

from src.features.schema import NON_FEATURE_POSTER_COLS, generate_feature_schema_metadata


class TestSchema(unittest.TestCase):
    def test_generate_feature_schema_metadata(self):
        posters_df = pd.DataFrame(
            [
                {
                    "poster_id": 1,
                    "split": "train",
                    "source_split": "test",
                    "canvas_width_px": 1000.0,
                    "canvas_height_px": 1400.0,
                    "orientation": "portrait",
                    "aspect_ratio": 1000.0 / 1400.0,
                    "log_aspect_ratio": -0.336,
                    "is_landscape": 0.0,
                    "is_portrait": 1.0,
                    "total_element_count": 3,
                    "known_element_count": 3,
                    "count_title": 1,
                    "union_box_coverage": 0.45,
                    "occ_grid_r0_c0": 0.5,
                    "estimated_column_count": 2,
                }
            ]
        )
        elements_df = pd.DataFrame([{"poster_id": 1, "element_id": 10}])
        relations_df = pd.DataFrame([{"poster_id": 1, "source_element_id": 10, "target_element_id": 11}])

        meta = generate_feature_schema_metadata(posters_df, elements_df, relations_df)

        self.assertEqual(meta["schema_version"], "2.0.0")
        self.assertEqual(meta["tables"]["posters"]["row_count"], 1)
        self.assertEqual(meta["tables"]["elements"]["row_count"], 1)
        self.assertEqual(meta["tables"]["relations"]["row_count"], 1)

        numeric_cols = set(meta["tables"]["posters"]["numeric_feature_columns"])
        self.assertTrue(numeric_cols.isdisjoint(NON_FEATURE_POSTER_COLS))
        self.assertIn("aspect_ratio", numeric_cols)
        self.assertIn("occ_grid_r0_c0", numeric_cols)


if __name__ == "__main__":
    unittest.main()
