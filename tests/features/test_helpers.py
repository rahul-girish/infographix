"""Unit tests for src.features.helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.helpers import (
    assign_pooled_splits,
    category_slug,
    classify_aspect_orientation,
    compute_orientation,
)


class TestHelpers(unittest.TestCase):
    def test_category_slug(self):
        self.assertEqual(category_slug("Author Info"), "author_info")
        self.assertEqual(category_slug("Title"), "title")
        self.assertEqual(category_slug("  ---  "), "unnamed")

    def test_orientation_classification(self):
        self.assertEqual(classify_aspect_orientation(0.75), "portrait")
        self.assertEqual(classify_aspect_orientation(1.40), "landscape")
        self.assertEqual(classify_aspect_orientation(1.00), "square")

        widths = np.array([800.0, 1400.0, 1000.0])
        heights = np.array([1200.0, 1000.0, 1000.0])
        np.testing.assert_array_equal(
            compute_orientation(widths, heights),
            np.array(["portrait", "landscape", "square"]),
        )

    def test_assign_pooled_splits_stratified_and_deterministic(self):
        rows = []
        for idx in range(40):
            w, h = (800, 1200) if idx < 20 else (1200, 800)
            rows.append({"id": 1000 - idx, "width": w, "height": h, "source_split": "train"})
        df = pd.DataFrame(rows)

        out1 = assign_pooled_splits(df, split_ratios=(0.70, 0.15, 0.15), random_state=42)
        out2 = assign_pooled_splits(df.sample(frac=1.0, random_state=7), split_ratios=(0.70, 0.15, 0.15), random_state=42)

        pd.testing.assert_frame_equal(out1[["id", "split"]], out2[["id", "split"]])
        self.assertEqual(out1["split"].value_counts().to_dict(), {"train": 28, "dev": 6, "test": 6})

        # Each orientation stratum (20 items) should split 14 / 3 / 3
        orientations = compute_orientation(out1["width"].to_numpy(float), out1["height"].to_numpy(float))
        ct = pd.crosstab(orientations, out1["split"])
        self.assertTrue((ct["train"] == 14).all())
        self.assertTrue((ct["dev"] == 3).all())
        self.assertTrue((ct["test"] == 3).all())

    def test_assign_pooled_splits_invalid_ratios(self):
        df = pd.DataFrame([{"id": 1, "width": 100, "height": 200}])
        with self.assertRaises(ValueError):
            assign_pooled_splits(df, split_ratios=(0.8, 0.2, 0.1))
        with self.assertRaises(ValueError):
            assign_pooled_splits(df, split_ratios=(0.8, 0.2, 0.0))


if __name__ == "__main__":
    unittest.main()
