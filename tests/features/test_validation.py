"""Unit tests for src.features.validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.features.validation import validate_and_extract_elements


class TestValidation(unittest.TestCase):
    def test_pool_resplit_clip_and_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            cats = [
                {"id": 487601213, "name": "Title"},
                {"id": 2871247081, "name": "Text"},
            ]

            ann_id = 1
            pid = 100
            for src_split, count in [("train", 10), ("dev", 5), ("test", 5)]:
                imgs, anns = [], []
                for i in range(count):
                    pid += 1
                    w, h = (800, 1200) if i % 2 == 0 else (1200, 800)
                    imgs.append({"id": pid, "width": w, "height": h, "file_name": f"{pid}.png"})
                    # Valid annotation with slight negative origin within tolerance (-0.5px -> clipped to 0.0)
                    anns.append(
                        {
                            "id": ann_id,
                            "image_id": pid,
                            "category_id": 487601213,
                            "bbox": [-0.5 if i == 0 and src_split == "train" else 20.0, 20.0, 200.0, 60.0],
                            "area": 12000.0,
                        }
                    )
                    ann_id += 1
                    anns.append(
                        {
                            "id": ann_id,
                            "image_id": pid,
                            "category_id": 2871247081,
                            "bbox": [40.0, 120.0, 300.0, 400.0],
                            "area": 120000.0,
                        }
                    )
                    ann_id += 1

                if src_split == "dev":
                    # Inject one out-of-bounds box and one zero-width box
                    anns.append(
                        {
                            "id": ann_id,
                            "image_id": pid,
                            "category_id": 2871247081,
                            "bbox": [10.0, 10.0, 5000.0, 100.0],
                            "area": 500000.0,
                        }
                    )
                    ann_id += 1
                    anns.append(
                        {
                            "id": ann_id,
                            "image_id": pid,
                            "category_id": 2871247081,
                            "bbox": [10.0, 10.0, 0.0, 100.0],
                            "area": 0.0,
                        }
                    )
                    ann_id += 1

                (raw_dir / f"{src_split}.json").write_text(
                    json.dumps({"images": imgs, "annotations": anns, "categories": cats}),
                    encoding="utf-8",
                )

            images_by_split, elements_df, rejected_df, summary = validate_and_extract_elements(raw_dir)

            split_counts = {k: len(v) for k, v in images_by_split.items()}
            self.assertEqual(split_counts, {"train": 14, "dev": 3, "test": 3})
            self.assertEqual(len(elements_df), 40)
            self.assertEqual(len(rejected_df), 2)
            self.assertEqual(
                summary["rejected_by_reason"],
                {"exceeds_canvas_bounds": 1, "non_positive_dimensions": 1},
            )
            self.assertIn("source_split", elements_df.columns)
            self.assertTrue((elements_df["raw_x"] >= 0.0).all())
            self.assertTrue((elements_df["ratio_area_to_train_cat_median"] > 0).all())


if __name__ == "__main__":
    unittest.main()
