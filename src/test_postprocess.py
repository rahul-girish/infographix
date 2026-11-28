"""Unit tests for NeuralLayoutRefiner (structured column-grid post-processor)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.neural_layout.dataset import CATEGORY_TO_IDX, CONTENT_FEAT_DIM
from src.neural_layout.evaluator import LayoutEvaluator
from src.neural_layout.postprocess import NeuralLayoutRefiner, RefinerConfig


def _make_synthetic_batch(
    categories: list[str],
    num_cols_true: int = 3,
    seed: int = 42,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    M = len(categories)
    content_feats = torch.zeros((1, M, CONTENT_FEAT_DIM), dtype=torch.float32)
    target_boxes = torch.zeros((1, M, 4), dtype=torch.float32)
    target_cols = torch.zeros((1, M), dtype=torch.long)
    target_orders = torch.linspace(0.0, 1.0, steps=max(1, M)).unsqueeze(0)

    pred_boxes = torch.zeros((1, M, 4), dtype=torch.float32)
    pred_col_logits = torch.zeros((1, M, 5), dtype=torch.float32)
    pred_orders = torch.linspace(0.0, 1.0, steps=max(1, M)).unsqueeze(0)
    pred_macro = torch.full((1, 5), -2.0, dtype=torch.float32)
    pred_macro[0, num_cols_true - 1] = 4.0

    for i, cat in enumerate(categories):
        c_idx = CATEGORY_TO_IDX[cat]
        content_feats[0, i, c_idx] = 1.0
        content_feats[0, i, 12] = 1.6 if cat == "figure" else 1.0
        content_feats[0, i, 13] = 1.0 if cat == "figure" else 0.0
        content_feats[0, i, 14] = float(i // 4) / max(1, M // 4)
        content_feats[0, i, 15] = float(i) / max(1, M - 1)

        # Noisy, overlapping raw neural predictions
        pred_boxes[0, i] = torch.tensor(
            [
                float(rng.uniform(0.0, 0.7)),
                float(rng.uniform(0.0, 0.8)),
                float(rng.uniform(0.15, 0.4)),
                float(rng.uniform(0.04, 0.22)),
            ],
            dtype=torch.float32,
        )
        col_i = i % num_cols_true
        pred_col_logits[0, i, col_i] = 3.0
        target_cols[0, i] = col_i
        target_boxes[0, i] = torch.tensor([0.1, 0.1, 0.2, 0.1], dtype=torch.float32)

    predictions = {
        "pred_boxes": pred_boxes,
        "pred_column_logits": pred_col_logits,
        "pred_order_scores": pred_orders,
        "pred_macro_column_logits": pred_macro,
    }
    batch = {
        "content_feats": content_feats,
        "content_mask": torch.ones((1, M), dtype=torch.bool),
        "template_feats": torch.zeros((1, 1, 16), dtype=torch.float32),
        "template_mask": torch.ones((1, 1), dtype=torch.bool),
        "target_boxes": target_boxes,
        "target_columns": target_cols,
        "target_orders": target_orders,
        "macro_targets": torch.tensor([num_cols_true - 1], dtype=torch.long),
    }
    return predictions, batch


class TestNeuralLayoutRefiner(unittest.TestCase):
    def setUp(self) -> None:
        self.refiner = NeuralLayoutRefiner(RefinerConfig())
        self.evaluator = LayoutEvaluator()

    def test_zero_overlap_and_canvas_bounds_standard_poster(self) -> None:
        cats = (
            ["title", "author_info", "logo", "logo"]
            + ["section", "text", "figure", "caption", "list"] * 4
        )
        preds, batch = _make_synthetic_batch(cats, num_cols_true=3, seed=7)
        refined = self.refiner.refine_batch(preds, batch)
        metrics = self.evaluator.evaluate_batch(refined, batch)

        self.assertEqual(metrics["canvas_violation_rate"], 0.0)
        self.assertEqual(metrics["unintended_overlap_rate"], 0.0)
        self.assertEqual(metrics["pairwise_iou_mean"], 0.0)
        self.assertLess(metrics["aspect_ratio_log_error"], 0.02)

    def test_dense_poster_38_elements_zero_overlap(self) -> None:
        cats = ["title", "author_info", "logo", "logo", "logo"] + ["section", "text", "figure", "caption", "table", "list"] * 5 + ["text", "text", "text"]
        preds, batch = _make_synthetic_batch(cats, num_cols_true=4, seed=99)
        refined = self.refiner.refine_batch(preds, batch)
        metrics = self.evaluator.evaluate_batch(refined, batch)

        self.assertEqual(metrics["canvas_violation_rate"], 0.0)
        self.assertEqual(metrics["unintended_overlap_rate"], 0.0)
        self.assertEqual(metrics["pairwise_iou_mean"], 0.0)

    def test_single_and_empty_elements(self) -> None:
        preds, batch = _make_synthetic_batch(["title"], num_cols_true=2)
        refined = self.refiner.refine_batch(preds, batch)
        metrics = self.evaluator.evaluate_batch(refined, batch)
        self.assertEqual(metrics["canvas_violation_rate"], 0.0)
        self.assertEqual(metrics["unintended_overlap_rate"], 0.0)

        rb, rc, ro = self.refiner.refine_single(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 5), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((5,), dtype=np.float32),
            np.zeros((0, CONTENT_FEAT_DIM), dtype=np.float32),
        )
        self.assertEqual(rb.shape, (0, 4))
        self.assertEqual(rc.shape, (0,))
        self.assertEqual(ro.shape, (0,))


if __name__ == "__main__":
    unittest.main()
