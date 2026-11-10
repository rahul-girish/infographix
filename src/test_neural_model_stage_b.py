"""Unit Tests for Stage B: Neural Architecture, Multi-Task Losses, and Training.

Tests:
  1. ContentEncoder with variable sequence lengths and attention masks.
  2. TemplateEncoder with variable template slot counts.
  3. LayoutDecoder output bounds and shapes.
  4. TemplateConditionedLayoutModel end-to-end forward and backward passes.
  5. MultiTaskLayoutLoss individual loss computations and masking.
  6. Boundary penalty sensitivity to canvas protrusions.
  7. Legitimate figure-caption overlap suppression.
  8. LayoutEvaluator metric computations and template comparison.
  9. Mini training smoke run with checkpointing and Aim initialization.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn.functional as F

from src.neural_layout.content_encoder import ContentEncoder
from src.neural_layout.template_encoder import TemplateEncoder
from src.neural_layout.layout_decoder import LayoutDecoder
from src.neural_layout.layout_model import TemplateConditionedLayoutModel
from src.neural_layout.losses import MultiTaskLayoutLoss, LossWeights, compute_giou
from src.neural_layout.evaluator import LayoutEvaluator
from src.neural_layout.dataset import CATEGORY_TO_IDX
from src.neural_layout.trainer import TrainingConfig, LayoutTrainer, set_seed


class TestStageBNeuralLayout(unittest.TestCase):
    """Test suite for neural architecture, losses, and training."""

    def setUp(self):
        set_seed(42)
        self.B = 2
        self.M = 6
        self.K = 5
        self.d_model = 64
        self.num_heads = 2
        self.ffn_dim = 128

        # Simulated batch
        self.content_feats = torch.randn(self.B, self.M, 18)
        self.content_feats[..., 0:9] = F.one_hot(torch.randint(0, 9, (self.B, self.M)), num_classes=9).float()
        self.content_feats[..., 11] = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]]).float()  # has_text_length
        self.content_feats[..., 13] = torch.tensor([[0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]]).float()  # has_aspect_ratio
        self.content_feats[..., 12] = 1.33  # aspect ratio

        self.template_feats = torch.randn(self.B, self.K, 16)
        self.template_feats[..., 0:9] = F.one_hot(torch.randint(0, 9, (self.B, self.K)), num_classes=9).float()
        self.template_feats[..., 9:13] = torch.sigmoid(torch.randn(self.B, self.K, 4))  # valid norm boxes

        # Attention masks: sample 0 has 5 valid items, sample 1 has 3 valid items
        self.content_mask = torch.tensor([
            [True, True, True, True, True, False],
            [True, True, True, False, False, False],
        ], dtype=torch.bool)

        self.template_mask = torch.tensor([
            [True, True, True, True, False],
            [True, True, True, False, False],
        ], dtype=torch.bool)

        # Targets
        self.target_boxes = torch.sigmoid(torch.randn(self.B, self.M, 4)) * 0.8 + 0.1
        self.target_columns = torch.randint(0, 3, (self.B, self.M))
        self.target_orders = torch.tensor([
            [0, 1, 2, 3, 4, 0],
            [0, 1, 2, 0, 0, 0],
        ], dtype=torch.long)
        self.macro_targets = torch.tensor([2, 1], dtype=torch.long)

        self.batch = {
            "content_feats": self.content_feats,
            "template_feats": self.template_feats,
            "content_mask": self.content_mask,
            "template_mask": self.template_mask,
            "target_boxes": self.target_boxes,
            "target_columns": self.target_columns,
            "target_orders": self.target_orders,
            "macro_targets": self.macro_targets,
        }

    def test_content_encoder_masking_and_shapes(self):
        """Test ContentEncoder forward pass and attention mask handling."""
        encoder = ContentEncoder(
            input_dim=18,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=2,
            ffn_dim=self.ffn_dim,
        )
        embeds, macro_embed = encoder(self.content_feats, self.content_mask)

        self.assertEqual(embeds.shape, (self.B, self.M, self.d_model))
        self.assertEqual(macro_embed.shape, (self.B, self.d_model))

        # Check padded element is zeroed out
        self.assertTrue(torch.all(embeds[0, 5] == 0.0))
        self.assertTrue(torch.all(embeds[1, 3] == 0.0))
        self.assertTrue(torch.all(embeds[1, 4] == 0.0))
        self.assertTrue(torch.all(embeds[1, 5] == 0.0))

        # Check valid elements are non-zero
        self.assertFalse(torch.all(embeds[0, 0] == 0.0))

    def test_template_encoder_masking_and_shapes(self):
        """Test TemplateEncoder forward pass and slot embedding shapes."""
        encoder = TemplateEncoder(
            input_dim=16,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=2,
            ffn_dim=self.ffn_dim,
        )
        embeds = encoder(self.template_feats, self.template_mask)
        self.assertEqual(embeds.shape, (self.B, self.K, self.d_model))
        self.assertTrue(torch.all(embeds[0, 4] == 0.0))
        self.assertFalse(torch.all(embeds[0, 0] == 0.0))

    def test_layout_decoder_outputs_and_bounds(self):
        """Test LayoutDecoder prediction heads and normalized bounding-box coordinates."""
        c_enc = ContentEncoder(input_dim=18, d_model=self.d_model, num_heads=self.num_heads, num_layers=2)
        t_enc = TemplateEncoder(input_dim=16, d_model=self.d_model, num_heads=self.num_heads, num_layers=2)
        decoder = LayoutDecoder(d_model=self.d_model, num_heads=self.num_heads, num_layers=2, num_columns=5)

        c_emb, macro = c_enc(self.content_feats, self.content_mask)
        t_emb = t_enc(self.template_feats, self.template_mask)

        preds = decoder(c_emb, t_emb, self.content_mask, self.template_mask, macro_embed=macro)

        pred_boxes = preds["pred_boxes"]
        self.assertEqual(pred_boxes.shape, (self.B, self.M, 4))
        # Coordinates must be strictly in (0, 1)
        self.assertTrue((pred_boxes >= 0.0).all() and (pred_boxes <= 1.0).all())

        pred_col = preds["pred_column_logits"]
        self.assertEqual(pred_col.shape, (self.B, self.M, 5))

        pred_orders = preds["pred_order_scores"]
        self.assertEqual(pred_orders.shape, (self.B, self.M))

        pred_macro = preds["pred_macro_column_logits"]
        self.assertEqual(pred_macro.shape, (self.B, 5))

    def test_end_to_end_forward_backward_pass(self):
        """Verify unified model end-to-end forward and backward gradient computation."""
        model = TemplateConditionedLayoutModel(
            content_dim=18,
            template_dim=16,
            d_model=self.d_model,
            num_heads=self.num_heads,
            encoder_layers=2,
            decoder_layers=2,
            ffn_dim=self.ffn_dim,
            num_columns=5,
        )
        preds = model(
            content_feats=self.content_feats,
            template_feats=self.template_feats,
            content_mask=self.content_mask,
            template_mask=self.template_mask,
        )

        loss_fn = MultiTaskLayoutLoss()
        total_loss, loss_dict = loss_fn(preds, self.batch)

        self.assertGreater(total_loss.item(), 0.0)
        self.assertFalse(torch.isnan(total_loss))

        # Backward pass
        total_loss.backward()

        # Check gradients exist and contain no NaNs
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_multi_task_loss_components(self):
        """Verify each of the 8 individual loss terms is tracked and logged."""
        model = TemplateConditionedLayoutModel(
            content_dim=18,
            template_dim=16,
            d_model=self.d_model,
            num_heads=self.num_heads,
            encoder_layers=2,
            decoder_layers=2,
        )
        preds = model(
            content_feats=self.content_feats,
            template_feats=self.template_feats,
            content_mask=self.content_mask,
            template_mask=self.template_mask,
        )

        loss_fn = MultiTaskLayoutLoss()
        total_loss, loss_dict = loss_fn(preds, self.batch)

        expected_keys = [
            "loss_total",
            "loss_box",
            "loss_giou",
            "loss_bound",
            "loss_overlap",
            "loss_col",
            "loss_aspect",
            "loss_order",
            "loss_macro",
        ]
        for k in expected_keys:
            self.assertIn(k, loss_dict)
            self.assertFalse(np.isnan(loss_dict[k]))
            self.assertGreaterEqual(loss_dict[k], 0.0)

    def test_boundary_violation_penalty_sensitivity(self):
        """Verify that canvas boundary penalty strongly increases when boxes overflow canvas."""
        loss_fn = MultiTaskLayoutLoss()

        # Valid in-canvas boxes
        preds_valid = {
            "pred_boxes": torch.tensor([[[0.1, 0.1, 0.4, 0.3], [0.5, 0.5, 0.3, 0.3]]]),
            "pred_column_logits": torch.zeros(1, 2, 5),
            "pred_order_scores": torch.zeros(1, 2),
            "pred_macro_column_logits": torch.zeros(1, 5),
        }
        # Out-of-bounds overflowing boxes
        preds_overflow = {
            "pred_boxes": torch.tensor([[[0.8, 0.8, 0.5, 0.6], [0.5, 0.5, 0.3, 0.3]]]),  # x0+w = 1.3, y0+h = 1.4
            "pred_column_logits": torch.zeros(1, 2, 5),
            "pred_order_scores": torch.zeros(1, 2),
            "pred_macro_column_logits": torch.zeros(1, 5),
        }
        dummy_batch = {
            "target_boxes": torch.tensor([[[0.1, 0.1, 0.4, 0.3], [0.5, 0.5, 0.3, 0.3]]]),
            "target_columns": torch.zeros(1, 2, dtype=torch.long),
            "target_orders": torch.zeros(1, 2, dtype=torch.long),
            "macro_targets": torch.zeros(1, dtype=torch.long),
            "content_mask": torch.tensor([[True, True]]),
            "content_feats": torch.zeros(1, 2, 18),
        }

        _, loss_dict_valid = loss_fn(preds_valid, dummy_batch)
        _, loss_dict_overflow = loss_fn(preds_overflow, dummy_batch)

        self.assertEqual(loss_dict_valid["loss_bound"], 0.0)
        self.assertGreater(loss_dict_overflow["loss_bound"], 0.3)

    def test_legitimate_figure_caption_overlap_suppression(self):
        """Verify that intentional figure-caption overlaps are suppressed in the overlap penalty."""
        loss_fn = MultiTaskLayoutLoss()
        fig_cat = CATEGORY_TO_IDX["figure"]
        cap_cat = CATEGORY_TO_IDX["caption"]
        text_cat = CATEGORY_TO_IDX["text"]

        # 2 overlapping boxes (50% overlapping)
        box_pair = torch.tensor([[[0.2, 0.2, 0.4, 0.4], [0.3, 0.3, 0.4, 0.4]]])

        # Case A: text + text overlap (disparate unintended collision)
        feats_text = torch.zeros(1, 2, 18)
        feats_text[0, 0, text_cat] = 1.0
        feats_text[0, 1, text_cat] = 1.0

        # Case B: figure + caption overlap (legitimate containment/nesting)
        feats_figcap = torch.zeros(1, 2, 18)
        feats_figcap[0, 0, fig_cat] = 1.0
        feats_figcap[0, 1, cap_cat] = 1.0

        batch_text = {
            "target_boxes": box_pair,
            "target_columns": torch.zeros(1, 2, dtype=torch.long),
            "target_orders": torch.zeros(1, 2, dtype=torch.long),
            "macro_targets": torch.zeros(1, dtype=torch.long),
            "content_mask": torch.tensor([[True, True]]),
            "content_feats": feats_text,
        }
        batch_figcap = {
            "target_boxes": box_pair,
            "target_columns": torch.zeros(1, 2, dtype=torch.long),
            "target_orders": torch.zeros(1, 2, dtype=torch.long),
            "macro_targets": torch.zeros(1, dtype=torch.long),
            "content_mask": torch.tensor([[True, True]]),
            "content_feats": feats_figcap,
        }
        preds = {
            "pred_boxes": box_pair,
            "pred_column_logits": torch.zeros(1, 2, 5),
            "pred_order_scores": torch.zeros(1, 2),
            "pred_macro_column_logits": torch.zeros(1, 5),
        }

        _, loss_dict_text = loss_fn(preds, batch_text)
        _, loss_dict_figcap = loss_fn(preds, batch_figcap)

        # The overlap penalty for text+text must be significantly higher than for figure+caption
        self.assertGreater(loss_dict_text["loss_overlap"], loss_dict_figcap["loss_overlap"] * 5.0)

    def test_evaluator_metrics(self):
        """Verify LayoutEvaluator calculates all multi-task metrics."""
        evaluator = LayoutEvaluator()
        preds = {
            "pred_boxes": self.target_boxes + 0.02,  # slight perturbation
            "pred_column_logits": F.one_hot(self.target_columns, num_classes=5).float() * 10.0,
            "pred_order_scores": self.target_orders.float(),
            "pred_macro_column_logits": F.one_hot(self.macro_targets, num_classes=5).float() * 10.0,
        }
        metrics = evaluator.evaluate_batch(preds, self.batch)

        self.assertIn("iou_mean", metrics)
        self.assertIn("giou_mean", metrics)
        self.assertIn("canvas_violation_rate", metrics)
        self.assertIn("unintended_overlap_rate", metrics)
        self.assertIn("column_accuracy", metrics)
        self.assertIn("reading_order_kendall_tau", metrics)
        self.assertIn("template_divergence_iou", metrics)

        # Column accuracy and order tau should be high on perfectly aligned signals
        self.assertAlmostEqual(metrics["column_accuracy"], 1.0, places=2)
        self.assertGreater(metrics["reading_order_kendall_tau"], 0.8)

    def test_trainer_with_clearml_tracking_and_checkpointing(self):
        """Verify LayoutTrainer logs config, all 8 losses, metrics to ClearML and saves checkpoints."""
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, sample, n=4):
                self.sample = sample
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                # Return single item with shape matching PosterLayoutDataset output
                return {
                    "content_feats": self.sample["content_feats"][0],
                    "template_feats": self.sample["template_feats"][0],
                    "target_boxes": self.sample["target_boxes"][0],
                    "target_columns": self.sample["target_columns"][0],
                    "target_orders": self.sample["target_orders"][0],
                    "macro_target": self.sample["macro_targets"][0],
                    "num_content": torch.tensor(self.sample["content_feats"].shape[1]),
                    "num_template": torch.tensor(self.sample["template_feats"].shape[1]),
                    "target_poster_id": torch.tensor(101),
                    "template_poster_id": torch.tensor(202),
                }

        train_ds = MockDataset(self.batch, n=4)
        val_ds = MockDataset(self.batch, n=2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_dir = Path(tmp_dir) / "checkpoints"

            mock_task = MagicMock()
            mock_task.id = "mock_task_001"
            mock_logger = MagicMock()
            mock_task.get_logger.return_value = mock_logger

            config = TrainingConfig(
                seed=42,
                batch_size=2,
                learning_rate=1e-3,
                num_epochs=1,
                d_model=self.d_model,
                num_heads=self.num_heads,
                encoder_layers=2,
                decoder_layers=2,
                ffn_dim=self.ffn_dim,
                checkpoint_dir=ckpt_dir,
                log_to_clearml=True,
                device="cpu",
            )

            model = TemplateConditionedLayoutModel(
                content_dim=18,
                template_dim=16,
                d_model=self.d_model,
                num_heads=self.num_heads,
                encoder_layers=2,
                decoder_layers=2,
                ffn_dim=self.ffn_dim,
            )

            trainer = LayoutTrainer(
                model=model,
                config=config,
                train_dataset=train_ds,
                val_dataset=val_ds,
                clearml_task=mock_task,
            )

            # 1. Verify ClearML configuration connected
            mock_task.connect.assert_called_once()
            connected_config = mock_task.connect.call_args[0][0]
            self.assertEqual(connected_config["model/architecture"], "TemplateConditionedLayoutModel")
            self.assertEqual(connected_config["training/batch_size"], 2)
            self.assertEqual(connected_config["training/learning_rate"], 1e-3)
            self.assertIn("loss_weights/w_box", connected_config)
            self.assertIn("loss_weights/w_giou", connected_config)
            self.assertIn("loss_weights/w_bound", connected_config)
            self.assertIn("loss_weights/w_overlap", connected_config)
            self.assertIn("dataset/version", connected_config)

            # 2. Train 1 epoch
            history = trainer.train()

            # 3. Verify all 8 individual loss components reported to ClearML
            reported_titles = [call[1]["title"] for call in mock_logger.report_scalar.call_args_list]
            expected_loss_titles = [
                "Loss - Total",
                "Loss - Box Regression",
                "Loss - GIoU",
                "Loss - Boundary",
                "Loss - Overlap",
                "Loss - Column Assignment",
                "Loss - Aspect Ratio",
                "Loss - Reading Order",
                "Loss - Macro Column",
            ]
            for lt in expected_loss_titles:
                self.assertIn(lt, reported_titles, f"Missing metric report for {lt}")

            # 4. Verify evaluation metrics reported
            self.assertIn("Metrics - Alignment", reported_titles)
            self.assertIn("Metrics - Geometric Constraints", reported_titles)
            self.assertIn("Metrics - Structure", reported_titles)

            # 5. Verify local checkpoint saving still functions
            self.assertTrue((ckpt_dir / "best_model.pt").exists())
            self.assertTrue((ckpt_dir / "latest_checkpoint.pt").exists())


if __name__ == "__main__":
    unittest.main()
