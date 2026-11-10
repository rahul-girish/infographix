"""Unit Tests for ClearML Native Model Registry Integration (src/registry/clearml_registry.py).

Covers:
  1. Model bundle construction & optimizer exclusion.
  2. Manifest generation & all required metadata fields.
  3. SHA-256 checksum calculation and verification.
  4. Corrupted or incomplete artifact bundle detection.
  5. Provenance verification against ClearML Task (valid and mismatch cases).
  6. Immutable versioning enforcement (no silent overwrite).
  7. Registration interface (`register_model`) with mocked ClearML SDK.
  8. Querying (`query_registered_models`) and explicit lifecycle transitions (`set_model_lifecycle_status`).
  9. Exact-version retrieval (`load_registered_model`), missing model error handling, and no local fallback.
 10. Model reconstruction and prediction equivalence before and after bundle packaging.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.neural_layout.layout_model import TemplateConditionedLayoutModel
from src.neural_layout.losses import LossWeights
from src.neural_layout.trainer import TrainingConfig, set_seed
from src.registry.clearml_registry import (
    build_model_bundle,
    build_model_name,
    compute_sha256,
    load_registered_model,
    normalize_version,
    query_registered_models,
    register_model,
    set_model_lifecycle_status,
    verify_bundle_checksums,
    verify_checkpoint_provenance,
)


class TestClearMLRegistryUnit(unittest.TestCase):
    """Unit test suite for ClearML model bundling, provenance, registration, and retrieval."""

    def setUp(self):
        set_seed(42)
        self.model = TemplateConditionedLayoutModel(
            content_dim=18,
            template_dim=16,
            d_model=64,
            num_heads=2,
            encoder_layers=2,
            decoder_layers=2,
            ffn_dim=128,
            num_columns=5,
            dropout=0.1,
        )
        self.model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=4e-4)
        self.config = TrainingConfig(
            seed=42,
            batch_size=8,
            learning_rate=4e-4,
            num_epochs=3,
            d_model=64,
            num_heads=2,
            encoder_layers=2,
            decoder_layers=2,
            ffn_dim=128,
            dropout=0.1,
            loss_weights=LossWeights(),
            task_name="Neural Layout Training - Unit Test",
        )
        self.metrics = {
            "val_loss_total": 5.1234,
            "val_iou_mean": 0.1234,
            "val_column_accuracy": 0.65,
        }
        self.checkpoint_data = {
            "epoch": 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics,
            "config": self.config,
        }

    def test_version_normalization(self):
        self.assertEqual(normalize_version("0.1.0"), "0.1.0")
        self.assertEqual(normalize_version("v0.2.3"), "0.2.3")
        self.assertEqual(build_model_name("poster_layout_generator", "v0.1.0"), "poster_layout_generator-v0.1.0")
        with self.assertRaises(ValueError):
            normalize_version("latest")
        with self.assertRaises(ValueError):
            normalize_version("0.1")

    def test_bundle_construction_manifest_and_checksums(self):
        """Verify bundle files, optimizer exclusion, manifest fields, and SHA-256 verification."""
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            manifest = build_model_bundle(
                bundle_dir=bundle_dir,
                checkpoint_data=self.checkpoint_data,
                model_config={
                    "architecture": "TemplateConditionedLayoutModel",
                    "content_dim": 18,
                    "template_dim": 16,
                    "d_model": 64,
                    "num_heads": 2,
                    "encoder_layers": 2,
                    "decoder_layers": 2,
                    "ffn_dim": 128,
                    "num_columns": 5,
                    "dropout": 0.1,
                },
                training_config={"batch_size": 8, "learning_rate": 4e-4},
                training_task_id="task_abc_123",
                dataset_identifier="SciPostLayout v2.0.0",
                semantic_version="0.1.0",
                validation_metrics=self.metrics,
                lifecycle_status="candidate",
                git_commit="deadbeef",
            )

            expected_files = [
                "model_weights.pt",
                "model_config.json",
                "feature_schema.json",
                "preprocessing_config.json",
                "environment.json",
                "manifest.json",
            ]
            for fname in expected_files:
                self.assertTrue((bundle_dir / fname).exists(), f"Missing {fname}")

            # Ensure optimizer state was NOT bundled
            loaded_weights = torch.load(bundle_dir / "model_weights.pt", map_location="cpu", weights_only=True)
            self.assertNotIn("optimizer_state_dict", loaded_weights)
            self.assertIn("content_encoder.input_proj.0.weight", loaded_weights)

            # Check manifest fields
            self.assertEqual(manifest["project_name"], "infographix")
            self.assertEqual(manifest["model_family"], "poster_layout_generator")
            self.assertEqual(manifest["semantic_version"], "0.1.0")
            self.assertEqual(manifest["clearml_training_task_id"], "task_abc_123")
            self.assertEqual(manifest["dataset_identifier"], "SciPostLayout v2.0.0")
            self.assertEqual(manifest["git_commit"], "deadbeef")
            self.assertEqual(manifest["checkpoint_sha256"], compute_sha256(bundle_dir / "model_weights.pt"))

            # Verify checksum validation passes on intact bundle
            verified = verify_bundle_checksums(bundle_dir)
            self.assertEqual(verified["checkpoint_sha256"], manifest["checkpoint_sha256"])

            # Verify corrupted weights detection
            with open(bundle_dir / "model_weights.pt", "ab") as f:
                f.write(b"corrupt")
            with self.assertRaises(ValueError):
                verify_bundle_checksums(bundle_dir)

            # Restore and verify missing file detection
            torch.save(self.model.state_dict(), bundle_dir / "model_weights.pt")
            (bundle_dir / "feature_schema.json").unlink()
            with self.assertRaises(FileNotFoundError):
                verify_bundle_checksums(bundle_dir)

    def test_provenance_verification(self):
        """Verify checkpoint provenance validation catches mismatches."""
        mock_task = MagicMock()
        mock_task.id = "task_123"
        mock_task.name = "Neural Layout Training - Unit Test"
        mock_task.get_parameters_as_dict.return_value = {
            "Experiment Configuration": {
                "training": {"batch_size": "8", "seed": "42"},
                "model": {"d_model": "64"},
            }
        }
        mock_task.get_reported_scalars.return_value = {
            "Loss - Total": {"val": {"x": [1, 2], "y": [6.0, 5.1234]}}
        }

        res = verify_checkpoint_provenance(self.checkpoint_data, mock_task)
        self.assertTrue(res["verified"])

        # Mismatch on explicit clearml_task_id
        bad_ckpt = dict(self.checkpoint_data, clearml_task_id="wrong_task_id")
        with self.assertRaises(ValueError):
            verify_checkpoint_provenance(bad_ckpt, mock_task)

        # Mismatch on validation metric at epoch+1
        bad_task = MagicMock()
        bad_task.id = "task_123"
        bad_task.name = "Neural Layout Training - Unit Test"
        bad_task.get_parameters_as_dict.return_value = mock_task.get_parameters_as_dict.return_value
        bad_task.get_reported_scalars.return_value = {
            "Loss - Total": {"val": {"x": [1, 2], "y": [6.0, 9.9999]}}
        }
        with self.assertRaises(ValueError):
            verify_checkpoint_provenance(self.checkpoint_data, bad_task)

    def test_register_and_load_equivalence_mocked_clearml(self):
        """End-to-end mocked registration, querying, retrieval, corruption guard, and prediction equivalence."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_path = tmp_path / "best_model.pt"
            torch.save(self.checkpoint_data, ckpt_path)

            # Fixed validation sample input
            c_feats = torch.randn(1, 5, 18)
            t_feats = torch.randn(1, 4, 16)
            c_mask = torch.ones(1, 5, dtype=torch.bool)
            t_mask = torch.ones(1, 4, dtype=torch.bool)
            with torch.no_grad():
                orig_out = self.model(c_feats, t_feats, c_mask, t_mask)

            # Mock ClearML infrastructure storing uploaded bundle in tmp_path/server_bundle
            server_bundle_copy = tmp_path / "server_bundle"
            mock_orig_task = MagicMock()
            mock_orig_task.id = "8cd6cb755a084535a7bba3ae551e3e20"
            mock_orig_task.name = "Neural Layout Training - Unit Test"
            mock_orig_task.output_uri = "https://files.clear.ml"
            mock_orig_task.get_parameters_as_dict.return_value = {
                "Experiment Configuration": {
                    "training": {"batch_size": "8", "seed": "42"},
                    "model": {"d_model": "64"},
                    "environment": {"git_commit": "abc1234"},
                }
            }
            mock_orig_task.get_reported_scalars.return_value = {
                "Loss - Total": {"val": {"x": [1, 2], "y": [6.0, 5.1234]}}
            }

            registered_state = {}

            def fake_update_weights(weights_filename, upload_uri, target_filename, auto_delete_file, async_enable, is_package):
                import shutil
                shutil.copytree(weights_filename, server_bundle_copy)
                registered_state["uri"] = f"https://files.clear.ml/{target_filename}"
                return registered_state["uri"]

            mock_output_model = MagicMock()
            mock_output_model.id = "clearml_model_001"
            mock_output_model.update_weights.side_effect = fake_update_weights
            mock_output_model.set_all_metadata.return_value = True

            mock_input_model = MagicMock()
            mock_input_model.id = "clearml_model_001"
            mock_input_model.url = "https://files.clear.ml/poster_layout_generator-v0.1.0.zip"
            mock_input_model.tags = ["family:poster_layout_generator", "version:v0.1.0", "stage:candidate"]
            mock_input_model.get_local_copy.return_value = str(server_bundle_copy)

            mock_query_model = MagicMock()
            mock_query_model.id = "clearml_model_001"
            mock_query_model.name = "poster_layout_generator-v0.1.0"
            mock_query_model.tags = ["family:poster_layout_generator", "version:v0.1.0", "stage:candidate"]
            mock_query_model.published = False
            mock_query_model.task = "8cd6cb755a084535a7bba3ae551e3e20"
            mock_query_model.url = "https://files.clear.ml/poster_layout_generator-v0.1.0.zip"
            mock_query_model.get_all_metadata.return_value = {
                "lifecycle_status": "candidate",
                "semantic_version": "0.1.0",
                "dataset_identifier": "SciPostLayout v2.0.0",
                "git_commit": "abc1234",
                "checkpoint_sha256": "dummyhash",
            }

            with patch("src.registry.clearml_registry.Task.get_task", return_value=mock_orig_task), \
                 patch("src.registry.clearml_registry.OutputModel", return_value=mock_output_model), \
                 patch("src.registry.clearml_registry.InputModel", return_value=mock_input_model), \
                 patch("src.registry.clearml_registry.APIClient") as mock_api_cls, \
                 patch("src.registry.clearml_registry.Model.query_models") as mock_query:

                # First call (duplication check) returns [], subsequent calls return [mock_query_model]
                mock_query.side_effect = [[], [mock_query_model], [mock_query_model], [mock_query_model]]

                # 1. Register
                model_id = register_model(
                    checkpoint_path=ckpt_path,
                    training_task_id="8cd6cb755a084535a7bba3ae551e3e20",
                    semantic_version="0.1.0",
                    dataset_identifier="SciPostLayout v2.0.0",
                    parent_model_id="parent_000",
                )
                self.assertEqual(model_id, "clearml_model_001")
                mock_output_model.wait_for_uploads.assert_called_once()
                mock_api_cls.return_value.models.edit.assert_called_with(model="clearml_model_001", parent="parent_000")

                # 2. Query
                found = query_registered_models(semantic_version="0.1.0")
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0]["model_id"], "clearml_model_001")
                self.assertFalse(found[0]["published"])

                # 3. Duplicate version registration must fail
                with self.assertRaises(ValueError):
                    register_model(
                        checkpoint_path=ckpt_path,
                        training_task_id="8cd6cb755a084535a7bba3ae551e3e20",
                        semantic_version="0.1.0",
                    )

                # 4. Delete original local checkpoint to prove zero dependence on local training artifact
                ckpt_path.unlink()
                self.assertFalse(ckpt_path.exists())

                # 5. Retrieve by semantic version and verify prediction equivalence
                loaded_model, manifest = load_registered_model(
                    semantic_version="0.1.0",
                    return_manifest=True,
                )
                self.assertEqual(manifest["semantic_version"], "0.1.0")
                with torch.no_grad():
                    new_out = loaded_model(c_feats, t_feats, c_mask, t_mask)

                for key in ["pred_boxes", "pred_column_logits", "pred_order_scores", "pred_macro_column_logits"]:
                    np.testing.assert_allclose(
                        orig_out[key].numpy(),
                        new_out[key].numpy(),
                        rtol=1e-6,
                        atol=1e-6,
                    )

                # 6. Lifecycle status transition (does not publish)
                res_stage = set_model_lifecycle_status("staging", model_id="clearml_model_001")
                self.assertEqual(res_stage["lifecycle_status"], "staging")
                self.assertIn("stage:staging", res_stage["tags"])

    def test_missing_and_ambiguous_retrieval_guards(self):
        """Verify loader fails cleanly without falling back to local files or latest model."""
        with self.assertRaises(ValueError):
            load_registered_model()

        with patch("src.registry.clearml_registry.Model.query_models", return_value=[]):
            with self.assertRaises(LookupError):
                load_registered_model(semantic_version="9.9.9")

        m1 = SimpleNamespace(id="id1", name="poster_layout_generator-v0.1.0")
        m2 = SimpleNamespace(id="id2", name="poster_layout_generator-v0.1.0")
        with patch("src.registry.clearml_registry.Model.query_models", return_value=[m1, m2]):
            with self.assertRaises(ValueError):
                load_registered_model(semantic_version="0.1.0")


if __name__ == "__main__":
    unittest.main()
