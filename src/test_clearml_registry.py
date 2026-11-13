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

import joblib
import numpy as np
import pandas as pd
import torch

from src.features.config import KNOWN_CATEGORY_SLUGS
from src.features.request_schema import ContentBlockDemand, PosterRequest
from src.models.layout_retrieval import LayoutRetriever
from src.neural_layout.layout_model import TemplateConditionedLayoutModel
from src.neural_layout.losses import LossWeights
from src.neural_layout.trainer import TrainingConfig, set_seed
from src.registry.clearml_registry import (
    build_model_bundle,
    build_model_name,
    build_retriever_bundle,
    compute_sha256,
    compute_train_poster_ids_sha256,
    extract_training_config_from_checkpoint,
    load_registered_model,
    load_registered_retriever,
    migrate_local_retriever_to_clearml,
    normalize_version,
    query_registered_models,
    register_model,
    register_retriever_model,
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


def _make_synthetic_posters_df(split: str = "train", start_id: int = 100, count: int = 12) -> pd.DataFrame:
    """Construct a minimal DataFrame satisfying all LayoutRetriever feature columns."""
    rows = []
    for i in range(count):
        is_port = (i % 2 == 0)
        aspect = 0.71 if is_port else 1.41
        row = {
            "poster_id": start_id + i,
            "split": split,
            "orientation": "portrait" if is_port else "landscape",
            "aspect_ratio": aspect,
            "log_aspect_ratio": float(np.log(aspect)),
            "is_landscape": 0.0 if is_port else 1.0,
            "is_portrait": 1.0 if is_port else 0.0,
            "known_element_count": 10 + (i % 5),
            "visual_element_count": 2 + (i % 2),
            "textual_element_count": 5,
            "heading_element_count": 2,
            "media_to_text_count_ratio": 0.4,
            "visual_area_sum": 0.25,
            "textual_area_sum": 0.35,
            "header_area_sum": 0.10,
            "sum_box_area_fraction": 0.70,
            "visual_to_body_area_ratio": 0.42,
            "area_norm_p25": 0.02,
            "area_norm_p50": 0.05,
            "area_norm_p75": 0.12,
            "w_norm_p50": 0.28,
            "h_norm_p50": 0.15,
            "estimated_column_count": 2 if is_port else 3,
            "union_box_coverage": 0.65,
            "horiz_profile_symmetry": 0.85,
            "frac_aligned_left": 0.40,
            "count_title": 1,
            "count_section": 2,
            "count_text": 3,
            "count_list": 1,
            "count_figure": 2,
            "count_table": 0,
            "count_caption": 2,
        }
        for slug in KNOWN_CATEGORY_SLUGS:
            row[f"log1p_count_{slug}"] = float(np.log1p(row.get(f"count_{slug}", 1)))
            row[f"fraction_{slug}"] = 0.1
            row[f"area_sum_{slug}"] = 0.08
        for r in range(4):
            for c in range(4):
                row[f"occ_grid_r{r}_c{c}"] = 0.5 + 0.01 * ((i + r + c) % 4)
        for b in range(16):
            row[f"horiz_occ_{b:02d}"] = 0.5
            row[f"vert_occ_{b:02d}"] = 0.5
        rows.append(row)
    return pd.DataFrame(rows)


class TestClearMLRetrieverRegistryUnit(unittest.TestCase):
    """Unit tests for ClearML registration, verification, and loading of LayoutRetriever."""

    def setUp(self):
        self.train_df = _make_synthetic_posters_df(split="train", start_id=100, count=14)
        self.retriever = LayoutRetriever(mmr_lambda=0.78, strict_orientation=True).fit(self.train_df)
        self.sample_request = PosterRequest(
            request_id="req_test_01",
            requested_aspect_ratio=0.71,
            requested_orientation="portrait",
            blocks=[
                ContentBlockDemand("b1", "title", "title", word_count=12),
                ContentBlockDemand("b2", "text", "text", word_count=120),
                ContentBlockDemand("b3", "figure", "figure", aspect_ratio=1.3),
            ],
        )

    def test_retriever_bundle_checksums_and_anti_leakage_guard(self):
        """Verify retriever bundle creation, SHA-256 verification, and train-only split guard."""
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "ret_bundle"
            manifest = build_retriever_bundle(
                bundle_dir=bundle_dir,
                retriever_model=self.retriever,
                training_task_id="task_ret_001",
                dataset_identifier="SciPostLayout v2 (70/15/15)",
                semantic_version="2.1.0",
                validation_metrics={"orientation_match_at_k": 1.0},
            )
            self.assertTrue((bundle_dir / "retriever.joblib").exists())
            self.assertTrue((bundle_dir / "model_config.json").exists())
            self.assertTrue((bundle_dir / "feature_schema.json").exists())
            self.assertTrue((bundle_dir / "preprocessing_config.json").exists())
            self.assertEqual(manifest["model_config"]["architecture"], "LayoutRetriever")
            self.assertEqual(
                manifest["model_config"]["train_poster_ids_sha256"],
                compute_train_poster_ids_sha256(self.train_df),
            )
            verified = verify_bundle_checksums(bundle_dir)
            self.assertEqual(verified["semantic_version"], "2.1.0")

            # Corrupt joblib artifact -> must raise ValueError
            (bundle_dir / "retriever.joblib").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                verify_bundle_checksums(bundle_dir)

        # Unfitted retriever -> must raise ValueError
        unfitted = LayoutRetriever()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_retriever_bundle(
                    bundle_dir=Path(tmp) / "bad",
                    retriever_model=unfitted,
                    training_task_id="t",
                    dataset_identifier="d",
                    semantic_version="1.0.0",
                )

        # Retriever containing dev/test rows -> must fail train-only guard
        leaked_df = pd.concat([self.train_df, _make_synthetic_posters_df(split="dev", start_id=900, count=2)], ignore_index=True)
        leaked_ret = LayoutRetriever().fit(leaked_df)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_retriever_bundle(
                    bundle_dir=Path(tmp) / "leak",
                    retriever_model=leaked_ret,
                    training_task_id="t",
                    dataset_identifier="d",
                    semantic_version="1.0.0",
                )

    def test_register_download_and_inference_equivalence(self):
        """End-to-end mocked ClearML registration, download, deserialization, and exact query equivalence."""
        orig_top_k = self.retriever.retrieve_for_request(self.sample_request, top_k=5, diversify=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cloud_storage = tmp_path / "mock_clearml_ret"
            mock_task = MagicMock()
            mock_task.id = "task_ret_123"
            mock_task.output_uri = "https://files.clear.ml"

            registered_models: list[SimpleNamespace] = []

            def fake_update_weights(weights_filename, **kwargs):
                import shutil
                shutil.copytree(weights_filename, cloud_storage)
                return "https://files.clear.ml/poster_layout_retriever-v2.1.0.zip"

            mock_output = MagicMock()
            mock_output.id = "clearml_ret_001"
            mock_output.update_weights.side_effect = fake_update_weights
            mock_output.set_all_metadata.return_value = True

            def fake_output_ctor(**kwargs):
                manifest = kwargs["config_dict"]["manifest"]
                registered_models.append(
                    SimpleNamespace(
                        id="clearml_ret_001",
                        name=kwargs["name"],
                        tags=kwargs["tags"],
                        published=False,
                        task=mock_task.id,
                        url="https://files.clear.ml/poster_layout_retriever-v2.1.0.zip",
                        _meta={
                            "model_family": manifest["model_family"],
                            "semantic_version": manifest["semantic_version"],
                            "lifecycle_status": manifest["lifecycle_status"],
                            "dataset_identifier": manifest["dataset_identifier"],
                            "git_commit": manifest["git_commit"],
                            "checkpoint_sha256": manifest["checkpoint_sha256"],
                        },
                    )
                )
                return mock_output

            def fake_input_ctor(model_id):
                inst = MagicMock()
                inst.id = model_id
                inst.url = "https://files.clear.ml/poster_layout_retriever-v2.1.0.zip"
                inst.get_local_copy.return_value = str(cloud_storage)
                return inst

            with (
                patch("src.registry.clearml_registry.Task.get_task", return_value=mock_task),
                patch("src.registry.clearml_registry.Model.query_models", side_effect=lambda **kw: [
                    m for m in registered_models if kw.get("model_name") is None or m.name == kw.get("model_name")
                ]),
                patch("src.registry.clearml_registry.OutputModel", side_effect=fake_output_ctor),
                patch("src.registry.clearml_registry.InputModel", side_effect=fake_input_ctor),
            ):
                ret_id = register_retriever_model(
                    retriever_model=self.retriever,
                    training_task_id="task_ret_123",
                    semantic_version="2.1.0",
                    validation_metrics={"orientation_match_at_k": 1.0},
                    lifecycle_status="candidate",
                )
                self.assertEqual(ret_id, "clearml_ret_001")

                # Duplicate version must be rejected
                with self.assertRaises(ValueError):
                    register_retriever_model(
                        retriever_model=self.retriever,
                        training_task_id="task_ret_123",
                        semantic_version="2.1.0",
                    )

                loaded_ret, manifest = load_registered_retriever(
                    semantic_version="2.1.0",
                    return_manifest=True,
                )
                self.assertEqual(loaded_ret.clearml_model_id, "clearml_ret_001")
                self.assertEqual(loaded_ret.clearml_model_version, "2.1.0")
                self.assertEqual(manifest["model_config"]["architecture"], "LayoutRetriever")

                loaded_top_k = loaded_ret.retrieve_for_request(self.sample_request, top_k=5, diversify=True)
                pd.testing.assert_frame_equal(orig_top_k, loaded_top_k)

                # Test neural generator checkpoint/manifest captures retriever provenance
                tcfg = TrainingConfig(
                    retriever_model_id=loaded_ret.clearml_model_id,
                    retriever_version=loaded_ret.clearml_model_version,
                )
                model = TemplateConditionedLayoutModel()
                ckpt_data = {
                    "epoch": 0,
                    "model_state_dict": model.state_dict(),
                    "config": tcfg,
                    "retriever_info": {
                        "model_name": "poster_layout_retriever",
                        "model_id": loaded_ret.clearml_model_id,
                        "version": loaded_ret.clearml_model_version,
                    },
                }
                extracted_tcfg = extract_training_config_from_checkpoint(ckpt_data)
                self.assertEqual(extracted_tcfg["retriever_model_id"], "clearml_ret_001")
                self.assertEqual(extracted_tcfg["retriever_version"], "2.1.0")

                gen_manifest = build_model_bundle(
                    bundle_dir=tmp_path / "gen_bundle",
                    checkpoint_data=ckpt_data,
                    model_config={"architecture": "TemplateConditionedLayoutModel", "d_model": 128},
                    training_config=extracted_tcfg,
                    training_task_id="task_gen_1",
                    dataset_identifier="SciPostLayout v2 (70/15/15)",
                    semantic_version="0.2.0",
                    validation_metrics={"val_iou_mean": 0.4},
                )
                self.assertEqual(gen_manifest["retriever"]["model_id"], "clearml_ret_001")
                self.assertEqual(gen_manifest["retriever"]["version"], "2.1.0")

    def test_retriever_failure_and_legacy_migration(self):
        """Verify failure on missing/incompatible artifacts and legacy pipeline.joblib compatibility."""
        # Implicit loading without ID or version must fail
        with self.assertRaises(ValueError):
            load_registered_retriever()

        with patch("src.registry.clearml_registry.Model.query_models", return_value=[]):
            with self.assertRaises(LookupError):
                load_registered_retriever(semantic_version="9.9.9")

        # Incompatible architecture (e.g. neural generator bundle passed to load_registered_retriever)
        with tempfile.TemporaryDirectory() as tmp:
            bad_bundle = Path(tmp) / "wrong_arch"
            bad_bundle.mkdir()
            (bad_bundle / "model_config.json").write_text(
                json.dumps({"architecture": "TemplateConditionedLayoutModel"}), encoding="utf-8"
            )
            (bad_bundle / "manifest.json").write_text(
                json.dumps({"bundle_sha256s": {"model_config.json": compute_sha256(bad_bundle / "model_config.json")}}),
                encoding="utf-8",
            )
            fake_input = MagicMock()
            fake_input.get_local_copy.return_value = str(bad_bundle)
            with (
                patch("src.registry.clearml_registry.InputModel", return_value=fake_input),
                self.assertRaises(ValueError),
            ):
                load_registered_retriever(model_id="bad_arch_id")

            # Missing retriever.joblib inside a LayoutRetriever bundle -> FileNotFoundError
            missing_dir = Path(tmp) / "missing_joblib"
            missing_dir.mkdir()
            (missing_dir / "model_config.json").write_text(
                json.dumps({"architecture": "LayoutRetriever", "artifact_filename": "retriever.joblib"}),
                encoding="utf-8",
            )
            (missing_dir / "manifest.json").write_text(
                json.dumps({"bundle_sha256s": {"model_config.json": compute_sha256(missing_dir / "model_config.json")}}),
                encoding="utf-8",
            )
            fake_input_missing = MagicMock()
            fake_input_missing.get_local_copy.return_value = str(missing_dir)
            with (
                patch("src.registry.clearml_registry.InputModel", return_value=fake_input_missing),
                self.assertRaises(FileNotFoundError),
            ):
                load_registered_retriever(model_id="missing_joblib_id")

            # Legacy local pipeline.joblib migration helper
            legacy_dir = Path(tmp) / "legacy_v2"
            legacy_dir.mkdir()
            joblib.dump({"model": self.retriever, "preprocessor": None, "feature_columns": []}, legacy_dir / "pipeline.joblib")
            (legacy_dir / "manifest.json").write_text(
                json.dumps({"version": "v2.0.0", "metrics": {"orientation_match_at_k": 1.0}}),
                encoding="utf-8",
            )

            mock_task = MagicMock()
            mock_task.id = "mig_task"
            mock_task.output_uri = "https://files.clear.ml"
            mock_out = MagicMock()
            mock_out.id = "migrated_ret_id"
            mock_out.update_weights.return_value = "https://files.clear.ml/migrated.zip"
            mock_out.set_all_metadata.return_value = True
            mock_in = MagicMock()
            mock_in.url = "https://files.clear.ml/migrated.zip"

            with (
                patch("src.registry.clearml_registry.Task.get_task", return_value=mock_task),
                patch("src.registry.clearml_registry.Model.query_models", return_value=[]),
                patch("src.registry.clearml_registry.OutputModel", return_value=mock_out),
                patch("src.registry.clearml_registry.InputModel", return_value=mock_in),
            ):
                mig_id = migrate_local_retriever_to_clearml(legacy_dir, training_task_id="mig_task")
                self.assertEqual(mig_id, "migrated_ret_id")




class TestClearMLModelRegistryV1(unittest.TestCase):
    """Unit test for v1 ClearMLModelRegistry dual-mirror and offline lifecycle."""

    def test_clearml_registry_offline_lifecycle(self):
        import shutil
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from src.registry.clearml_registry import ClearMLModelRegistry

        temp_dir = Path(tempfile.mkdtemp())
        try:
            registry = ClearMLModelRegistry(
                project_name="Infographix-Tests",
                offline=True,
                local_fallback_root=temp_dir,
            )

            dummy = DummyClassifier(strategy="constant", constant=0)
            dummy.fit(np.zeros((10, 4)), np.zeros(10))

            res = registry.register_model(
                model_name="test_clearml_clusterer",
                version="v1.0.0",
                model_object=dummy,
                preprocessor=None,
                feature_schema_columns=["f1", "f2", "f3", "f4"],
                hyperparameters={"eps": 4.8},
                metrics={"train_silhouette": 0.24},
                stage="production",
                description="ClearML Test Clusterer",
                tags=["test", "unit-test"],
            )

            self.assertEqual(res["model_name"], "test_clearml_clusterer")
            self.assertEqual(res["version"], "v1.0.0")
            self.assertEqual(res["stage"], "production")
            self.assertTrue(Path(res["local_path"]).exists())

            model, prep, meta = registry.load_production_model("test_clearml_clusterer")
            self.assertIsNotNone(model)
            pred = model.predict(np.zeros((2, 4)))
            self.assertEqual(list(pred), [0, 0])

            models = registry.list_models()
            self.assertIn("test_clearml_clusterer", models)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
