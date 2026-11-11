from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from inference_service.app.config import Settings
from inference_service.app.main import create_app
from inference_service.app.services.inference_manager import (
    InferenceExecutionError,
    validate_and_normalize_geometry,
)
from src.features.config import KNOWN_CATEGORY_SLUGS
from src.models.layout_retrieval import (
    POSTER_COMPOSITION_COLS,
    SHARED_CANVAS_COLS,
    SHARED_CAPACITY_COLS,
    SHARED_INVENTORY_COLS,
    SHARED_STRUCTURE_COLS,
    LayoutRetriever,
)
from src.neural_layout.layout_model import TemplateConditionedLayoutModel
from src.neural_layout.losses import LossWeights
from src.neural_layout.trainer import TrainingConfig, set_seed
from src.registry.clearml_registry import (
    build_model_bundle,
    build_retriever_bundle,
)


async def _asgi_request(
    app,
    method: str,
    path: str,
    body: dict | None = None,
) -> tuple[int, dict]:
    """Invoke a FastAPI ASGI application directly without requiring httpx."""
    raw_body = json.dumps(body).encode("utf-8") if body is not None else b""
    headers = [(b"host", b"testserver")]
    if body is not None:
        headers.append((b"content-type", b"application/json"))
        headers.append((b"content-length", str(len(raw_body)).encode("ascii")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    sent_chunks: list[bytes] = []
    status_code = 500
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": raw_body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = int(message["status"])
        elif message["type"] == "http.response.body":
            sent_chunks.append(message.get("body", b""))

    await app(scope, receive, send)
    payload_bytes = b"".join(sent_chunks)
    parsed = json.loads(payload_bytes.decode("utf-8")) if payload_bytes else {}
    return status_code, parsed


def _make_mock_posters_df(n_posters: int = 12) -> pd.DataFrame:
    """Create a synthetic train-split posters dataframe matching LayoutRetriever schema."""
    rng = np.random.default_rng(42)
    all_cols = set(
        SHARED_CANVAS_COLS
        + SHARED_INVENTORY_COLS
        + SHARED_CAPACITY_COLS
        + SHARED_STRUCTURE_COLS
        + POSTER_COMPOSITION_COLS
        + [f"count_{s}" for s in KNOWN_CATEGORY_SLUGS]
        + [
            "poster_id",
            "split",
            "aspect_ratio",
            "orientation",
            "canvas_width_px",
            "canvas_height_px",
            "total_element_count",
        ]
    )
    rows = []
    for i in range(n_posters):
        orient = "landscape" if i % 2 == 0 else "portrait"
        asp = 1.414 if orient == "landscape" else 0.707
        row: dict[str, object] = {c: 0.05 for c in all_cols}
        row.update({
            "poster_id": 1000 + i,
            "split": "train",
            "aspect_ratio": asp,
            "log_aspect_ratio": float(np.log(asp)),
            "orientation": orient,
            "is_landscape": 1.0 if orient == "landscape" else 0.0,
            "is_portrait": 1.0 if orient == "portrait" else 0.0,
            "canvas_width_px": 1000.0 if orient == "landscape" else 700.0,
            "canvas_height_px": 700.0 if orient == "landscape" else 1000.0,
            "estimated_column_count": 3 if orient == "landscape" else 2,
            "total_element_count": 10,
            "known_element_count": 10,
            "visual_element_count": 1,
            "textual_element_count": 5,
            "heading_element_count": 4,
            "count_title": 1,
            "count_author_info": 1,
            "count_section": 3,
            "count_text": 3,
            "count_figure": 1,
            "count_caption": 1,
            "union_box_coverage": 0.65,
            "sum_box_area_fraction": 0.68,
            "visual_area_sum": 0.18,
            "textual_area_sum": 0.40,
            "header_area_sum": 0.10,
        })
        for r in range(4):
            for c in range(4):
                row[f"occ_grid_r{r}_c{c}"] = float(rng.uniform(0.2, 0.8))
        rows.append(row)
    return pd.DataFrame(rows)


def _make_mock_elements_df(poster_ids: list[int]) -> pd.DataFrame:
    """Create valid elements.parquet fixture matching LayoutAdapter and TemplateSlotItem."""
    rows = []
    for pid in poster_ids:
        elements = [
            (1, "title", 0.08, 0.04, 0.84, 0.08),
            (2, "author_info", 0.12, 0.13, 0.76, 0.05),
            (3, "section", 0.05, 0.22, 0.28, 0.04),
            (4, "text", 0.05, 0.27, 0.28, 0.30),
            (5, "section", 0.36, 0.22, 0.28, 0.04),
            (6, "figure", 0.36, 0.27, 0.28, 0.26),
            (7, "caption", 0.36, 0.54, 0.28, 0.05),
            (8, "section", 0.67, 0.22, 0.28, 0.04),
            (9, "text", 0.67, 0.27, 0.28, 0.32),
        ]
        for eid, slug, x0, y0, w, h in elements:
            rows.append({
                "poster_id": pid,
                "element_id": eid,
                "category_slug": slug,
                "x0": x0,
                "y0": y0,
                "w_norm": w,
                "h_norm": h,
                "cx": x0 + w / 2.0,
                "cy": y0 + h / 2.0,
                "area_norm": w * h,
                "raw_x": x0 * 1000.0,
                "raw_y": y0 * 700.0,
                "raw_w": w * 1000.0,
                "raw_h": h * 700.0,
            })
    return pd.DataFrame(rows)


class TestInferenceService(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

        # Build authentic ClearML bundles on disk (downloaded via mocked ClearML SDK)
        posters_df = _make_mock_posters_df(12)
        self.retriever = LayoutRetriever(mmr_lambda=0.78, strict_orientation=True).fit(posters_df)
        self.retriever_bundle_dir = self.root / "clearml_cache" / "retriever_bundle"
        build_retriever_bundle(
            bundle_dir=self.retriever_bundle_dir,
            retriever_model=self.retriever,
            training_task_id="task_ret_001",
            dataset_identifier="SciPostLayout v2 (test)",
            semantic_version="2.0.0",
            validation_metrics={"column_exact_match_at_1": 0.97},
            lifecycle_status="production",
        )

        neural_model = TemplateConditionedLayoutModel(
            content_dim=18,
            template_dim=16,
            d_model=64,
            num_heads=2,
            encoder_layers=2,
            decoder_layers=2,
            ffn_dim=128,
            num_columns=5,
            dropout=0.0,
        )
        cfg = TrainingConfig(
            seed=42,
            batch_size=4,
            d_model=64,
            num_heads=2,
            encoder_layers=2,
            decoder_layers=2,
            ffn_dim=128,
            dropout=0.0,
            loss_weights=LossWeights(),
        )
        ckpt_data = {
            "epoch": 2,
            "model_state_dict": neural_model.state_dict(),
            "metrics": {"val_loss_total": 1.25},
            "config": cfg,
        }
        self.neural_bundle_dir = self.root / "clearml_cache" / "neural_bundle"
        build_model_bundle(
            bundle_dir=self.neural_bundle_dir,
            checkpoint_data=ckpt_data,
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
                "dropout": 0.0,
                "bbox_parameterization": "independent",
                "use_anchor_conditioning": False,
                "min_box_width": 0.01,
                "min_box_height": 0.005,
            },
            training_config={"checkpoint_epoch": 2},
            training_task_id="task_neural_001",
            dataset_identifier="SciPostLayout v2 (test)",
            semantic_version="0.1.0",
            validation_metrics={"val_loss_total": 1.25},
            lifecycle_status="candidate",
        )

        # Write elements.parquet fixture
        self.elements_path = self.root / "elements.parquet"
        _make_mock_elements_df(posters_df["poster_id"].tolist()).to_parquet(
            self.elements_path, index=False
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _base_settings(self, **overrides) -> Settings:
        params = {
            "CLEARML_API_ACCESS_KEY": "test_key_123",
            "CLEARML_API_SECRET_KEY": "test_secret_456",
            "RETRIEVER_MODEL_VERSION": "v2.0.0",
            "NEURAL_MODEL_VERSION": "v0.1.0",
            "ELEMENTS_PARQUET_PATH": str(self.elements_path),
            "DEFAULT_ENGINE": "neural",
            "SKIP_STARTUP_INIT": True,
        }
        params.update(overrides)
        return Settings(**params)

    def _mock_clearml_registry(self, retriever_dir: Path | None = None, neural_dir: Path | None = None):
        r_dir = retriever_dir or self.retriever_bundle_dir
        n_dir = neural_dir or self.neural_bundle_dir

        def fake_query_models(project_name=None, model_name=None, **kwargs):
            if model_name == "poster_layout_retriever-v2.0.0":
                return [SimpleNamespace(id="clearml_ret_id_200", name=model_name, tags=["stage:production"])]
            if model_name == "poster_layout_generator-v0.1.0":
                return [SimpleNamespace(id="clearml_gen_id_010", name=model_name, tags=["stage:candidate"])]
            return []

        def fake_input_model(model_id: str):
            mock_obj = MagicMock()
            if model_id == "clearml_ret_id_200":
                mock_obj.get_local_copy.return_value = str(r_dir)
            elif model_id == "clearml_gen_id_010":
                mock_obj.get_local_copy.return_value = str(n_dir)
            else:
                mock_obj.get_local_copy.return_value = None
            return mock_obj

        return (
            patch("src.registry.clearml_registry.Model.query_models", side_effect=fake_query_models),
            patch("src.registry.clearml_registry.InputModel", side_effect=fake_input_model),
            patch("inference_service.app.services.model_loader.Task.set_credentials"),
        )

    def _sample_generate_payload(self, engine: str = "neural") -> dict:
        return {
            "request": {
                "title": "ClearML-Backed Poster Layout Generation",
                "authors": ["Alice Smith", "Bob Jones"],
                "affiliations": ["MIT CSAIL"],
                "orientation": "landscape",
                "columns": 3,
                "content_blocks": [
                    {
                        "id": "intro",
                        "type": "section",
                        "heading": "1. Introduction",
                        "text": "Scientific posters arrange dense research narratives across structured columns.",
                    },
                    {
                        "id": "fig1",
                        "type": "figure",
                        "heading": "Pipeline Overview",
                        "aspectRatio": 1.4,
                        "caption": "Figure 1: End-to-end architecture.",
                    },
                    {
                        "id": "conc",
                        "type": "section",
                        "heading": "2. Conclusion",
                        "text": "Both neural and constraint optimization engines share ClearML provenance.",
                    },
                ],
            },
            "engine": engine,
        }

    def test_1_2_3_liveness_readiness_and_models_endpoints(self):
        cfg = self._base_settings()
        app = create_app(cfg)

        # 1. Liveness works before initialization
        status_code, body = asyncio.run(_asgi_request(app, "GET", "/health/live"))
        self.assertEqual(status_code, 200)
        self.assertEqual(body["status"], "alive")

        # 2. Readiness and /v1/models fail cleanly before initialization
        status_code, ready_before = asyncio.run(_asgi_request(app, "GET", "/health/ready"))
        self.assertEqual(status_code, 503)
        self.assertEqual(ready_before["status"], "not_ready")

        status_code, models_before = asyncio.run(_asgi_request(app, "GET", "/v1/models"))
        self.assertEqual(status_code, 503)
        self.assertEqual(models_before["models"], [])

        # Initialize with mocked ClearML SDK
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3:
            app.state.inference_manager.initialize(cfg)

        # Readiness succeeds after initialization
        status_code, ready_after = asyncio.run(_asgi_request(app, "GET", "/health/ready"))
        self.assertEqual(status_code, 200)
        self.assertEqual(ready_after["status"], "ready")
        self.assertTrue(ready_after["model_initialized"])
        self.assertTrue(ready_after["retriever_initialized"])
        self.assertTrue(ready_after["data_dependency_ready"])
        self.assertEqual(set(ready_after["supported_engines"]), {"adaptation", "neural"})

        # 3. /v1/models reports actual loaded ClearML IDs and versions
        status_code, models_after = asyncio.run(_asgi_request(app, "GET", "/v1/models"))
        self.assertEqual(status_code, 200)
        self.assertEqual(len(models_after["models"]), 2)
        ids = {m["clearml_model_id"] for m in models_after["models"]}
        self.assertEqual(ids, {"clearml_ret_id_200", "clearml_gen_id_010"})
        self.assertTrue(models_after["data_dependency"]["ready"])

    def test_4_request_validation(self):
        cfg = self._base_settings()
        app = create_app(cfg)
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3:
            app.state.inference_manager.initialize(cfg)

        # Invalid engine & empty title
        code, _ = asyncio.run(
            _asgi_request(
                app,
                "POST",
                "/v1/layouts/generate",
                {"request": {"title": ""}, "engine": "invalid_engine"},
            )
        )
        self.assertEqual(code, 422)

    def test_5_and_6_adaptation_and_neural_engines(self):
        cfg = self._base_settings()
        app = create_app(cfg)
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3:
            app.state.inference_manager.initialize(cfg)

        # Engine A: Adaptation
        code_a, res_a = asyncio.run(
            _asgi_request(app, "POST", "/v1/layouts/generate", self._sample_generate_payload("adaptation"))
        )
        self.assertEqual(code_a, 200)
        self.assertEqual(res_a["engine"], "adaptation")
        self.assertEqual(res_a["retriever_model_id"], "clearml_ret_id_200")
        self.assertIsNotNone(res_a["retrieved_template_id"])
        self.assertGreater(len(res_a["layout"]["elements"]), 4)
        self.assertEqual(res_a["generation_metrics"]["validated_overlap_area"], 0.0)

        # Engine B: Neural
        code_n, res_n = asyncio.run(
            _asgi_request(app, "POST", "/v1/layouts/generate", self._sample_generate_payload("neural"))
        )
        self.assertEqual(code_n, 200)
        self.assertEqual(res_n["engine"], "neural")
        self.assertEqual(res_n["model_id"], "clearml_gen_id_010")
        self.assertEqual(res_n["retriever_model_id"], "clearml_ret_id_200")
        self.assertGreater(len(res_n["layout"]["elements"]), 4)
        self.assertIn("latency_ms", res_n["generation_metrics"])

    def test_7_geometry_validation(self):
        # Out-of-bounds box is clamped cleanly
        validated, metrics = validate_and_normalize_geometry(
            raw_elements=[
                {"id": "b1", "type": "text", "bbox_norm": [-0.2, 0.95, 0.5, 0.2]},
                {"id": "b2", "type": "figure", "bbox_norm": [0.1, 0.1, 0.3, 0.3]},
            ],
            canvas_w=1000,
            canvas_h=700,
        )
        self.assertEqual(metrics["clamped_boxes_count"], 1)
        self.assertGreaterEqual(validated[0]["bbox_norm"][0], 0.0)
        self.assertLessEqual(validated[0]["bbox_norm"][1] + validated[0]["bbox_norm"][3], 1.0)

        # Non-finite coordinates raise InferenceExecutionError
        with self.assertRaises(InferenceExecutionError):
            validate_and_normalize_geometry(
                raw_elements=[{"id": "bad", "type": "text", "bbox_norm": [math.nan, 0.1, 0.2, 0.2]}],
                canvas_w=1000,
                canvas_h=700,
            )

    def test_8_missing_clearml_credentials(self):
        cfg = self._base_settings(CLEARML_API_ACCESS_KEY="", CLEARML_API_SECRET_KEY="")
        app = create_app(cfg)
        app.state.inference_manager.initialize(cfg)

        code, body = asyncio.run(_asgi_request(app, "GET", "/health/ready"))
        self.assertEqual(code, 503)
        self.assertIn("ClearML credentials missing", body["error"])

    def test_9_missing_model_artifact(self):
        cfg = self._base_settings(NEURAL_MODEL_VERSION="v9.9.9")
        app = create_app(cfg)
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3:
            app.state.inference_manager.initialize(cfg)

        code, body = asyncio.run(_asgi_request(app, "GET", "/health/ready"))
        self.assertEqual(code, 503)
        self.assertFalse(body["model_initialized"])

    def test_10_invalid_checksum_and_architecture(self):
        # Tamper with model_weights.pt to trigger SHA-256 mismatch
        weights_file = self.neural_bundle_dir / "model_weights.pt"
        weights_file.write_bytes(b"corrupted-bytes")

        cfg = self._base_settings()
        app = create_app(cfg)
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3:
            app.state.inference_manager.initialize(cfg)

        code, body = asyncio.run(_asgi_request(app, "GET", "/health/ready"))
        self.assertEqual(code, 503)
        self.assertIn("checksum", body["error"].lower())

        # Also test wrong architecture in model_config.json (with matching manifest hash)
        bad_cfg_dir = self.root / "clearml_cache" / "bad_arch_bundle"
        bad_cfg_dir.mkdir(parents=True, exist_ok=True)
        (bad_cfg_dir / "model_weights.pt").write_bytes(b"dummy")
        (bad_cfg_dir / "model_config.json").write_text(
            json.dumps({"architecture": "UnsupportedModel"}), encoding="utf-8"
        )
        from src.registry.clearml_registry import compute_sha256

        (bad_cfg_dir / "manifest.json").write_text(
            json.dumps({
                "bundle_sha256s": {
                    "model_weights.pt": compute_sha256(bad_cfg_dir / "model_weights.pt"),
                    "model_config.json": compute_sha256(bad_cfg_dir / "model_config.json"),
                }
            }),
            encoding="utf-8",
        )
        app2 = create_app(cfg)
        p1b, p2b, p3b = self._mock_clearml_registry(neural_dir=bad_cfg_dir)
        with p1b, p2b, p3b:
            app2.state.inference_manager.initialize(cfg)

        code2, body2 = asyncio.run(_asgi_request(app2, "GET", "/health/ready"))
        self.assertEqual(code2, 503)
        self.assertIn("Unsupported model architecture", body2["error"])

    def test_11_missing_adaptation_dataset(self):
        missing_path = self.root / "nonexistent_elements.parquet"
        cfg = self._base_settings(ELEMENTS_PARQUET_PATH=str(missing_path))
        app = create_app(cfg)
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3:
            app.state.inference_manager.initialize(cfg)

        code, body = asyncio.run(_asgi_request(app, "GET", "/health/ready"))
        self.assertEqual(code, 503)
        self.assertFalse(body["data_dependency_ready"])
        self.assertIn("elements.parquet", body["error"])

    def test_12_inference_error_handling(self):
        cfg = self._base_settings()
        app = create_app(cfg)

        # Request before ready returns 503 without stack trace
        code_503, err_503 = asyncio.run(
            _asgi_request(app, "POST", "/v1/layouts/generate", self._sample_generate_payload("neural"))
        )
        self.assertEqual(code_503, 503)
        self.assertNotIn("Traceback", json.dumps(err_503))

        # Initialize, then simulate an unhandled internal failure
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3:
            app.state.inference_manager.initialize(cfg)

        with patch.object(app.state.inference_manager, "generate", side_effect=RuntimeError("secret_key_leak_check")):
            code_500, err_500 = asyncio.run(
                _asgi_request(app, "POST", "/v1/layouts/generate", self._sample_generate_payload("neural"))
            )
            self.assertEqual(code_500, 500)
            self.assertEqual(err_500["detail"], "Internal error during layout generation.")

    def test_13_concurrent_mixed_orientation_inference(self):
        from concurrent.futures import ThreadPoolExecutor

        cfg = self._base_settings()
        app = create_app(cfg)
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3:
            app.state.inference_manager.initialize(cfg)

        def run_one(idx: int):
            engine = "adaptation" if idx % 2 == 0 else "neural"
            orient = "landscape" if idx % 3 != 0 else "portrait"
            payload = self._sample_generate_payload(engine)
            payload["request"]["orientation"] = orient
            code, res = asyncio.run(_asgi_request(app, "POST", "/v1/layouts/generate", payload))
            return code, orient, engine, res["layout"]["width"], res["layout"]["height"]

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(run_one, range(8)))

        for code, orient, engine, w, h in results:
            self.assertEqual(code, 200)
            if orient == "landscape":
                self.assertEqual((w, h), (1000, 700))
            else:
                self.assertEqual((w, h), (700, 1000))

    def test_14_elements_parquet_checksum_mismatch(self):
        cfg = self._base_settings(ELEMENTS_PARQUET_SHA256="0" * 64)
        app = create_app(cfg)
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3:
            app.state.inference_manager.initialize(cfg)

        code, body = asyncio.run(_asgi_request(app, "GET", "/health/ready"))
        self.assertEqual(code, 503)
        self.assertFalse(body["data_dependency_ready"])
        self.assertIn("Checksum mismatch", body["error"])

    def _s3_settings(self, cache_path: Path, **overrides) -> Settings:
        from inference_service.app.services.data_loader import _compute_file_sha256

        sha = _compute_file_sha256(self.elements_path)
        params = {
            "ELEMENTS_PARQUET_PATH": str(cache_path),
            "ELEMENTS_PARQUET_SHA256": sha,
            "AWS_ACCESS_KEY_ID": "AKIA_TEST_KEY",
            "AWS_SECRET_ACCESS_KEY": "aws_super_secret_value",
            "AWS_DEFAULT_REGION": "us-east-1",
            "S3_BUCKET_NAME": "infographix-private-data",
            "S3_OBJECT_KEY": "features/v1/elements.parquet",
        }
        params.update(overrides)
        return self._base_settings(**params)

    def test_15_s3_successful_download_and_startup(self):
        import shutil

        cache_path = self.root / "s3_cache" / "elements.parquet"
        cfg = self._s3_settings(cache_path)

        mock_s3 = MagicMock()

        def fake_download(bucket, key, dest_file):
            self.assertEqual(bucket, "infographix-private-data")
            self.assertEqual(key, "features/v1/elements.parquet")
            # Ensure download targets a temp file inside cache directory, not dest directly
            self.assertNotEqual(Path(dest_file), cache_path)
            self.assertEqual(Path(dest_file).parent, cache_path.parent)
            shutil.copy2(self.elements_path, dest_file)

        mock_s3.download_file.side_effect = fake_download

        app = create_app(cfg)
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3, patch(
            "inference_service.app.services.data_loader.boto3.client", return_value=mock_s3
        ) as mock_boto:
            app.state.inference_manager.initialize(cfg)

        mock_boto.assert_called_once()
        _, kwargs = mock_boto.call_args
        self.assertEqual(kwargs["aws_access_key_id"], "AKIA_TEST_KEY")
        self.assertEqual(kwargs["aws_secret_access_key"], "aws_super_secret_value")
        self.assertEqual(kwargs["region_name"], "us-east-1")

        self.assertTrue(cache_path.exists())
        self.assertEqual(list(cache_path.parent.glob("*.tmp")), [])
        self.assertEqual(app.state.inference_manager.data_status.source, "s3")

        code, ready_body = asyncio.run(_asgi_request(app, "GET", "/health/ready"))
        self.assertEqual(code, 200)
        self.assertTrue(ready_body["data_dependency_ready"])

    def test_16_s3_missing_credentials_and_incomplete_config(self):
        from inference_service.app.services.data_loader import load_elements_dataframe

        # Missing AWS credentials (even if ELEMENTS_PARQUET_URL is set, must NOT fall back)
        cache_path = self.root / "s3_missing_creds" / "elements.parquet"
        cfg = self._s3_settings(
            cache_path,
            AWS_ACCESS_KEY_ID="",
            AWS_SECRET_ACCESS_KEY="",
            ELEMENTS_PARQUET_URL="https://example.com/should_not_fallback.parquet",
        )
        with patch("inference_service.app.services.data_loader._download_from_url") as mock_url:
            with self.assertRaises(RuntimeError) as ctx:
                load_elements_dataframe(cfg)
            self.assertIn("AWS S3 credentials missing", str(ctx.exception))
            mock_url.assert_not_called()

        # Incomplete S3 config (bucket set without key) must fail clearly without fallback
        cfg_partial = self._s3_settings(
            cache_path,
            S3_OBJECT_KEY="",
            ELEMENTS_PARQUET_URL="https://example.com/should_not_fallback.parquet",
        )
        with patch("inference_service.app.services.data_loader._download_from_url") as mock_url:
            with self.assertRaises(ValueError) as ctx:
                load_elements_dataframe(cfg_partial)
            self.assertIn("S3_BUCKET_NAME and S3_OBJECT_KEY", str(ctx.exception))
            mock_url.assert_not_called()

    def test_17_s3_access_denied(self):
        from botocore.exceptions import ClientError

        cache_path = self.root / "s3_403" / "elements.parquet"
        cfg = self._s3_settings(
            cache_path,
            ELEMENTS_PARQUET_URL="https://example.com/should_not_fallback.parquet",
        )
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
            "GetObject",
        )

        app = create_app(cfg)
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3, patch(
            "inference_service.app.services.data_loader.boto3.client", return_value=mock_s3
        ), patch("inference_service.app.services.data_loader._download_from_url") as mock_url:
            app.state.inference_manager.initialize(cfg)
            mock_url.assert_not_called()

        code, body = asyncio.run(_asgi_request(app, "GET", "/health/ready"))
        self.assertEqual(code, 503)
        self.assertIn("Access denied", body["error"])
        self.assertNotIn("aws_super_secret_value", body["error"])
        self.assertFalse(cache_path.exists())

    def test_18_s3_missing_object(self):
        from botocore.exceptions import ClientError

        cache_path = self.root / "s3_404" / "elements.parquet"
        cfg = self._s3_settings(cache_path)
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
            "GetObject",
        )

        app = create_app(cfg)
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3, patch(
            "inference_service.app.services.data_loader.boto3.client", return_value=mock_s3
        ):
            app.state.inference_manager.initialize(cfg)

        code, body = asyncio.run(_asgi_request(app, "GET", "/health/ready"))
        self.assertEqual(code, 503)
        self.assertIn("S3 object not found", body["error"])
        self.assertFalse(cache_path.exists())

    def test_19_s3_network_failure(self):
        from botocore.exceptions import EndpointConnectionError

        cache_path = self.root / "s3_net" / "elements.parquet"
        cfg = self._s3_settings(cache_path)
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = EndpointConnectionError(
            endpoint_url="https://s3.us-east-1.amazonaws.com"
        )

        app = create_app(cfg)
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3, patch(
            "inference_service.app.services.data_loader.boto3.client", return_value=mock_s3
        ):
            app.state.inference_manager.initialize(cfg)

        code, body = asyncio.run(_asgi_request(app, "GET", "/health/ready"))
        self.assertEqual(code, 503)
        self.assertIn("Network or transport error", body["error"])
        self.assertFalse(cache_path.exists())

    def test_20_s3_checksum_mismatch_leaves_cache_absent(self):
        import shutil

        cache_path = self.root / "s3_bad_sha" / "elements.parquet"
        cfg = self._s3_settings(cache_path, ELEMENTS_PARQUET_SHA256="f" * 64)
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = lambda b, k, dest: shutil.copy2(self.elements_path, dest)

        app = create_app(cfg)
        p1, p2, p3 = self._mock_clearml_registry()
        with p1, p2, p3, patch(
            "inference_service.app.services.data_loader.boto3.client", return_value=mock_s3
        ):
            app.state.inference_manager.initialize(cfg)

        code, body = asyncio.run(_asgi_request(app, "GET", "/health/ready"))
        self.assertEqual(code, 503)
        self.assertIn("Checksum mismatch", body["error"])
        # Corrupt download must NEVER be promoted into final cache path
        self.assertFalse(cache_path.exists())
        self.assertEqual(list(cache_path.parent.glob("*.tmp")), [])

    def test_21_s3_atomic_cache_replacement(self):
        import shutil
        from inference_service.app.services.data_loader import _compute_file_sha256, _download_from_s3

        cache_dir = self.root / "s3_atomic"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "elements.parquet"
        cache_path.write_bytes(b"previous-valid-cache-content")

        # 1. If S3 download fails checksum, existing cache_path remains untouched
        cfg_bad = self._s3_settings(cache_path, ELEMENTS_PARQUET_SHA256="a" * 64)
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = lambda b, k, dest: shutil.copy2(self.elements_path, dest)
        with patch("inference_service.app.services.data_loader.boto3.client", return_value=mock_s3):
            with self.assertRaises(ValueError):
                _download_from_s3(cfg_bad, cache_path)
        self.assertEqual(cache_path.read_bytes(), b"previous-valid-cache-content")
        self.assertEqual(list(cache_dir.glob("*.tmp")), [])

        # 2. If S3 download passes checksum, os.replace atomically swaps new file into cache_path
        expected_sha = _compute_file_sha256(self.elements_path)
        cfg_good = self._s3_settings(cache_path, ELEMENTS_PARQUET_SHA256=expected_sha)
        with patch("inference_service.app.services.data_loader.boto3.client", return_value=mock_s3):
            with patch(
                "inference_service.app.services.data_loader.os.replace", wraps=os.replace
            ) as mock_replace:
                actual_sha = _download_from_s3(cfg_good, cache_path)
        self.assertEqual(actual_sha, expected_sha)
        mock_replace.assert_called_once()
        self.assertEqual(_compute_file_sha256(cache_path), expected_sha)
        self.assertEqual(list(cache_dir.glob("*.tmp")), [])

    def test_22_existing_cached_file_reuse_and_local_path_loading(self):
        from inference_service.app.services.data_loader import load_elements_dataframe

        # Existing cached file is reused without invoking S3 even when S3 vars are configured
        cfg = self._s3_settings(self.elements_path)
        with patch("inference_service.app.services.data_loader.boto3.client") as mock_boto:
            df, status = load_elements_dataframe(cfg)
        mock_boto.assert_not_called()
        self.assertEqual(status.source, "local_cache")
        self.assertGreater(len(df), 0)

        # Local-path loading without S3 configured still works identically
        cfg_local = self._base_settings(ELEMENTS_PARQUET_PATH=str(self.elements_path))
        df_local, status_local = load_elements_dataframe(cfg_local)
        self.assertEqual(status_local.source, "local_cache")
        self.assertEqual(len(df_local), len(df))


if __name__ == "__main__":
    unittest.main()
