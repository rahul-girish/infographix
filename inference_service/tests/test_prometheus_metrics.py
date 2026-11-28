"""Unit and integration tests for Prometheus metrics in inference_service."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from inference_service.app.config import Settings
from inference_service.app.main import create_app
from inference_service.app.services.model_loader import LoadedModelMetadata
from src.neural_layout.layout_model import TemplateConditionedLayoutModel


class _DummyRetriever:
    def __init__(self) -> None:
        self.train_posters_ = pd.DataFrame(
            [
                {
                    "poster_id": 101,
                    "orientation": "landscape",
                    "estimated_column_count": 3,
                    "count_title": 1,
                    "count_author_info": 1,
                    "count_section": 3,
                    "count_text": 3,
                    "count_figure": 1,
                    "count_caption": 1,
                    "count_list": 0,
                    "count_table": 0,
                }
            ]
        )

    def retrieve_for_request(self, req, top_k: int = 5, diversify: bool = True):
        return pd.DataFrame(
            [
                {
                    "poster_id": 101,
                    "orientation": "landscape",
                    "estimated_column_count": 3,
                    "relevance_score": 0.875,
                }
            ]
        )


def _sample_generate_payload() -> dict:
    return {
        "engine": "neural",
        "request": {
            "title": "Neural Poster Layout Generation",
            "authors": "A. Researcher, B. Scientist",
            "affiliations": "Infographix Lab",
            "orientation": "landscape",
            "preferredColumns": 3,
            "blocks": [
                {
                    "id": "sec1",
                    "type": "section",
                    "heading": "Introduction",
                    "text": "Scientific posters require structured multi-column layouts.",
                },
                {
                    "id": "fig1",
                    "type": "figure",
                    "heading": "Architecture Diagram",
                    "aspectRatio": 1.5,
                    "caption": "Figure 1: Model overview.",
                },
            ],
        },
    }


class TestPrometheusMetrics(unittest.TestCase):
    def _build_ready_app(self, auth_token: str | None = None):
        settings = Settings(
            SKIP_STARTUP_INIT=True,
            METRICS_ENABLED=True,
            METRICS_AUTH_TOKEN=auth_token,
        )
        app = create_app(settings)
        mgr = app.state.inference_manager

        model = TemplateConditionedLayoutModel(
            d_model=32,
            num_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            ffn_dim=64,
            bbox_parameterization="bounded",
            use_anchor_conditioning=True,
        ).eval()

        ret_info = LoadedModelMetadata(
            role="retriever",
            model_name="poster_layout_retriever-v2.1.0",
            model_family="poster_layout_retriever",
            clearml_model_id="ret_test_id",
            semantic_version="2.1.0",
            architecture="LayoutRetriever",
            checkpoint_sha256="abc",
            lifecycle_status="production",
            dataset_identifier="SciPostLayout v2",
            supported_engines=["neural", "adaptation"],
        )
        gen_info = LoadedModelMetadata(
            role="neural_generator",
            model_name="poster_layout_generator-v0.2.0",
            model_family="poster_layout_generator",
            clearml_model_id="gen_test_id",
            semantic_version="0.2.0",
            architecture="TemplateConditionedLayoutModel",
            checkpoint_sha256="def",
            lifecycle_status="candidate",
            dataset_identifier="SciPostLayout v2",
            supported_engines=["neural"],
        )

        mgr.retriever = _DummyRetriever()
        mgr.retriever_info = ret_info
        mgr.neural_model = model
        mgr.neural_info = gen_info
        mgr.ready = True
        app.state.metrics.set_readiness(ready=True, retriever_info=ret_info, neural_info=gen_info)
        return app

    def test_metrics_endpoint_exposes_service_and_ml_metrics(self) -> None:
        app = self._build_ready_app()
        with TestClient(app) as client:
            r_health = client.get("/health/live")
            self.assertEqual(r_health.status_code, 200)

            r_gen = client.post("/v1/layouts/generate", json=_sample_generate_payload())
            self.assertEqual(r_gen.status_code, 200)

            r_metrics = client.get("/metrics")
            self.assertEqual(r_metrics.status_code, 200)
            self.assertIn("text/plain", r_metrics.headers["content-type"])
            body = r_metrics.text

            # Service & model provenance gauges
            self.assertIn("infographix_service_ready 1.0", body)
            self.assertIn('infographix_model_info{clearml_model_id="gen_test_id"', body)
            self.assertIn('infographix_model_info{clearml_model_id="ret_test_id"', body)
            self.assertIn("infographix_process_rss_mib ", body)

            # HTTP middleware counters & latency histograms
            self.assertIn(
                'infographix_http_requests_total{method="POST",path="/v1/layouts/generate",status_code="200"} 1.0',
                body,
            )
            self.assertIn(
                'infographix_http_requests_total{method="GET",path="/health/live",status_code="200"} 1.0',
                body,
            )

            # ML inference, retrieval, input feature, and output geometry metrics
            self.assertIn(
                'infographix_inference_requests_total{engine="neural",orientation="landscape",status="success"} 1.0',
                body,
            )
            self.assertIn('infographix_inference_duration_seconds_count{engine="neural"} 1.0', body)
            self.assertIn("infographix_retrieval_top1_relevance_count 1.0", body)
            self.assertIn('infographix_input_element_count_count{orientation="landscape"} 1.0', body)
            self.assertIn("infographix_input_total_words_count 1.0", body)
            self.assertIn("infographix_input_figure_count_count 1.0", body)
            self.assertIn('infographix_predicted_columns_count{engine="neural"} 1.0', body)
            self.assertIn('infographix_layout_overlap_area_count{engine="neural"} 1.0', body)
            self.assertIn('infographix_layout_raw_overflow_rate_count{engine="neural"} 1.0', body)

    def test_metrics_records_not_ready_and_error_outcomes(self) -> None:
        settings = Settings(SKIP_STARTUP_INIT=True, METRICS_ENABLED=True)
        app = create_app(settings)
        with TestClient(app) as client:
            r = client.post("/v1/layouts/generate", json=_sample_generate_payload())
            self.assertEqual(r.status_code, 503)

            body = client.get("/metrics").text
            self.assertIn("infographix_service_ready 0.0", body)
            self.assertIn(
                'infographix_inference_requests_total{engine="neural",orientation="landscape",status="not_ready"} 1.0',
                body,
            )

    def test_metrics_bearer_token_protection(self) -> None:
        app = self._build_ready_app(auth_token="super-secret-token")
        with TestClient(app) as client:
            r_unauth = client.get("/metrics")
            self.assertEqual(r_unauth.status_code, 401)

            r_wrong = client.get("/metrics", headers={"Authorization": "Bearer wrong"})
            self.assertEqual(r_wrong.status_code, 401)

            r_ok = client.get("/metrics", headers={"Authorization": "Bearer super-secret-token"})
            self.assertEqual(r_ok.status_code, 200)
            self.assertIn("infographix_service_ready 1.0", r_ok.text)

    def test_online_drift_detection_and_grafana_otlp_payload(self) -> None:
        from inference_service.app.services.grafana_pusher import (
            GrafanaCloudPusher,
            build_otlp_metrics_payload,
        )

        settings = Settings(
            SKIP_STARTUP_INIT=True,
            METRICS_ENABLED=True,
            DRIFT_WINDOW_SIZE=20,
            DRIFT_MIN_SAMPLES=5,
            DRIFT_PSI_THRESHOLD=0.20,
        )
        app = create_app(settings)
        mgr = app.state.inference_manager
        mgr.retriever = _DummyRetriever()
        mgr.retriever_info = LoadedModelMetadata(
            role="retriever",
            model_name="poster_layout_retriever-v2.1.0",
            model_family="poster_layout_retriever",
            clearml_model_id="ret_test_id",
            semantic_version="2.1.0",
            architecture="LayoutRetriever",
            checkpoint_sha256="abc",
            lifecycle_status="production",
            dataset_identifier="SciPostLayout v2",
            supported_engines=["neural"],
        )
        mgr.neural_model = TemplateConditionedLayoutModel(
            d_model=32, num_heads=4, encoder_layers=1, decoder_layers=1, ffn_dim=64,
            bbox_parameterization="bounded", use_anchor_conditioning=True,
        ).eval()
        mgr.neural_info = LoadedModelMetadata(
            role="neural_generator",
            model_name="poster_layout_generator-v2.2.0",
            model_family="poster_layout_generator",
            clearml_model_id="gen_test_id",
            semantic_version="2.2.0",
            architecture="TemplateConditionedLayoutModel",
            checkpoint_sha256="def",
            lifecycle_status="candidate",
            dataset_identifier="SciPostLayout v2",
            supported_engines=["neural"],
        )
        mgr.ready = True
        app.state.metrics.set_readiness(ready=True, retriever_info=mgr.retriever_info, neural_info=mgr.neural_info)

        # Send 6 heavily drifted requests (extreme block/figure count & aspect ratio)
        drifted_payload = {
            "engine": "neural",
            "request": {
                "title": "Extreme OOD Poster",
                "authors": "A. Researcher",
                "orientation": "landscape",
                "preferredColumns": 4,
                "blocks": [
                    {"id": f"s{i}", "type": "section", "heading": f"Sec {i}", "text": "word " * 180}
                    for i in range(12)
                ] + [
                    {"id": f"f{j}", "type": "figure", "heading": f"Fig {j}", "aspectRatio": 3.8, "caption": "cap " * 12}
                    for j in range(12)
                ],
            },
        }
        with TestClient(app) as client:
            for _ in range(6):
                resp = client.post("/v1/layouts/generate", json=drifted_payload)
                self.assertEqual(resp.status_code, 200)

            drift_json = client.get("/v1/monitoring/drift").json()
            self.assertTrue(drift_json["ready"])
            self.assertEqual(drift_json["window_samples"], 6)
            self.assertGreaterEqual(drift_json["overall_psi"], 0.20)
            self.assertTrue(drift_json["alert_active"])

            metrics_text = client.get("/metrics").text
            self.assertIn("infographix_drift_window_samples 6.0", metrics_text)
            self.assertIn("infographix_drift_alert_active 1.0", metrics_text)
            self.assertIn('infographix_feature_drift_psi{feature="total_element_count"}', metrics_text)

            # Verify OTLP JSON payload + mocked GrafanaCloudPusher
            otlp = build_otlp_metrics_payload(app.state.metrics)
            otlp_names = {
                m["name"]
                for m in otlp["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
            }
            self.assertIn("infographix_drift_overall_psi", otlp_names)
            self.assertIn("infographix_service_ready", otlp_names)

            pusher = GrafanaCloudPusher(
                metrics=app.state.metrics,
                otlp_endpoint="https://otlp-gateway-prod-us-east-0.grafana.net/otlp",
                instance_id="12345",
                api_key="glc_test_key",
            )
            fake_resp = SimpleNamespace(status_code=200, text="ok")
            with patch("inference_service.app.services.grafana_pusher.requests.post", return_value=fake_resp) as m_post:
                self.assertTrue(pusher.push_once())
                self.assertEqual(pusher.push_count, 1)
                self.assertTrue(pusher.last_push_ok)
                self.assertEqual(m_post.call_args[0][0], "https://otlp-gateway-prod-us-east-0.grafana.net/otlp/v1/metrics")
                self.assertEqual(m_post.call_args[1]["auth"], ("12345", "glc_test_key"))


if __name__ == "__main__":
    unittest.main()
