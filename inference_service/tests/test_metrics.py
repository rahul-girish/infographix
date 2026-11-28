"""Tests for Prometheus /metrics endpoint and inference instrumentation."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from inference_service.app.config import Settings
from inference_service.app.main import create_app


class TestPrometheusMetrics(unittest.TestCase):
    def _make_client(self, metrics_token: str | None = None) -> tuple[TestClient, object]:
        settings = Settings(
            SKIP_STARTUP_INIT=True,
            METRICS_AUTH_TOKEN=metrics_token,
        )
        app = create_app(settings=settings)
        return TestClient(app), app

    def test_metrics_endpoint_open_and_records_http(self) -> None:
        client, _ = self._make_client(metrics_token=None)
        with client:
            r_root = client.get("/")
            self.assertEqual(r_root.status_code, 200)
            self.assertEqual(r_root.json().get("metrics"), "/metrics")

            r_m = client.get("/metrics")
            self.assertEqual(r_m.status_code, 200)
            self.assertIn("text/plain", r_m.headers.get("content-type", ""))
            body = r_m.text
            self.assertIn("infographix_service_ready 0.0", body)
            self.assertIn("infographix_process_rss_bytes", body)
            self.assertIn(
                'infographix_http_requests_total{method="GET",path="/",status_code="200"} 1.0',
                body,
            )
            # /metrics itself should not pollute HTTP request counters
            self.assertNotIn('path="/metrics"', body)

    def test_metrics_bearer_token_protection(self) -> None:
        client, _ = self._make_client(metrics_token="super-secret-token")
        with client:
            self.assertEqual(client.get("/metrics").status_code, 401)
            self.assertEqual(
                client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code,
                401,
            )
            ok = client.get("/metrics", headers={"Authorization": "Bearer super-secret-token"})
            self.assertEqual(ok.status_code, 200)
            self.assertIn("infographix_service_ready", ok.text)

    def test_generation_metrics_recorded_on_success_and_error(self) -> None:
        client, app = self._make_client(metrics_token=None)
        mock_result = {
            "engine": "neural",
            "model_name": "poster_layout_generator",
            "model_version": "0.2.0",
            "model_id": "mdl-123",
            "retriever_version": "2.1.0",
            "retriever_model_id": "ret-456",
            "retrieved_template_id": "999",
            "candidate_templates": [
                {"rank": 1, "poster_id": 999, "distance": 0.42, "similarity": 0.70}
            ],
            "generation_metrics": {
                "num_elements": 4,
                "raw_overflow_rate": 0.0,
                "clamped_boxes_count": 1,
                "min_size_violation_rate": 0.0,
                "validated_overlap_area": 0.012,
                "latency_ms": 45.0,
            },
            "request_features": {
                "summary": {"total_element_count": 4},
                "blocks": [
                    {"block_id": "b1", "semantic_role": "title", "word_count": 10},
                    {"block_id": "b2", "semantic_role": "text", "word_count": 120},
                    {"block_id": "b3", "semantic_role": "figure", "word_count": 0},
                    {"block_id": "b4", "semantic_role": "caption", "word_count": 15},
                ],
            },
            "layout": {
                "id": "doc-1",
                "title": "Test Poster",
                "width": 1920,
                "height": 1080,
                "orientation": "landscape",
                "background": "#ffffff",
                "accentColor": "#000000",
                "columns": 3,
                "elements": [],
                "updatedAt": "2026-11-28T00:00:00+00:00",
            },
        }
        with client:
            app.state.metrics.record_generation_success(mock_result, duration_s=0.045)
            app.state.metrics.record_generation_error("neural", "not_ready", duration_s=0.005)
            body = client.get("/metrics").text

            self.assertIn(
                'infographix_generate_requests_total{engine="neural",status="success"} 1.0',
                body,
            )
            self.assertIn(
                'infographix_generate_requests_total{engine="neural",status="not_ready"} 1.0',
                body,
            )
            self.assertIn("infographix_retriever_top1_distance_last 0.42", body)
            self.assertIn(
                'infographix_layout_overlap_area_last{engine="neural"} 0.012',
                body,
            )
            self.assertIn(
                'infographix_layout_clamped_boxes_total{engine="neural"} 1.0',
                body,
            )
            self.assertIn(
                'infographix_active_model_info{engine="neural",model_name="poster_layout_generator",model_version="0.2.0",retriever_version="2.1.0"} 1.0',
                body,
            )


if __name__ == "__main__":
    unittest.main()
