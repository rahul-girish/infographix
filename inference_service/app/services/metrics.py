"""Prometheus metrics instrumentation for the Infographix Inference Service.

Exposes service health, HTTP latency/throughput, inference engine metrics,
input request feature distributions (for drift monitoring), retriever OOD
distance, and layout geometry quality metrics.
"""

from __future__ import annotations

from typing import Any

import psutil
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class InferenceMetrics:
    """Encapsulates all Prometheus collectors in a dedicated registry."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self._process = psutil.Process()

        # 1. HTTP layer metrics
        self.http_requests_total = Counter(
            "infographix_http_requests_total",
            "Total HTTP requests handled by the inference service",
            ["method", "path", "status_code"],
            registry=self.registry,
        )
        self.http_request_duration_seconds = Histogram(
            "infographix_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "path"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )

        # 2. Service & model lifecycle metrics
        self.service_ready = Gauge(
            "infographix_service_ready",
            "Whether the inference service is ready to serve requests (1=ready, 0=not ready)",
            registry=self.registry,
        )
        self.process_rss_bytes = Gauge(
            "infographix_process_rss_bytes",
            "Resident set size (RSS) memory of the inference process in bytes",
            registry=self.registry,
        )
        self.active_model_info = Gauge(
            "infographix_active_model_info",
            "Active ClearML model versions loaded for inference (always 1)",
            ["engine", "model_name", "model_version", "retriever_version"],
            registry=self.registry,
        )

        # 3. Generation throughput, errors, and latency
        self.generate_requests_total = Counter(
            "infographix_generate_requests_total",
            "Total layout generation requests by engine and outcome",
            ["engine", "status"],
            registry=self.registry,
        )
        self.generate_latency_seconds = Histogram(
            "infographix_generate_latency_seconds",
            "End-to-end layout generation latency in seconds",
            ["engine"],
            buckets=(0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
            registry=self.registry,
        )

        # 4. Input feature distributions (foundation for online data drift monitoring)
        self.request_element_count = Histogram(
            "infographix_request_element_count",
            "Number of content elements per incoming poster request",
            buckets=(5, 10, 15, 20, 25, 30, 40, 50, 75),
            registry=self.registry,
        )
        self.request_total_words = Histogram(
            "infographix_request_total_words",
            "Total word count across blocks in an incoming poster request",
            buckets=(50, 150, 300, 500, 800, 1200, 2000, 4000),
            registry=self.registry,
        )
        self.request_figure_count = Histogram(
            "infographix_request_figure_count",
            "Number of figure/visual blocks per incoming poster request",
            buckets=(0, 1, 2, 3, 4, 6, 8, 12),
            registry=self.registry,
        )
        self.request_aspect_ratio = Histogram(
            "infographix_request_aspect_ratio",
            "Requested poster canvas aspect ratio (width / height)",
            buckets=(0.5, 0.65, 0.75, 0.9, 1.1, 1.33, 1.5, 1.8, 2.2),
            registry=self.registry,
        )

        # 5. Model & layout output quality metrics (ground-truth-free)
        self.retriever_top1_distance = Histogram(
            "infographix_retriever_top1_distance",
            "Top-1 retrieved template distance (proxy for out-of-distribution input drift)",
            buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0),
            registry=self.registry,
        )
        self.retriever_top1_distance_last = Gauge(
            "infographix_retriever_top1_distance_last",
            "Most recent top-1 retrieved template distance",
            registry=self.registry,
        )
        self.layout_overlap_area = Histogram(
            "infographix_layout_overlap_area",
            "Validated pairwise overlap area of generated layout",
            ["engine"],
            buckets=(0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
            registry=self.registry,
        )
        self.layout_overlap_area_last = Gauge(
            "infographix_layout_overlap_area_last",
            "Most recent validated pairwise overlap area",
            ["engine"],
            registry=self.registry,
        )
        self.layout_raw_overflow_rate = Histogram(
            "infographix_layout_raw_overflow_rate",
            "Fraction of elements overflowing canvas bounds before clamping",
            ["engine"],
            buckets=(0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
            registry=self.registry,
        )
        self.layout_min_size_violation_rate = Histogram(
            "infographix_layout_min_size_violation_rate",
            "Fraction of elements below minimum dimension threshold",
            ["engine"],
            buckets=(0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
            registry=self.registry,
        )
        self.layout_clamped_boxes_total = Counter(
            "infographix_layout_clamped_boxes_total",
            "Cumulative count of boxes clamped during geometry validation",
            ["engine"],
            registry=self.registry,
        )
        self.layout_predicted_columns = Histogram(
            "infographix_layout_predicted_columns",
            "Predicted/used column count for generated posters",
            ["engine"],
            buckets=(1, 2, 3, 4, 5),
            registry=self.registry,
        )

    def record_http_request(self, method: str, path: str, status_code: int, duration_s: float) -> None:
        self.http_requests_total.labels(
            method=method.upper(),
            path=path,
            status_code=str(status_code),
        ).inc()
        self.http_request_duration_seconds.labels(
            method=method.upper(),
            path=path,
        ).observe(max(0.0, float(duration_s)))

    def record_generation_success(self, result: dict[str, Any], duration_s: float) -> None:
        engine = str(result.get("engine", "unknown"))
        self.generate_requests_total.labels(engine=engine, status="success").inc()
        self.generate_latency_seconds.labels(engine=engine).observe(max(0.0, float(duration_s)))

        self.active_model_info.labels(
            engine=engine,
            model_name=str(result.get("model_name", "unknown")),
            model_version=str(result.get("model_version", "unknown")),
            retriever_version=str(result.get("retriever_version", "unknown")),
        ).set(1.0)

        # Input feature summary
        req_feats = result.get("request_features") or {}
        summary = req_feats.get("summary") or {}
        blocks = req_feats.get("blocks") or []

        num_elements = len(blocks) if blocks else int(summary.get("total_element_count", 0))
        if num_elements > 0:
            self.request_element_count.observe(float(num_elements))

        total_words = sum(int(b.get("word_count") or 0) for b in blocks)
        self.request_total_words.observe(float(total_words))

        fig_count = sum(
            1 for b in blocks if str(b.get("semantic_role", "")).lower() in ("figure", "logo")
        )
        self.request_figure_count.observe(float(fig_count))

        layout_doc = result.get("layout") or {}
        w = float(layout_doc.get("width") or 0.0)
        h = float(layout_doc.get("height") or 0.0)
        if w > 0 and h > 0:
            self.request_aspect_ratio.observe(w / h)
        cols_used = layout_doc.get("columns")
        if cols_used is not None:
            self.layout_predicted_columns.labels(engine=engine).observe(float(cols_used))

        # Retriever distance (OOD signal)
        cands = result.get("candidate_templates") or []
        if cands and isinstance(cands[0], dict):
            dist = cands[0].get("distance")
            if dist is not None:
                d_val = float(dist)
                self.retriever_top1_distance.observe(d_val)
                self.retriever_top1_distance_last.set(d_val)

        # Geometry quality metrics
        gm = result.get("generation_metrics") or {}
        if "validated_overlap_area" in gm:
            ov = float(gm["validated_overlap_area"])
            self.layout_overlap_area.labels(engine=engine).observe(ov)
            self.layout_overlap_area_last.labels(engine=engine).set(ov)
        if "raw_overflow_rate" in gm:
            self.layout_raw_overflow_rate.labels(engine=engine).observe(float(gm["raw_overflow_rate"]))
        if "min_size_violation_rate" in gm:
            self.layout_min_size_violation_rate.labels(engine=engine).observe(
                float(gm["min_size_violation_rate"])
            )
        clamped = int(gm.get("clamped_boxes_count", 0))
        if clamped > 0:
            self.layout_clamped_boxes_total.labels(engine=engine).inc(clamped)

    def record_generation_error(self, engine: str, error_type: str, duration_s: float) -> None:
        self.generate_requests_total.labels(engine=engine or "unknown", status=error_type).inc()
        self.generate_latency_seconds.labels(engine=engine or "unknown").observe(
            max(0.0, float(duration_s))
        )

    def update_runtime_gauges(self, is_ready: bool) -> None:
        self.service_ready.set(1.0 if is_ready else 0.0)
        try:
            self.process_rss_bytes.set(float(self._process.memory_info().rss))
        except Exception:
            pass

    def render_prometheus(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
