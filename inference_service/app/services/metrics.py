"""Prometheus instrumentation for the Infographix Inference Service.

Exposes service health, HTTP request telemetry, active ClearML model provenance,
memory usage, input feature distributions, retrieval relevance, and output layout
geometry quality metrics in standard Prometheus text exposition format.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from inference_service.app.services.memory_profiler import get_rss_mib


class ServiceMetrics:
    """Encapsulates all Prometheus collectors for an Infographix app instance."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)

        # --- HTTP Telemetry ---
        self.http_requests_total = Counter(
            "infographix_http_requests_total",
            "Total HTTP requests handled by the inference service.",
            labelnames=("method", "path", "status_code"),
            registry=self.registry,
        )
        self.http_request_duration_seconds = Histogram(
            "infographix_http_request_duration_seconds",
            "HTTP request latency in seconds.",
            labelnames=("method", "path"),
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )
        self.http_requests_in_progress = Gauge(
            "infographix_http_requests_in_progress",
            "Number of HTTP requests currently being processed.",
            labelnames=("method",),
            registry=self.registry,
        )

        # --- Service & Model Provenance ---
        self.service_ready = Gauge(
            "infographix_service_ready",
            "1 if the inference service has initialized models and is ready, 0 otherwise.",
            registry=self.registry,
        )
        self.model_info = Gauge(
            "infographix_model_info",
            "Active ClearML model metadata loaded in the inference service (value is always 1).",
            labelnames=("role", "model_name", "semantic_version", "clearml_model_id"),
            registry=self.registry,
        )
        self.process_rss_mib = Gauge(
            "infographix_process_rss_mib",
            "Current inference service process resident set size (RSS) in MiB.",
            registry=self.registry,
        )

        # --- Inference & Layout Quality ---
        self.inference_requests_total = Counter(
            "infographix_inference_requests_total",
            "Total layout generation attempts by engine, orientation, and outcome status.",
            labelnames=("engine", "orientation", "status"),
            registry=self.registry,
        )
        self.inference_duration_seconds = Histogram(
            "infographix_inference_duration_seconds",
            "End-to-end layout generation latency in seconds.",
            labelnames=("engine",),
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=self.registry,
        )
        self.retrieval_top1_relevance = Histogram(
            "infographix_retrieval_top1_relevance",
            "Top-1 LayoutRetriever relevance score for incoming poster requests.",
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry,
        )

        # --- Input Feature Distributions (for monitoring & drift) ---
        self.input_element_count = Histogram(
            "infographix_input_element_count",
            "Number of content blocks in incoming poster generation requests.",
            labelnames=("orientation",),
            buckets=(2, 5, 8, 12, 16, 20, 25, 30, 40, 60),
            registry=self.registry,
        )
        self.input_total_words = Histogram(
            "infographix_input_total_words",
            "Total word count across all text blocks in incoming requests.",
            buckets=(10, 50, 100, 200, 350, 500, 750, 1000, 1500, 2500),
            registry=self.registry,
        )
        self.input_figure_count = Histogram(
            "infographix_input_figure_count",
            "Number of figure blocks in incoming poster generation requests.",
            buckets=(0, 1, 2, 3, 4, 5, 6, 8, 12),
            registry=self.registry,
        )

        # --- Output Layout Geometry Quality ---
        self.predicted_columns = Histogram(
            "infographix_predicted_columns",
            "Number of poster columns predicted/used in generated layouts.",
            labelnames=("engine",),
            buckets=(1, 2, 3, 4, 5),
            registry=self.registry,
        )
        self.layout_overlap_area = Histogram(
            "infographix_layout_overlap_area",
            "Validated pairwise overlap area of generated poster layouts.",
            labelnames=("engine",),
            buckets=(0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
            registry=self.registry,
        )
        self.layout_raw_overflow_rate = Histogram(
            "infographix_layout_raw_overflow_rate",
            "Fraction of raw predicted boxes overflowing canvas bounds prior to clamping.",
            labelnames=("engine",),
            buckets=(0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
            registry=self.registry,
        )
        self.layout_clamped_boxes_total = Counter(
            "infographix_layout_clamped_boxes_total",
            "Total number of predicted boxes clamped to canvas bounds.",
            labelnames=("engine",),
            registry=self.registry,
        )

        self.service_ready.set(0)
        self.update_process_memory()

    def update_process_memory(self) -> float:
        rss = float(get_rss_mib())
        self.process_rss_mib.set(rss)
        return rss

    def set_readiness(
        self,
        ready: bool,
        retriever_info: Any | None = None,
        neural_info: Any | None = None,
    ) -> None:
        self.service_ready.set(1.0 if ready else 0.0)
        self.model_info.clear()
        for info in (retriever_info, neural_info):
            if info is None:
                continue
            self.model_info.labels(
                role=str(getattr(info, "role", "unknown")),
                model_name=str(getattr(info, "model_name", "unknown")),
                semantic_version=str(getattr(info, "semantic_version", "unknown")),
                clearml_model_id=str(getattr(info, "clearml_model_id", "unknown")),
            ).set(1.0)
        self.update_process_memory()

    def record_inference_success(
        self,
        engine: str,
        orientation: str,
        duration_seconds: float,
        num_elements: int,
        total_words: int,
        figure_count: int,
        columns_used: int,
        top1_relevance: float | None,
        overlap_area: float,
        raw_overflow_rate: float,
        clamped_boxes: int,
    ) -> None:
        self.inference_requests_total.labels(
            engine=engine,
            orientation=orientation,
            status="success",
        ).inc()
        self.inference_duration_seconds.labels(engine=engine).observe(max(0.0, duration_seconds))
        self.input_element_count.labels(orientation=orientation).observe(max(0, num_elements))
        self.input_total_words.observe(max(0, total_words))
        self.input_figure_count.observe(max(0, figure_count))
        self.predicted_columns.labels(engine=engine).observe(max(1, columns_used))
        if top1_relevance is not None:
            self.retrieval_top1_relevance.observe(float(top1_relevance))
        self.layout_overlap_area.labels(engine=engine).observe(max(0.0, float(overlap_area)))
        self.layout_raw_overflow_rate.labels(engine=engine).observe(max(0.0, float(raw_overflow_rate)))
        if clamped_boxes > 0:
            self.layout_clamped_boxes_total.labels(engine=engine).inc(int(clamped_boxes))
        self.update_process_memory()

    def record_inference_error(
        self,
        engine: str,
        orientation: str,
        status: str,
    ) -> None:
        self.inference_requests_total.labels(
            engine=engine or "unknown",
            orientation=orientation or "unknown",
            status=status,
        ).inc()
        self.update_process_memory()

    def render_prometheus(self) -> tuple[bytes, str]:
        """Refresh process memory gauge and render metrics in Prometheus text exposition format."""
        self.update_process_memory()
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
