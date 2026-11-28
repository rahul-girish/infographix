"""Zero-dependency Grafana Cloud OTLP HTTP/JSON Metrics Pusher.

Periodically reads all Prometheus collectors from `ServiceMetrics.registry` and pushes
them to Grafana Cloud's OTLP gateway (`/v1/metrics`), where Grafana Cloud automatically
ingests them into hosted Prometheus for live dashboards and alerting.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests

from inference_service.app.services.metrics import ServiceMetrics

logger = logging.getLogger(__name__)


def _to_otlp_attributes(labels: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {"key": str(k), "value": {"stringValue": str(v)}}
        for k, v in sorted(labels.items())
    ]


def build_otlp_metrics_payload(
    metrics: ServiceMetrics,
    service_name: str = "infographix-inference",
    service_version: str = "1.0.0",
    environment: str = "production",
) -> dict[str, Any]:
    """Serialize all samples in ServiceMetrics.registry into an OTLP JSON payload."""
    metrics.update_process_memory()
    now_ns = str(time.time_ns())
    start_ns = now_ns

    otlp_metrics: list[dict[str, Any]] = []

    for family in metrics.registry.collect():
        mtype = family.type
        if mtype == "gauge":
            points = []
            for sample in family.samples:
                points.append(
                    {
                        "attributes": _to_otlp_attributes(sample.labels),
                        "timeUnixNano": now_ns,
                        "asDouble": float(sample.value),
                    }
                )
            if points:
                otlp_metrics.append(
                    {
                        "name": family.name,
                        "description": family.documentation,
                        "gauge": {"dataPoints": points},
                    }
                )

        elif mtype == "counter":
            points = []
            for sample in family.samples:
                if sample.name.endswith("_created"):
                    continue
                points.append(
                    {
                        "attributes": _to_otlp_attributes(sample.labels),
                        "startTimeUnixNano": start_ns,
                        "timeUnixNano": now_ns,
                        "asDouble": float(sample.value),
                    }
                )
            if points:
                otlp_metrics.append(
                    {
                        "name": f"{family.name}_total" if not family.name.endswith("_total") else family.name,
                        "description": family.documentation,
                        "sum": {
                            "dataPoints": points,
                            "aggregationTemporality": 2,  # CUMULATIVE
                            "isMonotonic": True,
                        },
                    }
                )

        elif mtype == "histogram":
            # Export _count, _sum, and _bucket as gauges/sums so PromQL queries match /metrics identically
            count_pts = []
            sum_pts = []
            bucket_pts = []
            for sample in family.samples:
                if sample.name.endswith("_created"):
                    continue
                if sample.name.endswith("_count"):
                    count_pts.append(
                        {
                            "attributes": _to_otlp_attributes(sample.labels),
                            "startTimeUnixNano": start_ns,
                            "timeUnixNano": now_ns,
                            "asDouble": float(sample.value),
                        }
                    )
                elif sample.name.endswith("_sum"):
                    sum_pts.append(
                        {
                            "attributes": _to_otlp_attributes(sample.labels),
                            "startTimeUnixNano": start_ns,
                            "timeUnixNano": now_ns,
                            "asDouble": float(sample.value),
                        }
                    )
                elif sample.name.endswith("_bucket"):
                    bucket_pts.append(
                        {
                            "attributes": _to_otlp_attributes(sample.labels),
                            "startTimeUnixNano": start_ns,
                            "timeUnixNano": now_ns,
                            "asDouble": float(sample.value),
                        }
                    )
            for suffix, pts in (("_count", count_pts), ("_sum", sum_pts), ("_bucket", bucket_pts)):
                if pts:
                    otlp_metrics.append(
                        {
                            "name": f"{family.name}{suffix}",
                            "description": family.documentation,
                            "sum": {
                                "dataPoints": pts,
                                "aggregationTemporality": 2,
                                "isMonotonic": True,
                            },
                        }
                    )

    return {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": _to_otlp_attributes(
                        {
                            "service.name": service_name,
                            "service.version": service_version,
                            "deployment.environment": environment,
                        }
                    )
                },
                "scopeMetrics": [
                    {
                        "scope": {"name": "infographix.prometheus_bridge", "version": service_version},
                        "metrics": otlp_metrics,
                    }
                ],
            }
        ]
    }


class GrafanaCloudPusher:
    """Background daemon that pushes Prometheus metrics to Grafana Cloud OTLP endpoint."""

    def __init__(
        self,
        metrics: ServiceMetrics,
        otlp_endpoint: str,
        instance_id: str,
        api_key: str,
        interval_seconds: float = 15.0,
        service_name: str = "infographix-inference",
        service_version: str = "1.0.0",
        environment: str = "production",
        timeout_seconds: float = 8.0,
    ) -> None:
        base = otlp_endpoint.rstrip("/")
        self.url = base if base.endswith("/v1/metrics") else f"{base}/v1/metrics"
        self.auth = (str(instance_id).strip(), str(api_key).strip())
        self.metrics = metrics
        self.interval_seconds = max(5.0, float(interval_seconds))
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.timeout_seconds = float(timeout_seconds)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.push_count: int = 0
        self.last_push_ok: bool | None = None
        self.last_error: str | None = None

    def push_once(self) -> bool:
        """Push one metrics snapshot to Grafana Cloud. Never raises."""
        try:
            payload = build_otlp_metrics_payload(
                metrics=self.metrics,
                service_name=self.service_name,
                service_version=self.service_version,
                environment=self.environment,
            )
            resp = requests.post(
                self.url,
                json=payload,
                auth=self.auth,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_seconds,
            )
            if 200 <= resp.status_code < 300:
                self.push_count += 1
                self.last_push_ok = True
                self.last_error = None
                return True
            self.last_push_ok = False
            self.last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            logger.warning("Grafana Cloud metrics push failed (%s)", self.last_error)
            return False
        except Exception as exc:
            self.last_push_ok = False
            self.last_error = str(exc)
            logger.warning("Grafana Cloud metrics push error: %s", exc)
            return False

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.push_once()
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="grafana-cloud-metrics-pusher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "GrafanaCloudPusher started (url=%s, interval=%.0fs)",
            self.url,
            self.interval_seconds,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
