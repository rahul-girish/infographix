"""Mock/interface adapter for Rahul's Data Drift & Monitoring module (out of scope)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol


class DriftIntegrationProtocol(Protocol):
    """Contract expected when integrating with the external Data Drift module."""

    def build_sample_drift_payload(
        self,
        *,
        model_id: str,
        monitoring_run_id: str,
        drift_status: str = "warning",
    ) -> dict[str, Any]: ...


class MockDriftAdapter:
    """Provides clearly labeled mock drift payloads for offline integration tests."""

    @staticmethod
    def build_sample_drift_payload(
        *,
        model_id: str,
        monitoring_run_id: str,
        drift_status: str = "warning",
    ) -> dict[str, Any]:
        return {
            "model_id": model_id,
            "monitoring_run_id": monitoring_run_id,
            "monitoring_window": "mock-window-2026-11-28T00:00Z/24h",
            "drift_status": drift_status,
            "drift_metrics": {
                "psi_element_count": 0.28 if drift_status == "critical" else 0.14,
                "ks_aspect_ratio_pvalue": 0.004 if drift_status == "critical" else 0.09,
                "mean_layout_overlap_delta": 0.06 if drift_status == "critical" else 0.01,
            },
            "threshold_config": {
                "psi_warning": 0.10,
                "psi_critical": 0.25,
                "ks_pvalue_critical": 0.01,
            },
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "is_mock": True,
        }
