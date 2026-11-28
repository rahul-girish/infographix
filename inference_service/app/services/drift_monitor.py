"""Online Input & Output Data Drift Monitor for Infographix Inference Service.

Maintains a lightweight, thread-safe rolling window of incoming production poster
requests and compares their feature distributions against the SciPostLayout v2
training-split reference baseline (`app/data/drift_reference.json`) using:
  - Population Stability Index (PSI): < 0.10 stable, 0.10-0.20 moderate, >= 0.20 significant drift
  - Two-sample Kolmogorov-Smirnov (KS) statistic
  - Orientation distribution shift (landscape rate delta)
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parent.parent / "data" / "drift_reference.json"


@dataclass(frozen=True)
class DriftObservation:
    total_element_count: float
    total_word_count: float
    figure_count: float
    mean_figure_aspect_ratio: float | None
    is_landscape: float
    top1_relevance: float | None = None
    overlap_area: float | None = None


def compute_psi(
    values: np.ndarray,
    inner_bin_edges: list[float],
    ref_bin_proportions: list[float],
    eps: float = 1e-4,
) -> float:
    """Compute Population Stability Index (PSI) against fixed reference bin edges."""
    if len(values) == 0:
        return 0.0
    full_edges = [-np.inf] + list(inner_bin_edges) + [np.inf]
    counts, _ = np.histogram(values, bins=full_edges)
    actual_props = counts.astype(np.float64) / max(1, len(values))
    ref_props = np.asarray(ref_bin_proportions, dtype=np.float64)

    # Align lengths defensively
    n = min(len(actual_props), len(ref_props))
    a = np.clip(actual_props[:n], eps, 1.0)
    r = np.clip(ref_props[:n], eps, 1.0)
    psi = float(np.sum((a - r) * np.log(a / r)))
    return max(0.0, round(psi, 4))


class OnlineDriftMonitor:
    """Thread-safe rolling-window drift detector backed by train-split reference stats."""

    def __init__(
        self,
        reference_path: Path | None = None,
        window_size: int = 100,
        min_samples: int = 10,
        psi_threshold: float = 0.20,
    ) -> None:
        self.reference_path = Path(reference_path) if reference_path else DEFAULT_REFERENCE_PATH
        self.window_size = max(5, int(window_size))
        self.min_samples = max(2, min(int(min_samples), self.window_size))
        self.psi_threshold = float(psi_threshold)

        self._lock = threading.Lock()
        self._window: deque[DriftObservation] = deque(maxlen=self.window_size)
        self._reference: dict[str, Any] = self._load_reference(self.reference_path)
        self._last_snapshot: dict[str, Any] = self._build_snapshot_locked()

    @staticmethod
    def _load_reference(path: Path) -> dict[str, Any]:
        if not path.exists():
            logger.warning("Drift reference file not found at %s; drift detection inactive.", path)
            return {"dataset": "unavailable", "landscape_rate": 0.31, "features": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def record(self, obs: DriftObservation) -> dict[str, Any]:
        """Record one production inference observation and return updated drift snapshot."""
        with self._lock:
            self._window.append(obs)
            self._last_snapshot = self._build_snapshot_locked()
            return self._last_snapshot

    def reset(self) -> dict[str, Any]:
        """Clear the rolling window (useful for fresh monitoring demos)."""
        with self._lock:
            self._window.clear()
            self._last_snapshot = self._build_snapshot_locked()
            return self._last_snapshot

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_snapshot)

    def _build_snapshot_locked(self) -> dict[str, Any]:
        n_samples = len(self._window)
        ref_features = self._reference.get("features", {})
        ref_landscape = float(self._reference.get("landscape_rate", 0.31))

        feature_reports: dict[str, dict[str, Any]] = {}
        psi_values: list[float] = []

        if n_samples == 0:
            for fname, fmeta in ref_features.items():
                feature_reports[fname] = {
                    "psi": 0.0,
                    "ks_stat": 0.0,
                    "current_mean": 0.0,
                    "reference_mean": float(fmeta.get("mean", 0.0)),
                    "samples": 0,
                    "status": "insufficient_data",
                }
            return {
                "reference_dataset": self._reference.get("dataset", "unknown"),
                "window_samples": 0,
                "window_size": self.window_size,
                "min_samples": self.min_samples,
                "psi_threshold": self.psi_threshold,
                "ready": False,
                "overall_psi": 0.0,
                "orientation_shift": 0.0,
                "current_landscape_rate": 0.0,
                "reference_landscape_rate": ref_landscape,
                "alert_active": False,
                "features": feature_reports,
            }

        extractors = {
            "total_element_count": [o.total_element_count for o in self._window],
            "total_word_count": [o.total_word_count for o in self._window],
            "figure_count": [o.figure_count for o in self._window],
            "mean_figure_aspect_ratio": [
                o.mean_figure_aspect_ratio
                for o in self._window
                if o.mean_figure_aspect_ratio is not None
            ],
        }

        ready = n_samples >= self.min_samples
        for fname, fmeta in ref_features.items():
            vals = np.asarray(extractors.get(fname, []), dtype=np.float64)
            ref_mean = float(fmeta.get("mean", 0.0))
            cur_mean = round(float(np.mean(vals)), 4) if len(vals) > 0 else 0.0

            if ready and len(vals) >= max(3, self.min_samples // 2):
                psi = compute_psi(
                    vals,
                    inner_bin_edges=fmeta["inner_bin_edges"],
                    ref_bin_proportions=fmeta["ref_bin_proportions"],
                )
                ks_ref = np.asarray(fmeta.get("ks_reference_sample", []), dtype=np.float64)
                ks_stat = round(float(ks_2samp(vals, ks_ref).statistic), 4) if len(ks_ref) > 0 else 0.0
                psi_values.append(psi)
                status = (
                    "drift_detected"
                    if psi >= self.psi_threshold
                    else ("moderate_shift" if psi >= 0.10 else "stable")
                )
            else:
                psi = 0.0
                ks_stat = 0.0
                status = "insufficient_data"

            feature_reports[fname] = {
                "psi": psi,
                "ks_stat": ks_stat,
                "current_mean": cur_mean,
                "reference_mean": ref_mean,
                "samples": int(len(vals)),
                "status": status,
            }

        cur_landscape = round(float(np.mean([o.is_landscape for o in self._window])), 4)
        orientation_shift = round(abs(cur_landscape - ref_landscape), 4) if ready else 0.0
        overall_psi = round(max(psi_values), 4) if psi_values else 0.0
        alert_active = bool(ready and overall_psi >= self.psi_threshold)

        return {
            "reference_dataset": self._reference.get("dataset", "unknown"),
            "window_samples": n_samples,
            "window_size": self.window_size,
            "min_samples": self.min_samples,
            "psi_threshold": self.psi_threshold,
            "ready": ready,
            "overall_psi": overall_psi,
            "orientation_shift": orientation_shift,
            "current_landscape_rate": cur_landscape,
            "reference_landscape_rate": ref_landscape,
            "alert_active": alert_active,
            "features": feature_reports,
        }
