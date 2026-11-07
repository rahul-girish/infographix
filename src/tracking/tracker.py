"""Tracking abstraction supporting Aim with fallback to JSON / local logging.

Ensures experiment tracking works across Linux/WSL2 (where Aim runs natively)
and Windows environments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentTracker:
    """Unified experiment tracking wrapper for Aim with JSON fallback."""

    def __init__(
        self,
        experiment_name: str = "infographix-clustering",
        run_name: str | None = None,
        use_aim: bool = True,
        output_dir: str | Path | None = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name or "run"
        self.output_dir = Path(output_dir or "reports/comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.aim_run = None
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.tags: list[str] = []

        if use_aim:
            try:
                import os
                # Compatibility setting for protobuf
                os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
                from aim import Run
                self.aim_run = Run(experiment=experiment_name)
                if run_name:
                    self.aim_run.name = run_name
            except Exception as e:
                # Fallback to local JSON logging if Aim is not initialized or incompatible
                self.aim_run = None

    def set_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)
        if self.aim_run is not None:
            try:
                self.aim_run["hparams"] = params
            except Exception:
                pass

    def add_tag(self, tag: str) -> None:
        self.tags.append(tag)
        if self.aim_run is not None:
            try:
                self.aim_run.add_tag(tag)
            except Exception:
                pass

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "", step: int = 0) -> None:
        for k, v in metrics.items():
            key = f"{prefix}{k}"
            self.metrics[key] = v
            if self.aim_run is not None and isinstance(v, (int, float)):
                import numpy as np
                if np.isfinite(v):
                    try:
                        self.aim_run.track(float(v), name=key, step=step)
                    except Exception:
                        pass

    def close(self) -> None:
        if self.aim_run is not None:
            try:
                self.aim_run.close()
            except Exception:
                pass

        # Always persist a local JSON copy of the run for reproducibility
        run_file = self.output_dir / f"{self.run_name.replace(' ', '_')}_tracker_log.json"
        log_data = {
            "experiment": self.experiment_name,
            "run_name": self.run_name,
            "tags": self.tags,
            "params": self.params,
            "metrics": self.metrics,
        }
        run_file.write_text(json.dumps(log_data, indent=2, default=str), encoding="utf-8")
