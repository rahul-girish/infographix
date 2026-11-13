from __future__ import annotations

import ctypes
import gc
import logging
import os
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)

_PROCESS = psutil.Process(os.getpid())
_LIBC: Any = None
try:
    _LIBC = ctypes.CDLL("libc.so.6")
except Exception:
    _LIBC = None


def get_rss_mib() -> float:
    """Return current resident set size (RSS) of the process in MiB."""
    return round(float(_PROCESS.memory_info().rss) / (1024.0 * 1024.0), 2)


def trim_process_memory() -> None:
    """Trigger Python GC, PyArrow pool release (if loaded), and glibc malloc_trim."""
    gc.collect()
    try:
        import sys

        if "pyarrow" in sys.modules:
            sys.modules["pyarrow"].default_memory_pool().release_unused()
    except Exception:
        pass
    if _LIBC is not None and hasattr(_LIBC, "malloc_trim"):
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


@dataclass
class MemoryStageSnapshot:
    stage: str
    rss_mib: float
    delta_mib: float
    peak_python_alloc_mib: Optional[float] = None
    artifact_size_mib: Optional[float] = None
    dataset_mem_mib: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class MemoryTracker:
    """Records RSS and temporary peak Python allocations across lifecycle stages."""

    def __init__(self) -> None:
        self.snapshots: list[MemoryStageSnapshot] = []
        self._last_rss: float = get_rss_mib()

    def record(
        self,
        stage: str,
        *,
        peak_python_alloc_mib: Optional[float] = None,
        artifact_size_mib: Optional[float] = None,
        dataset_mem_mib: Optional[float] = None,
    ) -> MemoryStageSnapshot:
        current = get_rss_mib()
        delta = round(current - self._last_rss, 2)
        self._last_rss = current
        snap = MemoryStageSnapshot(
            stage=stage,
            rss_mib=current,
            delta_mib=delta,
            peak_python_alloc_mib=peak_python_alloc_mib,
            artifact_size_mib=artifact_size_mib,
            dataset_mem_mib=dataset_mem_mib,
        )
        self.snapshots.append(snap)
        logger.info(
            "Memory stage [%s]: rss=%.2f MiB (delta=%+.2f MiB)%s%s%s",
            stage,
            snap.rss_mib,
            snap.delta_mib,
            f", peak_py_alloc={peak_python_alloc_mib:.2f} MiB"
            if peak_python_alloc_mib is not None
            else "",
            f", artifact={artifact_size_mib:.2f} MiB" if artifact_size_mib is not None else "",
            f", dataset_df={dataset_mem_mib:.2f} MiB" if dataset_mem_mib is not None else "",
        )
        return snap

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.snapshots]


def measure_peak_python_alloc(fn, *args, **kwargs) -> tuple[Any, float]:
    """Run callable while tracking peak Python traced memory delta in MiB."""
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    try:
        result = fn(*args, **kwargs)
        _, peak_bytes = tracemalloc.get_traced_memory()
        return result, round(peak_bytes / (1024.0 * 1024.0), 2)
    finally:
        if not was_tracing:
            tracemalloc.stop()
