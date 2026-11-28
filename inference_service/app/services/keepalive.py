"""Lightweight background keepalive pinger for companion Render services."""

from __future__ import annotations

import logging
import threading

import requests

logger = logging.getLogger(__name__)


class KeepalivePinger:
    """Periodically sends HTTP GET pings to a target URL while the app is running."""

    def __init__(
        self,
        target_url: str,
        interval_seconds: float = 300.0,
        timeout_seconds: float = 5.0,
    ) -> None:
        base = target_url.rstrip("/")
        self.target_url = f"{base}/"
        self.interval_seconds = max(0.01, float(interval_seconds))
        self.timeout_seconds = max(0.5, float(timeout_seconds))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                requests.get(self.target_url, timeout=self.timeout_seconds)
            except Exception as exc:
                logger.debug("Keepalive ping to %s failed: %s", self.target_url, exc)
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="infographix-keepalive",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
