from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class PeerKeepAlive:
    """Periodic background HTTP pinger to keep a peer container warm."""

    def __init__(
        self,
        peer_url: str,
        ping_path: str = "/",
        interval_seconds: float = 600.0,
        timeout_seconds: float = 10.0,
        service_name: str = "infographix-inference",
    ) -> None:
        base = peer_url.strip().rstrip("/")
        path = ping_path.strip()
        if not path.startswith("/"):
            path = f"/{path}"
        self.target_url = f"{base}{path}"
        self.interval_seconds = float(interval_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.service_name = service_name
        self._task: Optional[asyncio.Task[None]] = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def ping_once(self) -> bool:
        """Send a single lightweight GET request to the peer service."""
        try:
            resp = requests.get(
                self.target_url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": f"{self.service_name}-keepalive/1.0"},
            )
            if 200 <= resp.status_code < 400:
                logger.info(
                    "Keep-alive ping succeeded: %s -> %s",
                    self.target_url,
                    resp.status_code,
                )
                return True
            logger.warning(
                "Keep-alive ping returned HTTP %s from %s",
                resp.status_code,
                self.target_url,
            )
            return False
        except Exception as exc:
            logger.warning("Keep-alive ping failed for %s: %s", self.target_url, exc)
            return False

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_seconds)
            await asyncio.to_thread(self.ping_once)

    def start(self) -> None:
        if self.interval_seconds <= 0:
            logger.info("Keep-alive disabled (interval <= 0).")
            return
        if self.is_running:
            return
        self._task = asyncio.create_task(self._loop(), name=f"{self.service_name}-keepalive")
        logger.info(
            "Keep-alive started: pinging %s every %.1fs",
            self.target_url,
            self.interval_seconds,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        logger.info("Keep-alive stopped for %s", self.target_url)
