from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Optional

import requests
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class PeerKeepAlive:
    """Periodic background HTTP pinger to keep a peer container warm."""

    def __init__(
        self,
        peer_url: str,
        ping_path: str = "/health/live",
        interval_seconds: float = 600.0,
        timeout_seconds: float = 10.0,
        service_name: str = "infographix-backend",
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


class DatabaseKeepAlive:
    """Periodic lightweight DB query to prevent Supabase free-tier 7-day auto-pause."""

    def __init__(
        self,
        engine: Engine,
        interval_seconds: float = 86400.0,
        query: str = "SELECT id FROM generation_runs LIMIT 1",
        service_name: str = "infographix-backend-db",
    ) -> None:
        self.engine = engine
        self.interval_seconds = float(interval_seconds)
        self.query = query
        self.service_name = service_name
        self._task: Optional[asyncio.Task[None]] = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def ping_once(self) -> bool:
        """Execute a lightweight query against the configured database."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(self.query)).fetchone()
            logger.info("Database keep-alive ping succeeded.")
            return True
        except Exception as exc:
            logger.warning("Database keep-alive ping failed: %s", exc)
            return False

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_seconds)
            await asyncio.to_thread(self.ping_once)

    def start(self) -> None:
        if self.interval_seconds <= 0:
            logger.info("Database keep-alive disabled (interval <= 0).")
            return
        if self.is_running:
            return
        self._task = asyncio.create_task(self._loop(), name=f"{self.service_name}-keepalive")
        logger.info(
            "Database keep-alive started: querying every %.1fs",
            self.interval_seconds,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        logger.info("Database keep-alive stopped.")
