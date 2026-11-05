"""Shared recorder lifecycle: one combined websocket with reconnect and backoff,
buffered Parquet writers, and periodic flush and stats loops.

Subclasses set the streams and schemas, parse messages in _handle(), and can
add background tasks (snapshot timers, REST pollers).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Coroutine

import aiohttp
import pyarrow as pa
from websockets.asyncio.client import connect

from .writer import BufferedParquetWriter

log = logging.getLogger(__name__)


class StreamRecorder:
    RATE_KEYS: tuple[str, ...] = ("msgs",)          # per-minute counters, reset after each stats log
    TOTAL_KEYS: tuple[str, ...] = ("reconnects",)   # running totals

    def __init__(self, ws_url: str, root: Path, schemas: dict[str, pa.Schema], flush_seconds: float) -> None:
        self.ws_url = ws_url
        self.root = root
        self.label = f"{root.parent.name}/{root.name}"  # e.g. binance_futures/btcusdt
        self.writers = {
            name: BufferedParquetWriter(root, name, schema, max_seconds=flush_seconds)
            for name, schema in schemas.items()
        }
        self.stats = {k: 0 for k in (*self.RATE_KEYS, *self.TOTAL_KEYS)}
        self._http: aiohttp.ClientSession | None = None

    # ---- hooks for subclasses ----------------------------------------------

    def _handle(self, raw: str | bytes, recv_ns: int) -> None:
        raise NotImplementedError

    def on_connect(self) -> None:
        """Called after each (re)connect, before the first message."""

    def background(self) -> list[Coroutine]:
        """Extra coroutines to run alongside the websocket."""
        return []

    def pending(self) -> list[asyncio.Task]:
        """In-flight tasks to cancel on shutdown."""
        return []

    # ---- lifecycle ---------------------------------------------------------

    async def run(self) -> None:
        log.info("recording %s -> %s", self.ws_url, self.root)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as http:
            self._http = http
            background = [
                asyncio.create_task(self._flush_loop()),
                asyncio.create_task(self._stats_loop()),
                *(asyncio.create_task(c) for c in self.background()),
            ]
            try:
                await self._ws_loop()
            finally:
                background += self.pending()
                for t in background:
                    t.cancel()
                await asyncio.gather(*background, return_exceptions=True)
                await self.flush_all()
                log.info("%s flushed buffers, shutting down", self.label)

    async def flush_all(self) -> None:
        for w in self.writers.values():
            await w.flush()

    async def _ws_loop(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with connect(self.ws_url, max_size=2**23, open_timeout=10) as ws:
                    self.on_connect()
                    log.info("%s connected", self.label)
                    backoff = 1.0
                    async for raw in ws:
                        self._handle(raw, time.time_ns())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network errors, server closes, etc.
                log.warning("%s websocket error: %r", self.label, exc)
            self.stats["reconnects"] += 1
            log.info("%s reconnecting in %.0fs", self.label, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    # ---- housekeeping ------------------------------------------------------

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            for w in self.writers.values():
                await w.maybe_flush()

    async def _stats_loop(self) -> None:
        while True:
            await asyncio.sleep(60.0)
            s = self.stats
            log.info(
                "%s last 60s: %s | total %s",
                self.label,
                " ".join(f"{k}={s[k]}" for k in self.RATE_KEYS),
                " ".join(f"{k}={s[k]}" for k in self.TOTAL_KEYS),
            )
            for k in self.RATE_KEYS:
                s[k] = 0
