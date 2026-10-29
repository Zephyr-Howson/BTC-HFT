"""Records Binance spot trades, best bid/ask, and order-book depth to Parquet.

Design notes
- One combined websocket for trade, bookTicker and depth@100ms streams.
- Depth updates are checked for sequence gaps (each update's first id must be
  the previous update's final id + 1). Any gap, reconnect, or periodic timer
  triggers a REST snapshot so the book can always be rebuilt offline.
- Binance drops every connection after 24h; we reconnect with backoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import aiohttp
from websockets.asyncio.client import connect

from .schemas import (
    SCHEMAS,
    parse_book_ticker,
    parse_depth,
    parse_snapshot,
    parse_trade,
)
from .writer import BufferedParquetWriter

log = logging.getLogger(__name__)

ENDPOINTS = {
    # Standard endpoints.
    "default": ("wss://stream.binance.com:9443/stream", "https://api.binance.com/api/v3/depth"),
    # Market-data-only mirrors; try these if the default ones are blocked for you.
    "vision": ("wss://data-stream.binance.vision/stream", "https://data-api.binance.vision/api/v3/depth"),
}


class BinanceRecorder:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        out_dir: str | Path = "data",
        endpoint: str = "default",
        snapshot_interval: float = 900.0,
        depth_limit: int = 1000,
        flush_seconds: float = 60.0,
    ) -> None:
        self.symbol = symbol.lower()
        ws_base, self.rest_depth = ENDPOINTS[endpoint]
        streams = [f"{self.symbol}@trade", f"{self.symbol}@bookTicker", f"{self.symbol}@depth@100ms"]
        self.ws_url = f"{ws_base}?streams={'/'.join(streams)}"
        self.snapshot_interval = snapshot_interval
        self.depth_limit = depth_limit

        root = Path(out_dir) / "binance" / self.symbol
        self.writers = {
            name: BufferedParquetWriter(root, name, schema, max_seconds=flush_seconds)
            for name, schema in SCHEMAS.items()
        }

        self.session = 0
        self.prev_final_id: int | None = None
        self._snapshot_task: asyncio.Task | None = None
        self._http: aiohttp.ClientSession | None = None
        self.stats = {"msgs": 0, "trades": 0, "depth": 0, "ticker": 0, "gaps": 0, "reconnects": 0, "snapshots": 0}

    # ---- lifecycle ---------------------------------------------------------

    async def run(self) -> None:
        log.info("recording %s -> %s", self.ws_url, self.writers["trades"].dir.parent)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as http:
            self._http = http
            background = [
                asyncio.create_task(self._flush_loop()),
                asyncio.create_task(self._stats_loop()),
                asyncio.create_task(self._snapshot_timer()),
            ]
            try:
                await self._ws_loop()
            finally:
                if self._snapshot_task:
                    background.append(self._snapshot_task)
                for t in background:
                    t.cancel()
                await asyncio.gather(*background, return_exceptions=True)
                await self.flush_all()
                log.info("flushed buffers, shutting down")

    async def flush_all(self) -> None:
        for w in self.writers.values():
            await w.flush()

    # ---- websocket ---------------------------------------------------------

    async def _ws_loop(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with connect(self.ws_url, max_size=2**23, open_timeout=10) as ws:
                    self.session = time.time_ns()
                    self.prev_final_id = None  # forces a fresh snapshot
                    log.info("connected (session %d)", self.session)
                    backoff = 1.0
                    async for raw in ws:
                        self._handle(raw, time.time_ns())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network errors, server closes, etc.
                log.warning("websocket error: %r", exc)
            self.stats["reconnects"] += 1
            log.info("reconnecting in %.0fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    def _handle(self, raw: str | bytes, recv_ns: int) -> None:
        msg = json.loads(raw)
        stream, d = msg.get("stream", ""), msg.get("data")
        if d is None:
            return
        self.stats["msgs"] += 1

        if stream.endswith("@trade"):
            self.writers["trades"].append(parse_trade(d, recv_ns))
            self.stats["trades"] += 1
        elif stream.endswith("@bookTicker"):
            self.writers["book_ticker"].append(parse_book_ticker(d, recv_ns))
            self.stats["ticker"] += 1
        elif "@depth" in stream:
            self._check_sequence(d["U"], d["u"])
            self.writers["depth_updates"].extend(parse_depth(d, recv_ns, self.session))
            self.stats["depth"] += 1

    def _check_sequence(self, first_id: int, final_id: int) -> None:
        if self.prev_final_id is None:
            self._request_snapshot("connect")
        elif first_id != self.prev_final_id + 1:
            self.stats["gaps"] += 1
            log.warning("depth gap: expected %d, got %d", self.prev_final_id + 1, first_id)
            self._request_snapshot("gap")
        self.prev_final_id = final_id

    # ---- snapshots ---------------------------------------------------------

    def _request_snapshot(self, reason: str) -> None:
        if self._snapshot_task and not self._snapshot_task.done():
            return
        self._snapshot_task = asyncio.create_task(self._snapshot(reason))

    async def _snapshot(self, reason: str, attempts: int = 3) -> None:
        params = {"symbol": self.symbol.upper(), "limit": str(self.depth_limit)}
        for attempt in range(1, attempts + 1):
            try:
                async with self._http.get(self.rest_depth, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                recv_ns = time.time_ns()
                rows = parse_snapshot(data, recv_ns, self.session, reason)
                self.writers["depth_snapshots"].extend(rows)
                self.stats["snapshots"] += 1
                log.info("snapshot (%s) lastUpdateId=%d levels=%d", reason, data["lastUpdateId"], len(rows))
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("snapshot attempt %d/%d failed: %r", attempt, attempts, exc)
                await asyncio.sleep(5 * attempt)
        log.error("snapshot failed; will retry on next connect/gap/periodic trigger")

    async def _snapshot_timer(self) -> None:
        while True:
            await asyncio.sleep(self.snapshot_interval)
            self._request_snapshot("periodic")

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
                "last 60s: msgs=%d trades=%d depth=%d ticker=%d | total gaps=%d reconnects=%d snapshots=%d",
                s["msgs"], s["trades"], s["depth"], s["ticker"], s["gaps"], s["reconnects"], s["snapshots"],
            )
            for k in ("msgs", "trades", "depth", "ticker"):
                s[k] = 0
