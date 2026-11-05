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
from typing import Coroutine

from .base import StreamRecorder
from .schemas import (
    SCHEMAS,
    parse_book_ticker,
    parse_depth,
    parse_snapshot,
    parse_trade,
)

log = logging.getLogger(__name__)

ENDPOINTS = {
    # Standard endpoints.
    "default": ("wss://stream.binance.com:9443/stream", "https://api.binance.com/api/v3/depth"),
    # Market-data-only mirrors; try these if the default ones are blocked for you.
    "vision": ("wss://data-stream.binance.vision/stream", "https://data-api.binance.vision/api/v3/depth"),
}


class BinanceRecorder(StreamRecorder):
    RATE_KEYS = ("msgs", "trades", "depth", "ticker")
    TOTAL_KEYS = ("gaps", "reconnects", "snapshots")

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
        super().__init__(f"{ws_base}?streams={'/'.join(streams)}", Path(out_dir) / "binance" / self.symbol,
                         SCHEMAS, flush_seconds)
        self.snapshot_interval = snapshot_interval
        self.depth_limit = depth_limit

        self.session = 0
        self.prev_final_id: int | None = None
        self._snapshot_task: asyncio.Task | None = None

    # ---- hooks -------------------------------------------------------------

    def on_connect(self) -> None:
        self.session = time.time_ns()
        self.prev_final_id = None  # forces a fresh snapshot
        log.info("session %d", self.session)

    def background(self) -> list[Coroutine]:
        return [self._snapshot_timer()]

    def pending(self) -> list[asyncio.Task]:
        return [self._snapshot_task] if self._snapshot_task else []

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
