"""Records Binance USDⓈ-M futures data used as signals for spot trading:
perp trades, mark/index price and funding, open interest, and liquidations.

Design notes
- Binance now serves futures streams from separate websocket paths: /market
  carries aggTrade, markPrice and forceOrder; /public carries bookTicker and
  depth. The legacy /stream path silently delivers only bookTicker, so we use
  /market. (Perp best bid/ask is ~400 msgs/s and isn't needed for signals.)
- Liquidations come from the all-market stream, with a symbol column: cascades
  across the whole market move BTC too. Binance pushes at most one liquidation
  per symbol per second, so this is a sample, not a complete record.
- Open interest has no stream, so it is polled over REST.
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
    FUTURES_SCHEMAS,
    parse_futures_agg_trade,
    parse_liquidation,
    parse_mark_price,
    parse_open_interest,
)

log = logging.getLogger(__name__)

WS_MARKET = "wss://fstream.binance.com/market/stream"
REST_OPEN_INTEREST = "https://fapi.binance.com/fapi/v1/openInterest"


class FuturesRecorder(StreamRecorder):
    RATE_KEYS = ("msgs", "trades", "mark", "liquidations")
    TOTAL_KEYS = ("reconnects", "oi_polls", "oi_errors")

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        out_dir: str | Path = "data",
        oi_interval: float = 10.0,
        flush_seconds: float = 60.0,
    ) -> None:
        self.symbol = symbol.lower()
        streams = [f"{self.symbol}@aggTrade", f"{self.symbol}@markPrice@1s", "!forceOrder@arr"]
        super().__init__(f"{WS_MARKET}?streams={'/'.join(streams)}", Path(out_dir) / "binance_futures" / self.symbol,
                         FUTURES_SCHEMAS, flush_seconds)
        self.oi_interval = oi_interval

    def background(self) -> list[Coroutine]:
        return [self._open_interest_loop()]

    def _handle(self, raw: str | bytes, recv_ns: int) -> None:
        msg = json.loads(raw)
        stream, d = msg.get("stream", ""), msg.get("data")
        if d is None:
            return
        self.stats["msgs"] += 1

        if stream.endswith("@aggTrade"):
            self.writers["agg_trades"].append(parse_futures_agg_trade(d, recv_ns))
            self.stats["trades"] += 1
        elif "@markPrice" in stream:
            self.writers["mark_price"].append(parse_mark_price(d, recv_ns))
            self.stats["mark"] += 1
        elif stream == "!forceOrder@arr":
            for event in d if isinstance(d, list) else [d]:
                self.writers["liquidations"].append(parse_liquidation(event, recv_ns))
                self.stats["liquidations"] += 1

    async def _open_interest_loop(self) -> None:
        params = {"symbol": self.symbol.upper()}
        while True:
            try:
                async with self._http.get(REST_OPEN_INTEREST, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                self.writers["open_interest"].append(parse_open_interest(data, time.time_ns()))
                self.stats["oi_polls"] += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.stats["oi_errors"] += 1
                log.warning("open interest poll failed: %r", exc)
            await asyncio.sleep(self.oi_interval)
