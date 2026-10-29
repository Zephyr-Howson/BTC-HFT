"""Parquet schemas and parsers for Binance spot market-data messages.

Every row carries recv_ns: local wall-clock receive time in nanoseconds.
Comparing it with the exchange's event time lets you measure your real
feed latency later, which the backtester will need.
"""

from __future__ import annotations

import pyarrow as pa

TRADES = pa.schema([
    ("recv_ns", pa.int64()),
    ("event_ms", pa.int64()),
    ("trade_ms", pa.int64()),
    ("trade_id", pa.int64()),
    ("price", pa.float64()),
    ("qty", pa.float64()),
    ("buyer_is_maker", pa.bool_()),  # True => aggressor was the seller
])

BOOK_TICKER = pa.schema([
    ("recv_ns", pa.int64()),
    ("update_id", pa.int64()),
    ("bid_px", pa.float64()),
    ("bid_qty", pa.float64()),
    ("ask_px", pa.float64()),
    ("ask_qty", pa.float64()),
])

# One row per changed price level. qty == 0 means the level was removed.
DEPTH_UPDATES = pa.schema([
    ("recv_ns", pa.int64()),
    ("session", pa.int64()),
    ("event_ms", pa.int64()),
    ("first_update_id", pa.int64()),
    ("final_update_id", pa.int64()),
    ("side", pa.string()),  # "b" bid / "a" ask
    ("price", pa.float64()),
    ("qty", pa.float64()),
])

# Full order-book snapshots used as starting points for book reconstruction.
DEPTH_SNAPSHOTS = pa.schema([
    ("recv_ns", pa.int64()),
    ("session", pa.int64()),
    ("snapshot_id", pa.int64()),
    ("last_update_id", pa.int64()),
    ("reason", pa.string()),  # connect / gap / periodic
    ("side", pa.string()),
    ("price", pa.float64()),
    ("qty", pa.float64()),
])

SCHEMAS = {
    "trades": TRADES,
    "book_ticker": BOOK_TICKER,
    "depth_updates": DEPTH_UPDATES,
    "depth_snapshots": DEPTH_SNAPSHOTS,
}


def parse_trade(d: dict, recv_ns: int) -> dict:
    return {
        "recv_ns": recv_ns,
        "event_ms": d["E"],
        "trade_ms": d["T"],
        "trade_id": d["t"],
        "price": float(d["p"]),
        "qty": float(d["q"]),
        "buyer_is_maker": d["m"],
    }


def parse_book_ticker(d: dict, recv_ns: int) -> dict:
    return {
        "recv_ns": recv_ns,
        "update_id": d["u"],
        "bid_px": float(d["b"]),
        "bid_qty": float(d["B"]),
        "ask_px": float(d["a"]),
        "ask_qty": float(d["A"]),
    }


def parse_depth(d: dict, recv_ns: int, session: int) -> list[dict]:
    base = {
        "recv_ns": recv_ns,
        "session": session,
        "event_ms": d["E"],
        "first_update_id": d["U"],
        "final_update_id": d["u"],
    }
    rows = []
    for side in ("b", "a"):
        for px, qty in d[side]:
            rows.append({**base, "side": side, "price": float(px), "qty": float(qty)})
    return rows


def parse_snapshot(d: dict, recv_ns: int, session: int, reason: str) -> list[dict]:
    base = {
        "recv_ns": recv_ns,
        "session": session,
        "snapshot_id": recv_ns,
        "last_update_id": d["lastUpdateId"],
        "reason": reason,
    }
    rows = []
    for key, side in (("bids", "b"), ("asks", "a")):
        for px, qty in d[key]:
            rows.append({**base, "side": side, "price": float(px), "qty": float(qty)})
    return rows
