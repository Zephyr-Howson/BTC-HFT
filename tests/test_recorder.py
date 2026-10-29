"""Offline tests: parsing, sequence-gap detection and Parquet writing."""

import asyncio
import json

import pyarrow.parquet as pq

from recorder.binance import BinanceRecorder
from recorder.schemas import DEPTH_UPDATES, parse_depth, parse_snapshot, parse_trade
from recorder.writer import BufferedParquetWriter

TRADE = {"e": "trade", "E": 1761700000123, "s": "BTCUSDT", "t": 42, "p": "71234.50",
         "q": "0.015", "T": 1761700000120, "m": True}
DEPTH = {"e": "depthUpdate", "E": 1761700000200, "s": "BTCUSDT", "U": 100, "u": 103,
         "b": [["71234.40", "1.2"], ["71234.30", "0"]], "a": [["71234.60", "0.5"]]}


def test_parse_trade():
    row = parse_trade(TRADE, 1)
    assert row["price"] == 71234.5 and row["qty"] == 0.015 and row["buyer_is_maker"] is True


def test_parse_depth_flattens_levels():
    rows = parse_depth(DEPTH, 1, session=7)
    assert len(rows) == 3
    assert {r["side"] for r in rows} == {"b", "a"}
    assert all(r["first_update_id"] == 100 and r["final_update_id"] == 103 for r in rows)


def test_parse_snapshot():
    rows = parse_snapshot({"lastUpdateId": 99, "bids": [["1", "2"]], "asks": [["3", "4"]]}, 5, 7, "connect")
    assert [r["side"] for r in rows] == ["b", "a"] and rows[0]["last_update_id"] == 99


def test_gap_detection(tmp_path):
    async def go():
        rec = BinanceRecorder(out_dir=tmp_path)
        reasons = []
        rec._request_snapshot = reasons.append  # don't hit the network
        rec._check_sequence(100, 103)  # first update after connect
        rec._check_sequence(104, 110)  # contiguous
        rec._check_sequence(115, 120)  # gap
        return rec, reasons
    rec, reasons = asyncio.run(go())
    assert reasons == ["connect", "gap"] and rec.stats["gaps"] == 1


def test_handle_routes_messages(tmp_path):
    async def go():
        rec = BinanceRecorder(out_dir=tmp_path)
        rec._request_snapshot = lambda reason: None
        rec._handle(json.dumps({"stream": "btcusdt@trade", "data": TRADE}), 1)
        rec._handle(json.dumps({"stream": "btcusdt@depth@100ms", "data": DEPTH}), 2)
        return rec
    rec = asyncio.run(go())
    assert rec.writers["trades"]._n == 1 and rec.writers["depth_updates"]._n == 3


def test_writer_roundtrip(tmp_path):
    async def go():
        w = BufferedParquetWriter(tmp_path, "depth_updates", DEPTH_UPDATES)
        w.extend(parse_depth(DEPTH, 1761700000000000000, session=7))
        await w.flush()
        return w
    w = asyncio.run(go())
    files = list(tmp_path.rglob("*.parquet"))
    assert len(files) == 1 and "date=2025-10-29" in str(files[0])
    assert pq.read_table(files[0]).num_rows == 3 == w.rows_written
