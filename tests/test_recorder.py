"""Offline tests: parsing, sequence-gap detection, message routing and Parquet writing."""

import asyncio
import json

import pyarrow.parquet as pq

from recorder.binance import BinanceRecorder
from recorder.futures import FuturesRecorder
from recorder.schemas import (DEPTH_UPDATES, parse_depth, parse_liquidation, parse_mark_price,
                              parse_open_interest, parse_snapshot, parse_trade)
from recorder.writer import BufferedParquetWriter

TRADE = {"e": "trade", "E": 1761700000123, "s": "BTCUSDT", "t": 42, "p": "71234.50",
         "q": "0.015", "T": 1761700000120, "m": True}
DEPTH = {"e": "depthUpdate", "E": 1761700000200, "s": "BTCUSDT", "U": 100, "u": 103,
         "b": [["71234.40", "1.2"], ["71234.30", "0"]], "a": [["71234.60", "0.5"]]}

# Futures messages, as captured from wss://fstream.binance.com/market/stream.
F_AGG = {"e": "aggTrade", "E": 1793837791321, "a": 3508086748, "s": "BTCUSDT", "p": "80675.70", "q": "0.323",
         "nq": "0.323", "f": 8235677490, "l": 8235677490, "T": 1793837791170, "m": True, "st": 1}
F_MARK = {"e": "markPriceUpdate", "E": 1793837792000, "s": "BTCUSDT", "p": "80675.70000000",
          "ap": "80675.70000000", "P": "80595.98893924", "i": "80709.47652174", "r": "0.00002675",
          "T": 1793865600000, "st": 1}
F_LIQ = {"e": "forceOrder", "E": 1793837796161, "o": {"s": "FLUIDUSDT", "S": "BUY", "o": "LIMIT", "f": "IOC",
         "q": "30.5", "p": "2.563000", "ap": "2.548000", "X": "FILLED", "l": "30.5", "z": "30.5",
         "T": 1793837795158, "ps": "FLUIDUSDT", "st": 1}}


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


def test_parse_futures_messages():
    m = parse_mark_price(F_MARK, 1)
    assert m["mark_px"] == 80675.7 and m["funding_rate"] == 0.00002675 and m["next_funding_ms"] == 1793865600000
    liq = parse_liquidation(F_LIQ, 1)
    assert liq["symbol"] == "FLUIDUSDT" and liq["side"] == "BUY" and liq["avg_price"] == 2.548
    oi = parse_open_interest({"symbol": "BTCUSDT", "openInterest": "96176.640", "time": 1793837782854}, 1)
    assert oi == {"recv_ns": 1, "time_ms": 1793837782854, "open_interest": 96176.64}


def test_futures_handle_routes_messages(tmp_path):
    rec = FuturesRecorder(out_dir=tmp_path)
    no_nq = {k: v for k, v in F_AGG.items() if k != "nq"}
    for stream, data in [("btcusdt@aggTrade", F_AGG), ("btcusdt@aggTrade", no_nq),
                         ("btcusdt@markPrice@1s", F_MARK), ("!forceOrder@arr", F_LIQ),
                         ("!forceOrder@arr", [F_LIQ, F_LIQ]), ("btcusdt@unknown", F_AGG)]:
        rec._handle(json.dumps({"stream": stream, "data": data}), 1)
    w = rec.writers
    assert (w["agg_trades"]._n, w["mark_price"]._n, w["liquidations"]._n) == (2, 1, 3)
    assert w["agg_trades"]._cols["normal_qty"] == [0.323, None]
    assert "binance_futures" in str(w["agg_trades"].dir)


def test_futures_writer_roundtrip(tmp_path):
    async def go():
        rec = FuturesRecorder(out_dir=tmp_path)
        rec._handle(json.dumps({"stream": "!forceOrder@arr", "data": F_LIQ}), 1793837796200000000)
        await rec.flush_all()
    asyncio.run(go())
    files = list(tmp_path.rglob("liquidations/**/*.parquet"))
    assert len(files) == 1 and pq.read_table(files[0]).column("symbol").to_pylist() == ["FLUIDUSDT"]
