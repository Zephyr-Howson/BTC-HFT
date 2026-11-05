"""Offline tests: date ranges, download planning, CSV conversion, fetch verification, bars and retention."""

import hashlib
import io
import threading
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from history import vision
from history.bars import build
from history.datasets import DATASETS, convert, in_range, load
from history.vision import Item, fetch, plan

SPOT_AGG = DATASETS["spot/aggTrades"]
FUT_AGG = DATASETS["futures/aggTrades"]
METRICS = DATASETS["futures/metrics"]


def make_zip(path, text, name="x.csv"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name, text)
    return path


def test_in_range():
    assert in_range("2026-10", "2026-10-15", "2026-11")        # month overlaps a mid-month start
    assert in_range("2026-11-03", "2026-11", "2026-11")        # day inside a month range
    assert not in_range("2026-11-03", "", "2026-11-02")
    assert not in_range("2025-12", "2026-01")
    assert in_range("2017-08")                                  # default bounds take everything


def test_plan_prefers_monthly_and_fills_gaps_with_daily():
    listing = {
        "monthly": [("data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-09.zip", 10),
                    ("data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-09.zip.CHECKSUM", 1),
                    ("data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-10.zip", 10)],
        "daily": [(f"data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{d}.zip", 1)
                  for d in ("2026-09-30", "2026-10-31", "2026-11-01", "2026-11-02")],
    }
    items = plan(SPOT_AGG, "BTCUSDT", start="2026-10", lister=lambda p: listing[p.split("/")[2]])
    assert [i.label for i in items] == ["2026-10", "2026-11-01", "2026-11-02"]


def test_convert_spot_aggtrades_normalises_ms_and_us(tmp_path):
    # No header; one millisecond-era row and one microsecond-era row (post 2025-01-01).
    csv = ("1,42000.5,0.1,1,1,1704067200123,True,True\n"
           "2,90000.0,0.2,2,3,1735689600000456,False,True\n")
    out = tmp_path / "2025-01.parquet"
    assert convert(make_zip(tmp_path / "a.zip", csv), out, SPOT_AGG) == 2
    t = pq.read_table(out).to_pydict()
    assert t["transact_time_us"] == [1704067200123000, 1735689600000456]
    assert t["is_buyer_maker"] == [True, False] and "is_best_match" not in t


def test_convert_detects_header(tmp_path):
    csv = ("agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
           "7,77522.6,0.016,9,9,1793491200016,true\n")
    out = tmp_path / "2026-11-01.parquet"
    assert convert(make_zip(tmp_path / "a.zip", csv), out, FUT_AGG) == 1
    assert pq.read_table(out).column("agg_trade_id").to_pylist() == [7]


def test_convert_sorts_and_dedupes(tmp_path):
    header = ",".join(METRICS.columns)
    row = "2026-11-01 {},BTCUSDT,1,2,3,4,5,6"
    csv = "\n".join([header, row.format("00:10:00"), row.format("00:05:00"), row.format("00:05:00")]) + "\n"
    out = tmp_path / "2026-11-01.parquet"
    assert convert(make_zip(tmp_path / "a.zip", csv), out, METRICS) == 2
    times = pq.read_table(out).column("create_time_us").to_pylist()
    assert times == [1793491500000000, 1793491800000000]  # 00:05 then 00:10 UTC, as epoch us


class FakeUrlopen:
    """Serves the zip and its .CHECKSUM without touching the network."""

    def __init__(self, payload: bytes, checksum: str):
        self.payload, self.checksum = payload, checksum

    def __call__(self, url, timeout=None):
        body = f"{self.checksum}  file.zip".encode() if url.endswith(".CHECKSUM") else self.payload
        return io.BytesIO(body)


def _spot_item(label):
    return Item(SPOT_AGG, f"data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{label}.zip", label, 1)


def serve_zip(tmp_path, monkeypatch, csv="1,42000.5,0.1,1,1,1704067200123,True,True\n"):
    payload = make_zip(tmp_path / "src.zip", csv).read_bytes()
    monkeypatch.setattr(vision.urllib.request, "urlopen", FakeUrlopen(payload, hashlib.sha256(payload).hexdigest()))


def test_fetch_verifies_builds_bars_and_replaces_daily_files(tmp_path, monkeypatch):
    serve_zip(tmp_path, monkeypatch)
    root = tmp_path / "h"
    item = _spot_item("2024-01")
    for d in (item.out_path(root, "BTCUSDT").parent, item.bars_path(root, "BTCUSDT").parent):
        d.mkdir(parents=True)
        (d / "2024-01-30.parquet").write_bytes(b"stale daily")

    assert fetch(item, root, "BTCUSDT", threading.Event()) == 1
    for d in (item.out_path(root, "BTCUSDT").parent, item.bars_path(root, "BTCUSDT").parent):
        assert sorted(f.name for f in d.iterdir()) == ["2024-01.parquet"]
    assert len(load("spot/aggTrades", root, start="2024-01", end="2024-01")) == 1
    assert load("spot/tradeBars_1m", root).volume.tolist() == [0.1]


def test_process_keeps_only_bars_for_old_months(tmp_path, monkeypatch):
    serve_zip(tmp_path, monkeypatch)
    item = Item(SPOT_AGG, _spot_item("2024-01").key, "2024-01", 1, keep_raw=False)
    assert item.needs_work(tmp_path, "BTCUSDT")
    vision.process(item, tmp_path, "BTCUSDT", threading.Event())
    assert not item.out_path(tmp_path, "BTCUSDT").exists() and item.bars_path(tmp_path, "BTCUSDT").exists()
    assert not item.needs_work(tmp_path, "BTCUSDT")  # re-runs don't download it again


def test_process_builds_bars_from_existing_raw_without_downloading(tmp_path, monkeypatch):
    serve_zip(tmp_path, monkeypatch)
    item = _spot_item("2024-01")
    fetch(item, tmp_path, "BTCUSDT", threading.Event())
    item.bars_path(tmp_path, "BTCUSDT").unlink()  # e.g. raw downloaded before bars existed
    monkeypatch.setattr(vision.urllib.request, "urlopen", None)  # any network use would fail
    assert vision.process(item, tmp_path, "BTCUSDT", threading.Event()) == "bars built from local raw file"
    assert item.bars_path(tmp_path, "BTCUSDT").exists()


def test_bars_group_taker_orders_and_merge_across_batches():
    m0, m1 = 1704067200_000000, 1704067260_000000  # two consecutive minutes
    rows = [  # (price, qty, time_us, is_buyer_maker)
        (100.0, 1000.0, m0 + 1, False),  # one taker buy sweeping two levels: $100,101 -> large
        (101.0, 1.0, m0 + 1, False),
        (99.0, 0.005, m0 + 2, True),     # small sell ($0.50)
        (98.0, 20.0, m1 + 5, True),      # next minute: medium sell ($1,960)
        (97.0, 20.0, m1 + 6, False),     # medium buy ($1,940)
    ]

    def batch(rs):
        return pa.table({"price": [r[0] for r in rs], "quantity": [r[1] for r in rs],
                         "transact_time_us": [r[2] for r in rs], "is_buyer_maker": [r[3] for r in rs]})

    # Split mid-minute so minute 1's stats have to be merged from two batches.
    t = build([batch(rows[:3]), batch(rows[3:4]), batch(rows[4:])]).to_pandas()
    assert t.open_time_us.tolist() == [m0, m1]
    b0, b1 = t.iloc[0], t.iloc[1]
    assert (b0.open, b0.high, b0.low, b0.close) == (100.0, 101.0, 99.0, 99.0)
    assert b0.trades == 3 and b0.taker_orders == 2 and b0.buy_taker_orders == 1
    assert b0.max_order_qty == 1001.0 and b0.large_buy_volume == 1001.0 and b0.small_volume == 0.005
    assert (b1.open, b1.close, b1.volume, b1.buy_volume) == (98.0, 97.0, 40.0, 20.0)
    assert b1.taker_orders == 2 and b1.large_volume == 0 and b1.small_volume == 0


def test_fetch_rejects_bad_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(vision.urllib.request, "urlopen", FakeUrlopen(b"corrupt", "0" * 64))
    with pytest.raises(ValueError, match="checksum"):
        fetch(_spot_item("2024-01"), tmp_path, "BTCUSDT", threading.Event(), attempts=1)
    assert not list(tmp_path.rglob("*.parquet")) and not list(tmp_path.rglob("*.tmp"))
