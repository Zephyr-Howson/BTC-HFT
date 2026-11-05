"""Dataset definitions and CSV -> Parquet conversion for data.binance.vision files.

The raw files are inconsistent, so conversion normalises them:
- Spot files have no header; newer futures files do, older ones don't.
- Spot timestamps switched from milliseconds to microseconds on 2025-01-01.
- Some files (e.g. futures metrics) are unsorted and contain duplicate rows.

Output: every timestamp is int64 UTC microseconds, renamed <col>_us.
Files land in <root>/<market>/<dataset>/<SYMBOL>/<YYYY-MM or YYYY-MM-DD>.parquet.
Integer columns are delta-encoded: ids and timestamps only ever increase, so
this roughly halves the size of tick data, losslessly.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

I64, F64, BOOL, TS = pa.int64(), pa.float64(), pa.bool_(), pa.timestamp("us")

# Epoch values below this are milliseconds; at or above it, microseconds.
# (1e14 ms is the year 5138; 1e14 us is 1973.)
_US_THRESHOLD = 10**14


@dataclass(frozen=True)
class Dataset:
    name: str                       # e.g. "spot/aggTrades"
    path: str                       # bucket path, {period} = monthly|daily
    columns: tuple[str, ...]        # every CSV column, in file order
    keep: dict[str, pa.DataType]    # columns written to Parquet, with types
    times: tuple[str, ...]          # timestamp columns (epoch ints or datetime strings)
    sort_by: tuple[str, ...] = ()   # if set: load whole file, sort and drop duplicates
    bars: str = ""                  # tick data: streamed, summarised into this 1m bar dataset,
                                    # and limited by --trades-start
    schema: pa.Schema = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        fields = [(f"{c}_us" if c in self.times else c, I64 if c in self.times else t)
                  for c, t in self.keep.items()]
        object.__setattr__(self, "schema", pa.schema(fields))

    def prefix(self, period: str, symbol: str) -> str:
        return "data/" + self.path.format(period=period, symbol=symbol) + "/"

    def out_dir(self, root: Path, symbol: str) -> Path:
        return Path(root) / self.name / symbol


KLINE_COLS = ("open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore")
KLINE_KEEP = {"open_time": I64, "open": F64, "high": F64, "low": F64, "close": F64, "volume": F64,
              "quote_volume": F64, "count": I64, "taker_buy_volume": F64, "taker_buy_quote_volume": F64}
AGG_COLS = ("agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time",
            "is_buyer_maker")
AGG_KEEP = {"agg_trade_id": I64, "price": F64, "quantity": F64, "first_trade_id": I64,
            "last_trade_id": I64, "transact_time": I64, "is_buyer_maker": BOOL}

DATASETS = {d.name: d for d in [
    Dataset("spot/klines_1m", "spot/{period}/klines/{symbol}/1m", KLINE_COLS, KLINE_KEEP,
            times=("open_time",), sort_by=("open_time",)),
    Dataset("spot/aggTrades", "spot/{period}/aggTrades/{symbol}", AGG_COLS + ("is_best_match",),
            AGG_KEEP, times=("transact_time",), bars="spot/tradeBars_1m"),
    Dataset("futures/klines_1m", "futures/um/{period}/klines/{symbol}/1m", KLINE_COLS, KLINE_KEEP,
            times=("open_time",), sort_by=("open_time",)),
    Dataset("futures/aggTrades", "futures/um/{period}/aggTrades/{symbol}", AGG_COLS, AGG_KEEP,
            times=("transact_time",), bars="futures/tradeBars_1m"),
    # Perp premium over the index (a basis measure), as 1m OHLC.
    Dataset("futures/premiumIndex_1m", "futures/um/{period}/premiumIndexKlines/{symbol}/1m", KLINE_COLS,
            {"open_time": I64, "open": F64, "high": F64, "low": F64, "close": F64},
            times=("open_time",), sort_by=("open_time",)),
    Dataset("futures/fundingRate", "futures/um/{period}/fundingRate/{symbol}",
            ("calc_time", "funding_interval_hours", "last_funding_rate"),
            {"calc_time": I64, "funding_interval_hours": I64, "last_funding_rate": F64},
            times=("calc_time",), sort_by=("calc_time",)),
    # 5-minute open interest and long/short ratios.
    Dataset("futures/metrics", "futures/um/{period}/metrics/{symbol}",
            ("create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
             "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
             "count_long_short_ratio", "sum_taker_long_short_vol_ratio"),
            {"create_time": TS, "sum_open_interest": F64, "sum_open_interest_value": F64,
             "count_toptrader_long_short_ratio": F64, "sum_toptrader_long_short_ratio": F64,
             "count_long_short_ratio": F64, "sum_taker_long_short_vol_ratio": F64},
            times=("create_time",), sort_by=("create_time",)),
    # Book depth within +/-1..5% of mid, sampled roughly every 30s. Negative percentage = bids.
    Dataset("futures/bookDepth", "futures/um/{period}/bookDepth/{symbol}",
            ("timestamp", "percentage", "depth", "notional"),
            {"timestamp": TS, "percentage": F64, "depth": F64, "notional": F64},
            times=("timestamp",), sort_by=("timestamp", "percentage")),
]}


def _has_header(zf: zipfile.ZipFile, member: str) -> bool:
    with zf.open(member) as f:
        first = f.readline().lstrip()
    return bool(first) and not (first[:1].isdigit() or first[:1] == b"-")


def _normalise(table: pa.Table | pa.RecordBatch, ds: Dataset) -> pa.Table:
    arrays = []
    for col in ds.keep:
        arr = table.column(col)
        if col in ds.times:
            if pa.types.is_timestamp(arr.type):
                arr = arr.cast(I64)
            else:
                arr = pc.if_else(pc.less(arr, _US_THRESHOLD), pc.multiply(arr, pa.scalar(1000, I64)), arr)
        arrays.append(arr)
    return pa.Table.from_arrays(arrays, schema=ds.schema)


def parquet_options(schema: pa.Schema) -> dict:
    """zstd, with integer columns delta-encoded and everything else dictionary-encoded."""
    ints = [f.name for f in schema if pa.types.is_integer(f.type)]
    return {"compression": "zstd", "compression_level": 9,
            "use_dictionary": [n for n in schema.names if n not in ints],
            "column_encoding": {n: "DELTA_BINARY_PACKED" for n in ints}}


def write_table(table: pa.Table, path: Path) -> None:
    """Write atomically: readers never see half-written files."""
    tmp = path.with_suffix(".parquet.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp, **parquet_options(table.schema))
    os.replace(tmp, path)


def convert(zip_path: Path, out_path: Path, ds: Dataset, on_batch: Callable[[pa.Table], None] | None = None) -> int:
    """Convert one downloaded zip to Parquet, written atomically. Returns rows written.

    For streamed (tick) datasets, on_batch is called with each converted batch in order.
    """
    tmp = out_path.with_suffix(".parquet.tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.endswith(".csv"))
        read = pacsv.ReadOptions(column_names=list(ds.columns), skip_rows=int(_has_header(zf, member)),
                                 block_size=64 << 20)
        conv = pacsv.ConvertOptions(column_types=ds.keep, include_columns=list(ds.keep))
        with zf.open(member) as f:
            reader = pacsv.open_csv(f, read_options=read, convert_options=conv)
            if ds.sort_by:
                table = _normalise(reader.read_all(), ds)
                keys = [f"{c}_us" if c in ds.times else c for c in ds.sort_by]
                df = table.to_pandas().drop_duplicates().sort_values(keys, kind="stable")
                table = pa.Table.from_pandas(df, schema=ds.schema, preserve_index=False)
                write_table(table, out_path)
                return table.num_rows
            with pq.ParquetWriter(tmp, ds.schema, **parquet_options(ds.schema)) as writer:
                for batch in reader:
                    table = _normalise(batch, ds)
                    writer.write_table(table)
                    if on_batch:
                        on_batch(table)
                    rows += table.num_rows
    os.replace(tmp, out_path)  # atomic: readers never see half-written files
    return rows


def in_range(label: str, start: str = "", end: str = "9999") -> bool:
    """Does a file label (YYYY-MM or YYYY-MM-DD) overlap [start, end]? Bounds use the same formats."""
    return label >= start[:len(label)] and label[:len(end)] <= end


def load(name: str, root: str | Path = "data/history", symbol: str = "BTCUSDT",
         start: str = "", end: str = "9999", columns: list[str] | None = None) -> pd.DataFrame:
    """Load a dataset or bar set, e.g. load("spot/tradeBars_1m", start="2024-01", end="2024-06")."""
    files = sorted(f for f in (Path(root) / name / symbol).glob("*.parquet") if in_range(f.stem, start, end))
    if not files:
        return pd.DataFrame(columns=columns)
    return pd.concat((pd.read_parquet(f, columns=columns) for f in files), ignore_index=True)
