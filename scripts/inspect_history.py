"""Coverage report for downloaded history: date range, rows, and missing minutes in 1m data.

Usage: python scripts/inspect_history.py [data/history] [BTCUSDT]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history.datasets import DATASETS, load  # noqa: E402

MINUTE_US = 60_000_000


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/history")
    symbol = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"
    names = [n for ds in DATASETS.values() for n in (ds.name, ds.bars) if n]
    for name in names:
        files = sorted((root / name / symbol).glob("*.parquet"))
        if not files:
            print(f"{name:24s} no data")
            continue
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        size = sum(f.stat().st_size for f in files) / 1e9
        print(f"{name:24s} {files[0].stem} -> {files[-1].stem}  files={len(files):5d}  rows={rows:>13,}  {size:.2f} GB")

        if name.endswith("_1m"):
            t = load(name, root, symbol, columns=["open_time_us"]).open_time_us
            step = t.diff().dropna()
            gaps = step[step > MINUTE_US]
            missing = int(((gaps // MINUTE_US) - 1).sum())
            print(f"{'':24s} missing minutes={missing:,} in {len(gaps)} gaps", end="")
            if len(gaps):
                worst = gaps.idxmax()
                start = pd.to_datetime(t[worst - 1], unit="us", utc=True)
                print(f"; longest {gaps.max() / MINUTE_US - 1:,.0f} min from {start:%Y-%m-%d %H:%M} UTC", end="")
            print()


if __name__ == "__main__":
    main()
