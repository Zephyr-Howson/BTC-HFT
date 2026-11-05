"""Run with: python -m features [--history data/history] [--out data/features] [--freq 1h]"""

from __future__ import annotations

import argparse
from pathlib import Path

from .build import build, feature_columns, target_columns


def main() -> None:
    p = argparse.ArgumentParser(description="Build the feature table from the downloaded history.")
    p.add_argument("--history", default="data/history", help="history root (python -m history --out)")
    p.add_argument("--out", default="data/features")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--freq", default="1h", help="sample spacing, e.g. 1h, 15min")
    p.add_argument("--start", default="", help="YYYY-MM or YYYY-MM-DD")
    p.add_argument("--end", default="9999")
    a = p.parse_args()

    df = build(a.history, a.symbol, a.start, a.end, a.freq)
    out = Path(a.out) / f"{a.symbol}_{a.freq}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="zstd")
    print(f"{len(df):,} rows  {df.index[0]} -> {df.index[-1]}  -> {out}")
    print("coverage (share of rows with a value) and first date with data:")
    for c in feature_columns(df) + target_columns(df):
        s = df[c].dropna()
        first = f"{s.index[0]:%Y-%m-%d}" if len(s) else "-"
        print(f"  {c:24s} {len(s) / len(df):6.1%}  from {first}")


if __name__ == "__main__":
    main()
