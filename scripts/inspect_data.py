"""Quick health check of recorded data.

Usage: python scripts/inspect_data.py [data/binance/btcusdt]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame | None:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return None
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/binance/btcusdt")
    for stream in ("trades", "book_ticker", "depth_updates", "depth_snapshots"):
        df = load(root / stream)
        if df is None:
            print(f"{stream:16s} no data")
            continue
        t0 = pd.to_datetime(df.recv_ns.min(), unit="ns", utc=True)
        t1 = pd.to_datetime(df.recv_ns.max(), unit="ns", utc=True)
        print(f"{stream:16s} rows={len(df):>11,}  {t0:%Y-%m-%d %H:%M:%S} -> {t1:%Y-%m-%d %H:%M:%S} UTC")

        if stream == "trades":
            lat = df.recv_ns / 1e6 - df.event_ms
            print(f"{'':16s} feed latency ms (recv - exchange event): "
                  f"p50={lat.median():.1f} p99={lat.quantile(0.99):.1f}  "
                  f"last px={df.price.iloc[-1]:,.2f}")
        if stream == "depth_updates":
            ev = df.drop_duplicates(["session", "final_update_id"]).sort_values("recv_ns")
            gaps = 0
            for _, g in ev.groupby("session"):
                gaps += int((g.first_update_id.values[1:] != g.final_update_id.values[:-1] + 1).sum())
            print(f"{'':16s} events={len(ev):,} sessions={ev.session.nunique()} sequence gaps={gaps}")
        if stream == "depth_snapshots":
            print(f"{'':16s} snapshots by reason: {df.groupby('reason').snapshot_id.nunique().to_dict()}")


if __name__ == "__main__":
    main()
