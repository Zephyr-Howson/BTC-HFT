"""Quick health check of recorded data, spot and futures.

Usage: python scripts/inspect_data.py [data] [btcusdt]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SPOT = ("trades", "book_ticker", "depth_updates", "depth_snapshots")
FUTURES = ("agg_trades", "mark_price", "liquidations", "open_interest")


def load(path: Path) -> pd.DataFrame | None:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return None
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def latency(df: pd.DataFrame) -> str:
    lat = df.recv_ns / 1e6 - df.event_ms
    return f"feed latency ms (recv - exchange event): p50={lat.median():.1f} p99={lat.quantile(0.99):.1f}"


def spot_checks(stream: str, df: pd.DataFrame) -> None:
    if stream == "trades":
        print(f"{'':16s} {latency(df)}  last px={df.price.iloc[-1]:,.2f}")
    if stream == "depth_updates":
        # Order by Binance's ids, not recv_ns: on Windows the clock ticks every 1-16 ms, so
        # back-to-back messages can share a recv_ns and a sort on it would shuffle them.
        ev = df.drop_duplicates(["session", "final_update_id"]).sort_values(["session", "final_update_id"])
        gaps = 0
        for _, g in ev.groupby("session"):
            gaps += int((g.first_update_id.values[1:] != g.final_update_id.values[:-1] + 1).sum())
        print(f"{'':16s} events={len(ev):,} sessions={ev.session.nunique()} sequence gaps={gaps}")
    if stream == "depth_snapshots":
        print(f"{'':16s} snapshots by reason: {df.groupby('reason').snapshot_id.nunique().to_dict()}")


def futures_checks(stream: str, df: pd.DataFrame, symbol: str) -> None:
    if stream == "agg_trades":
        ids = df.agg_trade_id.sort_values().drop_duplicates()
        missing = int((ids.diff().dropna() - 1).sum())
        print(f"{'':16s} {latency(df)}  last px={df.price.iloc[-1]:,.2f}  missing agg ids={missing:,}")
    if stream == "mark_price":
        last = df.iloc[-1]
        span_s = (df.recv_ns.iloc[-1] - df.recv_ns.iloc[0]) / 1e9
        print(f"{'':16s} last mark={last.mark_px:,.2f} index={last.index_px:,.2f} "
              f"premium={(last.mark_px / last.index_px - 1) * 1e4:+.2f} bps funding={last.funding_rate:+.5%}  "
              f"coverage={len(df) / max(span_s, 1):.0%} of seconds")
    if stream == "liquidations":
        ours = df[df.symbol == symbol.upper()]
        notional = ours.avg_price * ours.filled_qty
        print(f"{'':16s} symbols={df.symbol.nunique()}  {symbol.upper()}: {len(ours)} "
              f"(longs liquidated ${notional[ours.side == 'SELL'].sum():,.0f}, "
              f"shorts ${notional[ours.side == 'BUY'].sum():,.0f})")
    if stream == "open_interest":
        print(f"{'':16s} last open interest={df.open_interest.iloc[-1]:,.3f}")


def main() -> None:
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    symbol = (sys.argv[2] if len(sys.argv) > 2 else "btcusdt").lower()
    for market, streams in (("binance", SPOT), ("binance_futures", FUTURES)):
        print(f"== {market}/{symbol}")
        for stream in streams:
            df = load(data / market / symbol / stream)
            if df is None:
                print(f"{stream:16s} no data")
                continue
            t0 = pd.to_datetime(df.recv_ns.min(), unit="ns", utc=True)
            t1 = pd.to_datetime(df.recv_ns.max(), unit="ns", utc=True)
            print(f"{stream:16s} rows={len(df):>11,}  {t0:%Y-%m-%d %H:%M:%S} -> {t1:%Y-%m-%d %H:%M:%S} UTC")
            if market == "binance":
                spot_checks(stream, df)
            else:
                futures_checks(stream, df, symbol)


if __name__ == "__main__":
    main()
