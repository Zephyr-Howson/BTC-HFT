"""Pre-registered comparison of long-only daily strategies. Run once; don't tune on the result.

    A  buy & hold
    B  trend only:      fully long while price > its 200-day average, else flat
    C  sizing only:     weight = today's composite prediction as a percentile of all past ones
    D  sizing + trend:  C, but flat while price < its 200-day average

All: one decision a day at 00:00 UTC, filled at the next minute's VWAP; rebalance only when the
target weight moves >= 10 points (or to exactly 0/1); 10 bps fee + 1 bp slippage. The composite is
refit monthly on hourly rows (walk-forward, 24h target), then sampled at the daily decision time.

Run with: python -m backtest.variants [--features ...] [--ma 200] [--band 0.1]
"""

from __future__ import annotations

import argparse

import pandas as pd

from features import feature_columns

from .engine import execution_prices, run
from .models import RankComposite
from .stats import t_stat
from .rules import at_hour, past_percentile, trend_gate, with_band
from .walkforward import information_coefficient, walk_forward

COLS = ["total_return", "cagr", "sharpe", "max_drawdown", "time_in_market", "trades", "turnover_per_year", "costs"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", default="data/features/BTCUSDT_1h.parquet")
    p.add_argument("--history", default="data/history")
    p.add_argument("--target", default="fwd_ret_24h")
    p.add_argument("--ma", type=int, default=200, help="trend filter moving average, days")
    p.add_argument("--band", type=float, default=0.1)
    p.add_argument("--hour", type=int, default=0, help="daily decision time, UTC hour")
    p.add_argument("--min-train-months", type=int, default=24)
    a = p.parse_args()

    df = pd.read_parquet(a.features)
    pred = walk_forward(df, feature_columns(df), a.target, RankComposite, a.min_train_months)
    daily = at_hour(df, a.hour).loc[pred.dropna().index[0]:]
    pred = pred.reindex(daily.index)
    px = execution_prices(daily.index, a.history)
    trend = daily[f"px_vs_ma_{a.ma}d"]

    size = past_percentile(pred)
    weights = {
        "A buy & hold": pd.Series(1.0, index=daily.index),
        "B trend only": trend_gate(pd.Series(1.0, index=daily.index), trend),
        "C sizing only": with_band(size, a.band),
        "D sizing + trend": with_band(trend_gate(size, trend), a.band),
    }
    results = {k: run(w, px) for k, w in weights.items()}

    ic = information_coefficient(pred, daily[a.target], a.target)
    print(f"daily at {a.hour:02d}:00 UTC, {daily.index[0]:%Y-%m-%d} -> {daily.index[-1]:%Y-%m-%d}; "
          f"composite predictive IC at decision times: {ic.mean():+.3f} (t={t_stat(ic):+.1f}), "
          f"positive in {(ic > 0).mean():.0%} of {len(ic)} months\n")
    table = pd.DataFrame({k: r.summary() for k, r in results.items()}).T[COLS]
    print(table.to_string(float_format=lambda v: f"{v:,.2f}"))
    yearly = pd.DataFrame({k: r.returns.groupby(r.returns.index.year).apply(lambda x: (1 + x).prod() - 1)
                           for k, r in results.items()})
    print("\nreturn by year:\n" + yearly.to_string(float_format=lambda v: f"{v:+.1%}"))


if __name__ == "__main__":
    main()
