"""Pre-registered comparison of long-only daily strategies. Don't tune on the result.

    A   buy & hold
    B   trend only:      fully long while price > its 200-day average, else flat
    C   sizing only:     weight = today's composite prediction as a percentile of all past ones
    D   sizing + trend:  C, but flat while price < its 200-day average
  Cost reduction (round 2), each a change to D:
    D1  wider band:      rebalance only on moves >= 25 points
    D2  smoothed:        target = mean of the last 5 days' sizes
    D3  thirds:          target rounded to 0, 1/3, 2/3 or 1
    D4  smoothed + wide: D2 and D1 together

All: one decision a day at 00:00 UTC, filled at the next minute's VWAP; rebalance only when the
target weight moves >= the band (default 10 points) or to exactly 0/1; 10 bps fee + 1 bp slippage.
The composite is refit monthly on hourly rows (walk-forward, 24h target), then sampled daily.

Each strategy is also shown
- before costs (sharpe_gross), to separate a lack of edge from an edge eaten by fees
- against 200 copies of itself shifted to random dates (random_sharpe = their median, beats_random =
  share it beats). Same exposure and turnover, random timing: if a strategy doesn't beat these, its
  Sharpe comes from holding less BTC, not from skill.

Locked final test: everything from HOLDOUT (Nov 2025) on is cut off the data before any model is fit,
so refinement never sees it. `--final` runs on the locked period instead. Run that once, at the end.

Run with: python -m backtest.variants [--final] [--features ...] [--ma 200]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from features import feature_columns

from .engine import execution_prices, random_timing, run
from .models import RankComposite
from .rules import at_hour, past_percentile, smooth, steps, trend_gate, with_band
from .stats import t_stat
from .walkforward import information_coefficient, walk_forward

HOLDOUT = "2025-11-01"
COLS = ["total_return", "cagr", "sharpe", "sharpe_gross", "random_sharpe", "beats_random", "max_drawdown",
        "exposure", "time_in_market", "turnover_per_year", "costs"]


def strategies(size: pd.Series, trend: pd.Series, band: float = 0.1) -> dict[str, pd.Series]:
    """Target weights by name. Sizes are past-only percentiles in [0, 1]; trend is log(price / MA)."""
    one = pd.Series(1.0, index=size.index)
    gate = lambda w: trend_gate(w, trend)  # noqa: E731
    return {
        "A buy & hold": one,
        "B trend only": gate(one),
        "C sizing only": with_band(size, band),
        "D sizing + trend": with_band(gate(size), band),
        "D1 band 25": with_band(gate(size), 0.25),
        "D2 smoothed 5d": with_band(gate(smooth(size, 5)), band),
        "D3 thirds": with_band(gate(steps(size, 3)), band),
        "D4 smoothed + band 25": with_band(gate(smooth(size, 5)), 0.25),
    }


def evaluate(weights: pd.Series, prices: pd.Series, n_random: int = 200) -> dict[str, float]:
    s = run(weights, prices).summary()
    s["sharpe_gross"] = run(weights, prices, 0.0, 0.0).summary()["sharpe"]
    rnd = random_timing(weights, prices, n_random)
    s["random_sharpe"] = float(np.nanmedian(rnd))
    s["beats_random"] = float((s["sharpe"] > rnd).mean())
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", default="data/features/BTCUSDT_1h.parquet")
    p.add_argument("--history", default="data/history")
    p.add_argument("--target", default="fwd_ret_24h")
    p.add_argument("--ma", type=int, default=200, help="trend filter moving average, days")
    p.add_argument("--band", type=float, default=0.1)
    p.add_argument("--hour", type=int, default=0, help="daily decision time, UTC hour")
    p.add_argument("--min-train-months", type=int, default=24)
    p.add_argument("--random", type=int, default=200, help="random-timing copies per strategy")
    p.add_argument("--final", action="store_true", help=f"evaluate on the locked period from {HOLDOUT}")
    a = p.parse_args()

    df = pd.read_parquet(a.features)
    cut = pd.Timestamp(HOLDOUT, tz="UTC")
    if not a.final:
        df = df[df.index < cut]
    pred = walk_forward(df, feature_columns(df), a.target, RankComposite, a.min_train_months)
    daily = at_hour(df, a.hour)
    size = past_percentile(pred.reindex(daily.index).dropna())  # percentile among all earlier predictions
    start = cut if a.final else size.index[0]
    daily, size = daily.loc[start:], size.loc[start:]
    px = execution_prices(daily.index, a.history)

    weights = strategies(size, daily[f"px_vs_ma_{a.ma}d"], a.band)
    table = pd.DataFrame({k: evaluate(w, px, a.random) for k, w in weights.items()}).T[COLS]
    results = {k: run(w, px) for k, w in weights.items()}

    ic = information_coefficient(pred.reindex(daily.index), daily[a.target], a.target)
    period = "LOCKED FINAL TEST" if a.final else f"development period (locked from {HOLDOUT})"
    print(f"{period}: daily at {a.hour:02d}:00 UTC, {daily.index[0]:%Y-%m-%d} -> {daily.index[-1]:%Y-%m-%d}")
    print(f"composite predictive IC at decision times: {ic.mean():+.3f} (t={t_stat(ic):+.1f}), "
          f"positive in {(ic > 0).mean():.0%} of {len(ic)} months\n")
    print(table.to_string(float_format=lambda v: f"{v:,.2f}"))
    yearly = pd.DataFrame({k: r.returns.groupby(r.returns.index.year).apply(lambda x: (1 + x).prod() - 1)
                           for k, r in results.items()})
    print("\nreturn by year:\n" + yearly.T.to_string(float_format=lambda v: f"{v:+.0%}"))


if __name__ == "__main__":
    main()
