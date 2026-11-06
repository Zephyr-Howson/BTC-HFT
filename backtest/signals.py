"""Does any single feature predict forward returns? Run before building models on them.

For each feature and target horizon:
- ic, t:   mean monthly predictive IC (backtest/stats.py) and its Newey-West t-stat. The statistic
           is centred on zero when returns are unpredictable, even for slow price-level features
           that fool a plain within-month rank correlation. With ~60 features x 3 horizons a few
           will pass |t| > 2 by luck, so treat that as a first filter.
- pos:     share of months with a positive IC (a consistent sign matters more than its size)
- early / late: mean IC in the first and second half of the months, to catch decayed signals
- top_bot_bps: mean forward return when the feature is in the top fifth of its trailing 90 days,
           minus when it's in the bottom fifth. Uses past-only percentiles and raw returns, so it is
           what a trader could have captured, before costs (~22 bps per round trip).

Run with: python -m backtest.signals [--features data/features/BTCUSDT_1h.parquet] [--start 2020-01]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from features import feature_columns, target_columns

from .stats import predictive_ic, t_stat, trailing_percentile
from .walkforward import horizon


def report(df: pd.DataFrame, features: list[str] | None = None, targets: list[str] | None = None) -> pd.DataFrame:
    """One row per (target, feature)."""
    out = []
    for f in features or feature_columns(df):
        pct = trailing_percentile(df[f])
        for t in targets or target_columns(df):
            ic = predictive_ic(df[f], df[t], horizon(t))
            if len(ic) < 6:
                continue
            half = len(ic) // 2
            y = df[t]
            out.append({"target": t, "feature": f, "months": len(ic), "ic": ic.mean(), "t": t_stat(ic),
                        "pos": (ic > 0).mean(), "early": ic.iloc[:half].mean(), "late": ic.iloc[half:].mean(),
                        "top_bot_bps": (y[pct > 0.3].mean() - y[pct < -0.3].mean()) * 1e4})
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Per-feature predictive power (monthly predictive IC).")
    p.add_argument("--features", default="data/features/BTCUSDT_1h.parquet")
    p.add_argument("--start", default="", help="only use rows from this date")
    p.add_argument("--top", type=int, default=15, help="rows shown per target")
    a = p.parse_args()
    df = pd.read_parquet(a.features)
    if a.start:
        df = df[df.index >= pd.Timestamp(a.start, tz="UTC")]
    r = report(df)
    fmt = {"ic": "{:+.3f}", "t": "{:+.1f}", "pos": "{:.0%}", "early": "{:+.3f}", "late": "{:+.3f}",
           "top_bot_bps": "{:+.1f}"}
    for t in target_columns(df):
        g = r[r.target == t]
        g = g.reindex(g.t.abs().sort_values(ascending=False).index)
        print(f"\n== {t}  (top {min(a.top, len(g))} of {len(g)} features by |t|)")
        print(g.head(a.top).drop(columns="target").to_string(index=False,
                                                             formatters={k: v.format for k, v in fmt.items()}))
        print(f"   features with |t| > 2: {(g.t.abs() > 2).sum()}   (|t| > 3: {(g.t.abs() > 3).sum()})")


if __name__ == "__main__":
    main()
