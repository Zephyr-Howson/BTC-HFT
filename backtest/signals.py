"""Does any single feature predict forward returns? Run before building models on them.

For each feature and target horizon:
- ic, t:   mean monthly rank IC and its t-stat (mean / standard error across months). Months are
           close to independent, so |t| > 2 is a first filter, not proof: with ~50 features x 3
           horizons, a few will pass by luck.
- pos:     share of months with a positive IC (a consistent sign matters more than its size)
- early / late: mean IC in the first and second half of the months, to catch decayed signals
- q5_q1:   mean forward return after the top fifth of readings minus the bottom fifth, in bps.
           Quintiles are cut within each month, so slow drifts in a feature's level don't count.
           Compare with the ~22 bps round-trip cost: the spread has to beat it to be tradable.

Run with: python -m backtest.signals [--features data/features/BTCUSDT_1h.parquet] [--start 2020-01]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from features import feature_columns, target_columns

from .walkforward import _months


def _monthly(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    both = pd.concat({"x": x, "y": y}, axis=1).dropna()
    rows = {}
    for m, g in both.groupby(_months(both.index)):
        if len(g) < 100 or g.x.nunique() < 5:
            continue
        q = pd.qcut(g.x.rank(method="first"), 5, labels=False)
        rows[m] = {"ic": g.x.rank().corr(g.y.rank()),
                   "spread": g.y[q == 4].mean() - g.y[q == 0].mean()}
    return pd.DataFrame(rows).T


def report(df: pd.DataFrame, features: list[str] | None = None, targets: list[str] | None = None) -> pd.DataFrame:
    """One row per (target, feature)."""
    out = []
    for t in targets or target_columns(df):
        for f in features or feature_columns(df):
            m = _monthly(df[f], df[t])
            if len(m) < 6:
                continue
            ic = m.ic.astype(float)
            half = len(ic) // 2
            out.append({"target": t, "feature": f, "months": len(ic), "ic": ic.mean(),
                        "t": ic.mean() / (ic.std() / np.sqrt(len(ic))), "pos": (ic > 0).mean(),
                        "early": ic.iloc[:half].mean(), "late": ic.iloc[half:].mean(),
                        "q5_q1_bps": m.spread.astype(float).mean() * 1e4})
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Per-feature predictive power (monthly rank IC).")
    p.add_argument("--features", default="data/features/BTCUSDT_1h.parquet")
    p.add_argument("--start", default="", help="only use rows from this date")
    p.add_argument("--top", type=int, default=15, help="rows shown per target")
    a = p.parse_args()
    df = pd.read_parquet(a.features)
    if a.start:
        df = df[df.index >= pd.Timestamp(a.start, tz="UTC")]
    r = report(df)
    fmt = {"ic": "{:+.3f}", "t": "{:+.1f}", "pos": "{:.0%}", "early": "{:+.3f}", "late": "{:+.3f}",
           "q5_q1_bps": "{:+.1f}"}
    for t, g in r.groupby("target", sort=False):
        g = g.reindex(g.t.abs().sort_values(ascending=False).index).head(a.top)
        print(f"\n== {t}  (top {len(g)} of {(r.target == t).sum()} features by |t|)")
        print(g.drop(columns="target").to_string(index=False, formatters={k: v.format for k, v in fmt.items()}))
        print(f"   features with |t| > 2: {(r[r.target == t].t.abs() > 2).sum()}")


if __name__ == "__main__":
    main()
