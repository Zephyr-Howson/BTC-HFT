"""Step 3, pre-registered: does a different model give a better signal than the baseline composite?

Every model is sized the same way, the D2 rule from backtest/variants.py: past-percentile of the daily
prediction, 5-day mean, flat below the 200-day average, 10-point band. Only the signal changes.

    M0  baseline:        composite, top 10 features with |t| >= 3, 24h target  (= D2)
    M1  broad:           top 25 with |t| >= 2, equal weights
    M2  t-weighted:      as M1, each feature weighted by its training |t|
    M3  groups:          fixed economic groups (models.GROUPS); members |t| >= 1, groups |t| >= 2
    M4  two horizons:    M0 fitted on the 4h and on the 24h target, daily percentiles averaged
  Not pre-registered (suggested after seeing round 2):
    M0 no trend gate:    M0 sized without the 200-day filter

stability = mean overlap (Jaccard) of the chosen features or groups between consecutive monthly refits:
1 means the model never changes its mind, near 0 means it's chasing noise.

Uses the development period only (data from variants.HOLDOUT on is cut off first).
Run with: python -m backtest.refine [--features ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import feature_columns

from .engine import execution_prices, run
from .models import GroupComposite, RankComposite
from .rules import at_hour, past_percentile, smooth, trend_gate, with_band
from .stats import t_stat
from .variants import COLS, HOLDOUT, evaluate
from .walkforward import information_coefficient, walk_forward

MODELS = {
    "M0 baseline": lambda: RankComposite(top=10, min_t=3.0),
    "M1 broad": lambda: RankComposite(top=25, min_t=2.0),
    "M2 t-weighted": lambda: RankComposite(top=25, min_t=2.0, weighting="t"),
    "M3 groups": lambda: GroupComposite(),
}


def stability(fitted: list) -> float:
    sets = [set(m.chosen.index) for _, m in fitted]
    overlap = [len(a & b) / len(a | b) for a, b in zip(sets, sets[1:]) if a | b]
    return float(np.mean(overlap)) if overlap else np.nan


def d2(size: pd.Series, trend: pd.Series | None) -> pd.Series:
    w = smooth(size, 5)
    return with_band(trend_gate(w, trend) if trend is not None else w, 0.1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", default="data/features/BTCUSDT_1h.parquet")
    p.add_argument("--history", default="data/history")
    p.add_argument("--min-train-months", type=int, default=24)
    p.add_argument("--random", type=int, default=200)
    p.add_argument("--cache", default="data/backtests/refine_preds.parquet",
                   help="fitted predictions are saved here and reused; delete it after changing a model")
    a = p.parse_args()

    df = pd.read_parquet(a.features)
    df = df[df.index < pd.Timestamp(HOLDOUT, tz="UTC")]
    feats = feature_columns(df)
    daily = at_hour(df, 0)

    cache = Path(a.cache)
    if cache.exists():                     # daily predictions and stabilities from an earlier run
        saved = pd.read_parquet(cache)
        preds = {k: saved[k] for k in saved.columns if k != "M0 4h"}
        p4 = saved["M0 4h"]
        stab = pd.read_json(cache.with_suffix(".json"), typ="series").to_dict()
    else:
        preds, stab = {}, {}
        for name, make in [*MODELS.items(), ("M0 4h", MODELS["M0 baseline"])]:
            fitted: list = []
            target = "fwd_ret_4h" if name == "M0 4h" else "fwd_ret_24h"
            preds[name] = walk_forward(df, feats, target, make, a.min_train_months, fitted=fitted).reindex(daily.index)
            stab[name] = stability(fitted)
            print(f"fitted {name}", flush=True)
        p4 = preds.pop("M0 4h")
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({**preds, "M0 4h": p4}).to_parquet(cache)
        pd.Series(stab).to_json(cache.with_suffix(".json"))

    sizes = {k: past_percentile(v.dropna()) for k, v in preds.items()}
    sizes["M4 two horizons"] = pd.concat([sizes["M0 baseline"], past_percentile(p4.dropna())],
                                         axis=1).mean(axis=1, skipna=False)
    stab["M4 two horizons"] = (stab["M0 baseline"] + stab["M0 4h"]) / 2
    start = max(s.first_valid_index() for s in sizes.values())
    daily = daily.loc[start:]
    sizes = {k: s.reindex(daily.index) for k, s in sizes.items()}
    trend = daily["px_vs_ma_200d"]
    px = execution_prices(daily.index, a.history)

    weights = {k: d2(s, trend) for k, s in sizes.items()}
    weights["M0 no trend gate (new)"] = d2(sizes["M0 baseline"], None)
    rows = {}
    for k, w in weights.items():
        r = evaluate(w, px, a.random)
        sig = sizes.get(k, sizes["M0 baseline"])
        ic = information_coefficient(sig, daily["fwd_ret_24h"], "fwd_ret_24h")
        r.update({"ic": ic.mean(), "ic_t": t_stat(ic), "stability": stab.get(k, stab["M0 baseline"])})
        rows[k] = r
    bh = run(pd.Series(1.0, index=daily.index), px).summary()

    print(f"\ndevelopment period (locked from {HOLDOUT}): {daily.index[0]:%Y-%m-%d} -> {daily.index[-1]:%Y-%m-%d}; "
          f"buy & hold Sharpe {bh['sharpe']:.2f}, max drawdown {bh['max_drawdown']:.0%}\n")
    cols = ["ic", "ic_t", "stability"] + [c for c in COLS if c not in ("time_in_market",)]
    print(pd.DataFrame(rows).T[cols].to_string(float_format=lambda v: f"{v:,.2f}"))
    results = {k: run(w, px) for k, w in weights.items()}
    yearly = pd.DataFrame({k: r.returns.groupby(r.returns.index.year).apply(lambda x: (1 + x).prod() - 1)
                           for k, r in results.items()})
    print("\nreturn by year:\n" + yearly.T.to_string(float_format=lambda v: f"{v:+.0%}"))


if __name__ == "__main__":
    main()
