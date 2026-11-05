"""Run with: python -m backtest [--target fwd_ret_4h] [--model ridge] [--enter-bps 25]"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features import feature_columns

from .engine import FEE_BPS, SLIPPAGE_BPS, execution_prices, run
from .models import MODELS
from .walkforward import information_coefficient, long_flat, walk_forward


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward backtest of a model on the feature table.")
    p.add_argument("--features", default="data/features/BTCUSDT_1h.parquet")
    p.add_argument("--history", default="data/history", help="for execution prices")
    p.add_argument("--target", default="fwd_ret_4h")
    p.add_argument("--model", choices=sorted(MODELS), default="ridge")
    p.add_argument("--min-train-months", type=int, default=24)
    p.add_argument("--test-start", default=None, help="first test month, YYYY-MM")
    p.add_argument("--enter-bps", type=float, default=2 * (FEE_BPS + SLIPPAGE_BPS),
                   help="go long when the predicted return exceeds this (default: round-trip cost)")
    p.add_argument("--exit-bps", type=float, default=0.0, help="go flat when it falls below this")
    p.add_argument("--fee-bps", type=float, default=FEE_BPS)
    p.add_argument("--slippage-bps", type=float, default=SLIPPAGE_BPS)
    p.add_argument("--delay", default="0min", help="extra delay before execution, e.g. 1min")
    p.add_argument("--out", default="data/backtests")
    a = p.parse_args()

    df = pd.read_parquet(a.features)
    preds = walk_forward(df, feature_columns(df), a.target, MODELS[a.model], a.min_train_months, a.test_start)
    test = preds.dropna().index
    df, preds = df.loc[test[0]:], preds.loc[test[0]:]
    weights = long_flat(preds, a.enter_bps / 1e4, a.exit_bps / 1e4)
    prices = execution_prices(df.index, a.history, delay=a.delay)
    res = run(weights, prices, a.fee_bps, a.slippage_bps)

    ic = information_coefficient(preds, df[a.target])
    print(f"{a.model} on {a.target}, out of sample {test[0]:%Y-%m-%d} -> {test[-1]:%Y-%m-%d}, "
          f"costs {a.fee_bps + a.slippage_bps:g} bps per side")
    print(f"rank IC: mean monthly {ic.mean():+.3f}, positive in {(ic > 0).mean():.0%} of {len(ic)} months\n")
    table = pd.DataFrame({"strategy": res.summary(), "buy & hold": res.benchmark()})
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))
    yearly = res.returns.groupby(res.returns.index.year).apply(lambda r: (1 + r).prod() - 1)
    bh = res.asset_returns.groupby(res.asset_returns.index.year).apply(lambda r: (1 + r).prod() - 1)
    print("\nreturn by year:\n" + pd.DataFrame({"strategy": yearly, "buy & hold": bh}).to_string(
        float_format=lambda v: f"{v:+.1%}"))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.model}_{a.target}_enter{a.enter_bps:g}"
    pd.DataFrame({"pred": preds, "weight": res.positions, "return": res.returns, "equity": res.equity,
                  "price": prices}).to_parquet(out / f"{name}.parquet")
    print(f"\nsaved {out / name}.parquet")


if __name__ == "__main__":
    main()
