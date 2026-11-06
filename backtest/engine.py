"""Spot backtest: long/flat positions at hourly (or any) decision times, with fees and execution delay.

A strategy supplies a target weight in [0, 1] (share of equity in BTC) for each decision time t.
It is filled at the execution price for t, never at the close used to compute features:
by default the VWAP of the first minute after t, which a market order sent at t would get.
Costs are charged on every change in weight: fee + slippage, in bps of the amount traded.

If there is no execution price at t (a data gap), nothing trades and the previous position is held.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from history.datasets import load

FEE_BPS = 10.0       # Binance spot taker, base tier (7.5 paying fees in BNB)
SLIPPAGE_BPS = 1.0   # BTCUSDT spread is ~0.01 USD; this covers impact for small orders

# Interest earned on the part of the account not in BTC: approximate yearly average of the RBA cash rate,
# which Australian high-interest savings accounts roughly track. Years after the table use its last value.
CASH_RATE_BY_YEAR = {2017: 0.015, 2018: 0.015, 2019: 0.010, 2020: 0.003, 2021: 0.001, 2022: 0.013,
                     2023: 0.039, 2024: 0.0435, 2025: 0.039, 2026: 0.036}


def cash_rates(index: pd.DatetimeIndex) -> pd.Series:
    """Annual cash rate in force at each time, from CASH_RATE_BY_YEAR."""
    last = CASH_RATE_BY_YEAR[max(CASH_RATE_BY_YEAR)]
    return pd.Series([CASH_RATE_BY_YEAR.get(y, last) for y in index.year], index=index, dtype=float)


def execution_prices(times: pd.DatetimeIndex, root: str | Path = "data/history", symbol: str = "BTCUSDT",
                     delay: str = "0min") -> pd.Series:
    """VWAP of the 1m spot bar starting at t + delay, for each decision time t."""
    start = (times[0] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (times[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    k = load("spot/klines_1m", root, symbol, start[:7], end[:7], ["open_time_us", "volume", "quote_volume"])
    vwap = (k.quote_volume / k.volume.where(k.volume > 0)).to_numpy()
    idx = pd.DatetimeIndex(pd.to_datetime(k.open_time_us.astype("int64"), unit="us", utc=True))
    s = pd.Series(vwap, index=idx)
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(times + pd.Timedelta(delay)).set_axis(times)


@dataclass
class Result:
    returns: pd.Series            # net return of each period (from one decision time to the next)
    positions: pd.Series          # weight actually held through each period
    costs: pd.Series              # cost charged at the start of each period
    asset_returns: pd.Series      # BTC return over each period
    cost_bps: float = FEE_BPS + SLIPPAGE_BPS
    periods_per_year: float = field(repr=False, default=8760.0)

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    def summary(self) -> dict[str, float]:
        return _summary(self.returns, self.positions, self.costs, self.periods_per_year)

    def benchmark(self) -> dict[str, float]:
        """Buy and hold over the same periods, paying one entry fee."""
        pos = pd.Series(1.0, index=self.positions.index)
        cost = pd.Series(0.0, index=pos.index)
        cost.iloc[0] = self.cost_bps / 1e4
        return _summary(self.asset_returns - cost, pos, cost, self.periods_per_year)


def run(weights: pd.Series, prices: pd.Series, fee_bps: float = FEE_BPS,
        slippage_bps: float = SLIPPAGE_BPS, cash_rate: float | pd.Series = 0.0) -> Result:
    """Backtest target weights (indexed by decision time) filled at prices (same index).

    cash_rate: annual interest on the share of the account not in BTC, either one number or a series on
    the same index (e.g. cash_rates(index)). It accrues for the actual time until the next decision.
    """
    if not weights.index.equals(prices.index):
        raise ValueError("weights and prices must share an index")
    target = weights.astype(float).clip(0, 1)
    tradable = prices.notna() & target.notna()
    held = target.where(tradable).ffill().fillna(0.0)
    cost = held.diff().abs().fillna(held.iloc[0]) * (fee_bps + slippage_bps) / 1e4
    px = prices.ffill()
    asset = (px.shift(-1) / px - 1).fillna(0.0)
    gaps = pd.Series(weights.index, index=weights.index).diff().shift(-1).fillna(pd.Timedelta(0))
    interest = (1 - held) * cash_rate * (gaps / pd.Timedelta(days=365))
    net = held * asset + interest - cost
    spacing = pd.Series(weights.index).diff().median()
    per_year = pd.Timedelta(days=365) / spacing if pd.notna(spacing) else 8760.0
    return Result(net, held, cost, asset, fee_bps + slippage_bps, per_year)


def random_timing(weights: pd.Series, prices: pd.Series, n: int = 200, min_shift: int = 30, seed: int = 0,
                  fee_bps: float = FEE_BPS, slippage_bps: float = SLIPPAGE_BPS,
                  cash_rate: float | pd.Series = 0.0) -> np.ndarray:
    """Sharpe ratios of the same weight path, circularly shifted by random offsets against the prices.

    Each copy has the same exposure, holding periods and turnover as the strategy, but its timing
    has nothing to do with the market. If the strategy doesn't beat most copies, its result comes
    from how much it holds, not when.
    """
    rng = np.random.default_rng(seed)
    w = weights.to_numpy(dtype=float)
    shifts = rng.integers(min_shift, len(w) - min_shift, size=n)
    return np.array([run(pd.Series(np.roll(w, k), index=weights.index), prices, fee_bps, slippage_bps, cash_rate)
                     .summary()["sharpe"] for k in shifts])


def _summary(ret: pd.Series, pos: pd.Series, cost: pd.Series, per_year: float) -> dict[str, float]:
    equity = (1 + ret).cumprod()
    years = len(ret) / per_year
    sd = ret.std()
    entries = int(((pos > 0) & (pos.shift(fill_value=0) == 0)).sum())
    in_market = pos > 0
    runs = (in_market != in_market.shift()).cumsum()[in_market]
    return {
        "total_return": equity.iloc[-1] - 1,
        "cagr": equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "sharpe": ret.mean() / sd * np.sqrt(per_year) if sd > 0 else np.nan,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "time_in_market": in_market.mean(),
        "exposure": pos.mean(),
        "trades": entries,
        "avg_hold_periods": runs.value_counts().mean() if entries else 0.0,
        "turnover_per_year": pos.diff().abs().fillna(pos.iloc[0]).sum() / years if years > 0 else np.nan,
        "costs": cost.sum(),
    }
