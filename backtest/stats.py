"""Predictive correlation between a signal and forward returns, without small-sample bias.

The naive approach (rank-correlate x and y within each month) is biased for persistent features:
the within-month mean of x contains future prices, so price-level features (7d return, distance
from a moving average) show strong "mean reversion" even on a pure random walk (IC around -0.1 to
-0.3 at a 24h horizon). This is the same effect as the Dickey-Fuller / Stambaugh bias.

Here every quantity is known at t, and nothing is demeaned within the evaluation window:
- x is scored as its percentile within its own trailing window (past only), centred at 0
- y is scaled by the trailing volatility of already-realised returns, and clipped at +/-5
- the monthly statistic is mean(x*y) * sqrt(12): roughly a correlation, as x has variance 1/12
  and y roughly unit variance, but with a fixed denominator (a sample one would contain future y)

If y is unpredictable from information at t (a martingale), E[x*y] = 0 exactly, so the statistic
is centred on zero under the null whatever x is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = pd.Timedelta(days=90)


def months(index: pd.DatetimeIndex) -> pd.PeriodIndex:
    return (index.tz_convert(None) if index.tz else index).to_period("M")


def _rows(index: pd.DatetimeIndex, span: pd.Timedelta) -> int:
    spacing = pd.Series(index).diff().median()
    return max(1, int(np.ceil(span / spacing)))


def trailing_percentile(x: pd.Series, window: pd.Timedelta = WINDOW) -> pd.Series:
    """Percentile of each value within the trailing window (inclusive), minus 0.5."""
    n = _rows(x.index, window)
    return x.rolling(n, min_periods=n // 3).rank(pct=True) - 0.5


def scaled_returns(y: pd.Series, horizon: pd.Timedelta, window: pd.Timedelta = WINDOW) -> pd.Series:
    """Forward returns divided by the trailing sd of forward returns already realised by t."""
    n, lag = _rows(y.index, window), _rows(y.index, horizon)
    sd = y.shift(lag).rolling(n, min_periods=n // 4).std()
    return (y / sd).clip(-5, 5)


def predictive_ic(x: pd.Series, y: pd.Series, horizon: pd.Timedelta, min_obs: int = 20,
                  window: pd.Timedelta = WINDOW) -> pd.Series:
    """Monthly uncentred correlation between trailing-percentile x and vol-scaled forward returns y."""
    d = pd.concat({"x": trailing_percentile(x, window), "y": scaled_returns(y, horizon, window)}, axis=1).dropna()
    key = months(d.index)
    # Fixed normalisation: x is a centred percentile (variance 1/12) and y has ~unit variance. Dividing
    # by the sample variances instead would put future returns in the denominator, which biases it.
    ic = (d.x * d.y).groupby(key).mean() * np.sqrt(12)
    n = d.groupby(key).size()
    return ic[n >= min_obs]


def t_stat(ic: pd.Series, lags: int = 6) -> float:
    """t-stat of the mean monthly IC, Newey-West: slow features give IC runs across months,
    so months aren't independent and a plain t-stat overstates the evidence."""
    v = ic.dropna().to_numpy()
    n = len(v)
    if n <= lags + 2 or v.std() == 0:
        return np.nan
    e = v - v.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return v.mean() / np.sqrt(max(var, 1e-18) / n)
