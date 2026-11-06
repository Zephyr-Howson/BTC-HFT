"""Turning predictions into position sizes (weights in [0, 1]) without look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd


def past_percentile(pred: pd.Series, min_periods: int = 30) -> pd.Series:
    """Each prediction's percentile among all predictions up to and including it.

    Raw predictions drift in level as the model is refit; their rank against history doesn't.
    NaN until min_periods predictions exist.
    """
    return pred.expanding(min_periods=min_periods).rank(pct=True)


def trend_gate(weights: pd.Series, px_vs_ma: pd.Series) -> pd.Series:
    """Zero the weight while price is below its moving average (px_vs_ma = log(price / MA) < 0).

    Where the average isn't known yet, the weight is left alone.
    """
    return weights.where(~(px_vs_ma < 0), 0.0)


def with_band(weights: pd.Series, band: float = 0.1) -> pd.Series:
    """Only rebalance when the target moves at least `band` from the current weight, or to exactly 0/1.

    Small adjustments cost the same fee per unit as big ones but rarely matter, so they are skipped.
    Moves to fully flat or fully long always happen. NaN targets keep the current weight.
    """
    out = np.zeros(len(weights))
    held = 0.0
    for i, w in enumerate(weights.to_numpy(dtype=float)):
        if not np.isnan(w) and (abs(w - held) >= band or (w in (0.0, 1.0) and w != held)):
            held = w
        out[i] = held
    return pd.Series(out, index=weights.index, name="weight")


def smooth(weights: pd.Series, periods: int) -> pd.Series:
    """Trailing mean over the last `periods` rows, so one noisy day moves the target less."""
    return weights.rolling(periods, min_periods=1).mean() if periods > 1 else weights


def steps(weights: pd.Series, n: int = 3) -> pd.Series:
    """Round to the nearest 1/n (n=3: 0, 1/3, 2/3, 1). NaN stays NaN."""
    return (weights * n).round() / n


def at_hour(df: pd.DataFrame | pd.Series, hour: int = 0) -> pd.DataFrame | pd.Series:
    """Rows at hh:00 UTC only: one decision a day."""
    return df[(df.index.hour == hour) & (df.index.minute == 0)]
