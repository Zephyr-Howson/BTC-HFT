"""Walk-forward evaluation: every prediction comes from a model trained only on earlier data.

For each test month, the model is refit on all earlier rows whose target was already known by the
start of that month. A row at t with target fwd_ret_4h is only known at t + 4h, so the last few
hours before each test month are left out of training (the embargo). Without this, training labels
overlap the test period and results look better than they are.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def _months(index: pd.DatetimeIndex) -> pd.PeriodIndex:
    return (index.tz_convert(None) if index.tz else index).to_period("M")


def horizon(target: str) -> pd.Timedelta:
    """fwd_ret_4h -> 4 hours."""
    return pd.Timedelta(target.removeprefix("fwd_ret_"))


def walk_forward(df: pd.DataFrame, features: list[str], target: str, make_model, min_train_months: int = 12,
                 test_start: str | None = None) -> pd.Series:
    """Out-of-sample predictions of `target`, refit monthly on an expanding window.

    make_model() returns a fresh model with fit/predict. Rows before the first test month (at least
    min_train_months after the data starts, or test_start if later) get no prediction.
    """
    h = horizon(target)
    months = _months(df.index).unique().sort_values()
    first = months[0] + min_train_months
    if test_start:
        first = max(first, pd.Period(test_start, "M"))
    preds = pd.Series(np.nan, index=df.index, name=f"pred_{target}")
    labelled = df[target].notna()
    for m in months[months >= first]:
        start = m.start_time.tz_localize("UTC")
        end = (m + 1).start_time.tz_localize("UTC")
        train = labelled & (df.index + h <= start)
        test = (df.index >= start) & (df.index < end)
        if train.sum() < 100 or not test.any():
            continue
        model = make_model().fit(df.loc[train, features], df.loc[train, target])
        preds[test] = model.predict(df.loc[test, features])
    return preds


def long_flat(pred: pd.Series, enter: float, exit: float = 0.0) -> pd.Series:
    """Weight 1 once pred rises above enter; back to 0 once it falls below exit.

    The gap between the two (hysteresis) stops the position flipping on noise and paying fees each
    time. With enter at about the round-trip cost, a trade only happens when the predicted move
    should cover it. Missing predictions keep the current position.
    """
    w = np.zeros(len(pred))
    on = False
    for i, p in enumerate(pred.to_numpy()):
        if not np.isnan(p):
            on = p > enter or (on and p >= exit)
        w[i] = float(on)
    return pd.Series(w, index=pred.index, name="weight")


def information_coefficient(pred: pd.Series, actual: pd.Series) -> pd.Series:
    """Monthly rank correlation between predictions and realised targets."""
    both = pd.concat({"p": pred, "a": actual}, axis=1).dropna()
    return both.groupby(_months(both.index)).apply(lambda g: g.p.rank().corr(g.a.rank()))  # Spearman without scipy
