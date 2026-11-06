"""Baseline models. Anything with fit(X, y) / predict(X) works with walk_forward."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .stats import predictive_ic, t_stat


class Ridge:
    """Linear regression with an L2 penalty, on standardised features. A deliberately simple baseline.

    alpha is relative to each feature's variance (Z'Z ~ n * correlation), so it means the same at any n.

    - Features that are mostly missing in the training window (e.g. book depth before 2023) are dropped.
    - Remaining gaps are filled with the training mean (0 after standardising).
    - Values are clipped at +/-5 sd, so one bad print can't dominate a prediction.
    """

    def __init__(self, alpha: float = 0.1, max_missing: float = 0.5) -> None:
        self.alpha, self.max_missing = alpha, max_missing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Ridge":
        self.cols = [c for c in X.columns if X[c].isna().mean() <= self.max_missing and X[c].std() > 0]
        self.mu, self.sd = X[self.cols].mean(), X[self.cols].std()
        Z = self._z(X)
        n, k = Z.shape
        self.intercept = float(y.mean())
        self.coef = np.linalg.solve(Z.T @ Z + self.alpha * n * np.eye(k), Z.T @ (y.to_numpy() - self.intercept))
        return self

    def _z(self, X: pd.DataFrame) -> np.ndarray:
        return ((X[self.cols] - self.mu) / self.sd).clip(-5, 5).fillna(0.0).to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + self._z(X) @ self.coef

    def weights(self) -> pd.Series:
        """Coefficients per feature, in target units per 1 sd move."""
        return pd.Series(self.coef, index=self.cols).sort_values(key=abs, ascending=False)


def predictive_t(X: pd.DataFrame, y: pd.Series, horizon: pd.Timedelta, min_months: int = 12) -> pd.Series:
    """Newey-West t-stat of each feature's monthly predictive IC with y (see backtest/stats.py)."""
    out = {}
    for c in X.columns:
        if X[c].notna().sum() < 500:
            continue
        ic = predictive_ic(X[c], y, horizon)
        if len(ic) >= min_months:
            out[c] = t_stat(ic)
    return pd.Series(out, dtype=float).dropna()


class RankComposite:
    """Average percentile rank of the most reliable features, each signed by its past IC.

    Fitting uses only the training rows: features are ranked by the t-stat of their monthly predictive
    IC with the target (backtest/stats.py), the top ones (|t| >= min_t) are kept with the sign they had, and each new value
    is scored by its percentile within that feature's training distribution. Averaging ranks means
    no single feature dominates and fat-tailed returns don't drive the fit. Finally a one-variable
    regression maps the score to an expected return, so entry thresholds stay in return units.
    """

    def __init__(self, top: int = 10, min_t: float = 3.0, min_months: int = 12, horizon: str | None = None) -> None:
        self.top, self.min_t, self.min_months, self.horizon = top, min_t, min_months, horizon

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RankComposite":
        h = pd.Timedelta(self.horizon or str(y.name).removeprefix("fwd_ret_"))   # e.g. y named fwd_ret_24h
        t = predictive_t(X, y, h, self.min_months)
        t = t[t.abs() >= self.min_t]
        self.chosen = t.reindex(t.abs().sort_values(ascending=False).index).head(self.top)
        self.ref = {c: np.sort(X[c].dropna().to_numpy()) for c in self.chosen.index}
        score = self.score(X)
        ok = score.notna() & y.notna()
        if ok.sum() > 100 and score[ok].var() > 0:
            # Clamped at 0: the score is signed so that higher means better, and an OLS slope on
            # fat-tailed returns can come out negative and silently invert every prediction.
            self.slope = max(0.0, float(np.cov(score[ok], y[ok])[0, 1] / score[ok].var()))
            self.intercept = float(y[ok].mean() - self.slope * score[ok].mean())
        else:
            self.slope, self.intercept = 0.0, float(y.mean())
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Mean signed percentile in [-0.5, 0.5], over the chosen features present in each row."""
        if self.chosen.empty:
            return pd.Series(np.nan, index=X.index)
        parts = []
        for c, t in self.chosen.items():
            x = X[c].to_numpy(dtype=float)
            p = np.searchsorted(self.ref[c], x, side="right") / len(self.ref[c]) - 0.5
            parts.append(np.where(np.isnan(x), np.nan, np.sign(t) * p))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)    # rows with none of the features: NaN
            return pd.Series(np.nanmean(np.vstack(parts), axis=0), index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.intercept + self.slope * self.score(X)).to_numpy()   # NaN score: no view, hold position


MODELS = {"ridge": Ridge, "composite": RankComposite}
