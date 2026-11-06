"""Baseline models. Anything with fit(X, y) / predict(X) works with walk_forward."""

from __future__ import annotations

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

    def __init__(self, top: int = 10, min_t: float = 3.0, min_months: int = 12, horizon: str | None = None,
                 weighting: str = "equal") -> None:
        self.top, self.min_t, self.min_months, self.horizon = top, min_t, min_months, horizon
        self.weighting = weighting    # "equal": each feature counts once; "t": in proportion to its |t|

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RankComposite":
        h = pd.Timedelta(self.horizon or str(y.name).removeprefix("fwd_ret_"))   # e.g. y named fwd_ret_24h
        t = predictive_t(X, y, h, self.min_months)
        t = t[t.abs() >= self.min_t]
        self.chosen = t.reindex(t.abs().sort_values(ascending=False).index).head(self.top)
        self.ref = {c: np.sort(X[c].dropna().to_numpy()) for c in self.chosen.index}
        return self._fit_slope(self.score(X), y)

    def _fit_slope(self, score: pd.Series, y: pd.Series) -> "RankComposite":
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
        w = self.chosen if self.weighting == "t" else np.sign(self.chosen)
        return _weighted_ranks(X, self.ref, w)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.intercept + self.slope * self.score(X)).to_numpy()   # NaN score: no view, hold position


def _percentile(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Percentile of x within the sorted training values ref, minus 0.5. NaN stays NaN."""
    return np.where(np.isnan(x), np.nan, np.searchsorted(ref, x, side="right") / max(len(ref), 1) - 0.5)


def _weighted_ranks(X: pd.DataFrame, ref: dict[str, np.ndarray], weights: pd.Series) -> pd.Series:
    """sum(w * percentile) / sum(|w|) over the features present in each row; NaN if none are."""
    if weights.empty:
        return pd.Series(np.nan, index=X.index)
    num = np.zeros(len(X))
    den = np.zeros(len(X))
    for c, w in weights.items():
        p = _percentile(ref[c], X[c].to_numpy(dtype=float))
        ok = ~np.isnan(p)
        num[ok] += w * p[ok]
        den[ok] += abs(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.Series(np.where(den > 0, num / den, np.nan), index=X.index)


# Fixed by what each feature measures, not by how it performed. Signs are still learned in training.
GROUPS = {
    "trend": ["ret_7d", "ret_30d", "px_vs_ma_30d", "ret_90d", "px_vs_ma_90d", "ret_200d", "px_vs_ma_200d"],
    "short_price": ["ret_1h", "ret_4h", "ret_24h", "px_vs_vwap_24h", "pos_in_range_24h", "pos_in_range_7d"],
    "volatility": ["rv_1h", "rv_4h", "rv_24h", "rv_7d", "rv_ratio_1h_24h", "range_1h", "volume_z_1h_7d"],
    "spot_flow": ["taker_imb_1h", "taker_imb_4h", "taker_imb_24h", "large_imb_1h", "small_imb_1h", "large_imb_4h",
                  "small_imb_4h", "large_imb_24h", "small_imb_24h", "large_share_24h", "avg_order_usd_1h",
                  "max_order_usd_1h"],
    "futures_flow": ["fut_taker_imb_1h", "fut_large_imb_1h", "fut_taker_imb_4h", "fut_large_imb_4h",
                     "fut_taker_imb_24h", "fut_large_imb_24h", "fut_spot_volume_24h"],
    "basis_funding": ["basis", "premium_1h", "premium_chg_24h", "funding", "funding_vs_7d"],
    "positioning": ["oi_chg_1h", "oi_chg_4h", "oi_chg_24h", "ls_accounts", "ls_top_traders", "taker_ls_5m"],
    "depth": ["depth_imb_1pct", "depth_imb_1pct_1h", "depth_1pct", "depth_imb_5pct", "depth_imb_5pct_1h",
              "depth_5pct"],
}


class GroupComposite(RankComposite):
    """Two-level composite: features are averaged within fixed economic groups, then groups are combined.

    In training, each member with |t| >= member_t joins its group's score with its sign; each group
    whose score has |t| >= min_t is then kept with its sign. The final score is the mean over kept
    groups. A group that's strong as a whole counts once, however many correlated members it has,
    and swapping one member for a near-identical one doesn't change the model.
    """

    def __init__(self, groups: dict[str, list[str]] | None = None, member_t: float = 1.0, min_t: float = 2.0,
                 min_months: int = 12, horizon: str | None = None) -> None:
        super().__init__(min_t=min_t, min_months=min_months, horizon=horizon)
        self.groups, self.member_t = groups or GROUPS, member_t

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GroupComposite":
        h = pd.Timedelta(self.horizon or str(y.name).removeprefix("fwd_ret_"))
        t = predictive_t(X[[c for cols in self.groups.values() for c in cols if c in X]], y, h, self.min_months)
        self.members = {}
        for g, cols in self.groups.items():
            m = t.reindex(cols).dropna()
            m = np.sign(m[m.abs() >= self.member_t])
            if not m.empty:
                self.members[g] = m
        self.ref = {c: np.sort(X[c].dropna().to_numpy()) for m in self.members.values() for c in m.index}
        gt = predictive_t(self.group_scores(X), y, h, self.min_months)
        self.chosen = gt[gt.abs() >= self.min_t]          # kept groups, with their t-stat
        return self._fit_slope(self.score(X), y)

    def group_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({g: _weighted_ranks(X, self.ref, m) for g, m in self.members.items()}, index=X.index)

    def score(self, X: pd.DataFrame) -> pd.Series:
        if self.chosen.empty:
            return pd.Series(np.nan, index=X.index)
        gs = self.group_scores(X)[self.chosen.index] * np.sign(self.chosen)
        return gs.mean(axis=1)    # skips missing groups; NaN if all are missing


MODELS = {"ridge": Ridge, "composite": RankComposite, "groups": GroupComposite}
