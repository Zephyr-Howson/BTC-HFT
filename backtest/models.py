"""Baseline models. Anything with fit(X, y) / predict(X) works with walk_forward."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


MODELS = {"ridge": Ridge}
