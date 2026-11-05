"""Spot long/flat backtester with walk-forward model evaluation: python -m backtest"""

from .engine import Result, execution_prices, run
from .walkforward import information_coefficient, long_flat, walk_forward

__all__ = ["Result", "execution_prices", "run", "walk_forward", "long_flat", "information_coefficient"]
