"""Backtester tests: P&L and cost arithmetic, gaps, execution timing, walk-forward embargo, hysteresis."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from backtest import execution_prices, long_flat, run, walk_forward
from backtest.models import Ridge
from history.datasets import DATASETS, write_table


def hours(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="1h", tz="UTC")


def test_pnl_and_costs_are_exact():
    t = hours(4)
    res = run(pd.Series([1, 1, 0, 0.0], t), pd.Series([100, 110, 121, 121.0], t), fee_bps=10, slippage_bps=0)
    assert res.returns.tolist() == pytest.approx([0.1 - 0.001, 0.1, -0.001, 0.0])
    assert res.equity.iloc[-1] == pytest.approx(1.099 * 1.1 * 0.999)
    s = res.summary()
    assert s["trades"] == 1 and s["costs"] == pytest.approx(0.002) and s["time_in_market"] == 0.5
    assert res.benchmark()["total_return"] == pytest.approx(1.099 * 1.1 - 1)


def test_partial_weights_and_clipping():
    t = hours(3)
    res = run(pd.Series([0.5, 2.0, -1.0], t), pd.Series([100, 100, 100.0], t), fee_bps=10, slippage_bps=0)
    assert res.positions.tolist() == [0.5, 1.0, 0.0]          # long/flat only: no leverage, no shorts
    assert res.costs.tolist() == pytest.approx([0.0005, 0.0005, 0.001])


def test_no_price_means_no_trade():
    t = hours(4)
    res = run(pd.Series([0, 1, 1, 0.0], t), pd.Series([100, np.nan, 110, 121.0], t), fee_bps=0, slippage_bps=0)
    assert res.positions.tolist() == [0, 0, 1, 0]             # entry waits for a price
    assert res.returns.tolist() == pytest.approx([0, 0, 0.1, 0])


def test_execution_price_is_vwap_of_minute_after_decision(tmp_path):
    minutes = pd.date_range("2024-01-01", periods=180, freq="1min", tz="UTC")
    px = np.arange(180, dtype=float) + 100
    k = pd.DataFrame({"open_time_us": minutes.asi8 // 1000, "open": px, "high": px, "low": px, "close": px,
                      "volume": 2.0, "quote_volume": 2 * px, "count": 1, "taker_buy_volume": 1.0,
                      "taker_buy_quote_volume": px})
    schema = DATASETS["spot/klines_1m"].schema
    write_table(pa.Table.from_pandas(k[schema.names], schema=schema, preserve_index=False),
                tmp_path / "spot/klines_1m/BTCUSDT/2024-01.parquet")
    t = hours(3)
    assert execution_prices(t, tmp_path).tolist() == [100, 160, 220]
    assert execution_prices(t, tmp_path, delay="2min").tolist() == [102, 162, 222]


class Recorder:
    """Model that records what it was trained on and predicts the training mean."""
    seen: list = []

    def fit(self, X, y):
        Recorder.seen.append(X.index.max())
        self.m = y.mean()
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


def test_walk_forward_embargo_and_no_future_labels():
    t = hours(24 * 120)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=len(t)), "fwd_ret_4h": rng.normal(size=len(t))}, index=t)
    Recorder.seen = []
    preds = walk_forward(df, ["x"], "fwd_ret_4h", Recorder, min_train_months=1)
    test_months = preds.dropna().index.tz_convert(None).to_period("M").unique()
    assert len(test_months) == 3 and preds[preds.index < "2024-02-01"].isna().all()
    for last_train, m in zip(Recorder.seen, test_months):
        assert last_train + pd.Timedelta("4h") <= m.start_time.tz_localize("UTC")
    # Changing the targets in the last month can't move any earlier prediction.
    df2 = df.copy()
    df2.loc[df.index >= "2024-04-01", "fwd_ret_4h"] += 100
    preds2 = walk_forward(df2, ["x"], "fwd_ret_4h", Recorder, min_train_months=1)
    pd.testing.assert_series_equal(preds, preds2)


def test_long_flat_hysteresis():
    pred = pd.Series([0.0, 0.003, 0.001, -0.001, 0.001, np.nan, 0.004, np.nan], index=hours(8))
    assert long_flat(pred, enter=0.002, exit=0.0).tolist() == [0, 1, 1, 0, 0, 0, 1, 1]


def test_ridge_recovers_a_linear_signal():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(5000, 3)), columns=["a", "b", "c"])
    X["mostly_missing"] = np.where(rng.random(5000) < 0.9, np.nan, 1.0)
    y = 0.002 * X.a - 0.001 * X.b + rng.normal(0, 0.001, 5000)
    m = Ridge(alpha=0.0).fit(X, pd.Series(y))
    assert "mostly_missing" not in m.cols
    assert m.weights()["a"] == pytest.approx(0.002, rel=0.05) and m.weights()["b"] == pytest.approx(-0.001, rel=0.1)
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.85
