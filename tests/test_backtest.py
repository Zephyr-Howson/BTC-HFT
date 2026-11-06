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


def test_signal_report_finds_a_real_signal_and_not_noise():
    from backtest.signals import report
    t = hours(24 * 400)
    rng = np.random.default_rng(3)
    x = rng.normal(size=len(t))
    df = pd.DataFrame({"good": x, "noise": rng.normal(size=len(t)), "close": 1.0,
                       "fwd_ret_1h": 0.001 * x + rng.normal(0, 0.003, len(t))}, index=t)
    r = report(df).set_index("feature")
    assert r.loc["good", "t"] > 10 and r.loc["good", "pos"] == 1 and r.loc["good", "top_bot_bps"] > 15
    assert abs(r.loc["noise", "t"]) < 4


def test_rank_composite_picks_real_signals_with_their_signs():
    from backtest.models import RankComposite
    t = hours(24 * 640)
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(len(t), 4)), index=t, columns=["up", "down", "noise1", "noise2"])
    X.loc[X.index < "2024-06-01", "down"] = np.nan          # a feature that starts later
    y = pd.Series(0.001 * X.up.fillna(0) - 0.001 * X.down.fillna(0) + rng.normal(0, 0.004, len(t)), index=t,
                  name="fwd_ret_1h")
    m = RankComposite(min_t=3).fit(X, y)
    assert set(m.chosen.index) == {"up", "down"}
    assert m.chosen["up"] > 0 > m.chosen["down"] and m.slope > 0
    pred = pd.Series(m.predict(X), index=t)
    assert np.corrcoef(pred, y)[0, 1] > 0.2
    assert not pred.isna().any()                          # rows without "down" still use "up"


def test_sizing_rules():
    from backtest.rules import at_hour, past_percentile, trend_gate, with_band
    t = hours(6)
    pct = past_percentile(pd.Series([1.0, 3, 2, 0, 5, 4], t), min_periods=2)
    assert np.isnan(pct.iloc[0]) and pct.iloc[1] == 1.0 and pct.iloc[2] == pytest.approx(2 / 3)
    assert pct.iloc[3] == 0.25                                   # only earlier values count
    gated = trend_gate(pd.Series([0.8, 0.8, 0.8], hours(3)), pd.Series([0.1, -0.1, np.nan], hours(3)))
    assert gated.tolist() == [0.8, 0.0, 0.8]
    w = with_band(pd.Series([0.5, 0.55, 0.65, 0.0, 0.05, np.nan, 1.0], hours(7)), band=0.1)
    assert w.tolist() == [0.5, 0.5, 0.65, 0.0, 0.0, 0.0, 1.0]
    assert len(at_hour(pd.Series(1, hours(72)))) == 3


def test_predictive_ic_is_unbiased_on_a_random_walk():
    """A within-month rank IC reports strong 'mean reversion' here (about -0.2 for px_vs_ma_30d)."""
    from backtest.stats import predictive_ic
    t = hours(24 * 365 * 3)
    lp = pd.Series(np.cumsum(np.random.default_rng(5).normal(0, 0.007, len(t))), t)
    y, h = lp.shift(-24) - lp, pd.Timedelta("24h")
    for x in (lp - lp.shift(168), lp - lp.rolling(720).mean()):
        assert abs(predictive_ic(x, y, h).mean()) < 0.03
