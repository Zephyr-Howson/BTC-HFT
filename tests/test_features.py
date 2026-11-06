"""Feature pipeline tests on synthetic history: no look-ahead, target alignment, gaps."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from features import build, feature_columns, target_columns
from history.bars import SCHEMA as BAR_SCHEMA
from history.datasets import DATASETS, write_table

START = pd.Timestamp("2024-01-01", tz="UTC")
DAYS = 10
US = 1_000_000


def _us(times) -> np.ndarray:
    return (pd.DatetimeIndex(times).asi8 // 1000).astype("int64")


def _write(root, name, df, schema):
    write_table(pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False),
                root / name / "BTCUSDT" / "2024-01.parquet")


def make_history(root, seed=0, perturb_after=None, drop=None):
    """Synthetic data for every dataset the pipeline uses.

    perturb_after: change every value that became known after this time.
    drop: (start, end) of spot 1m bars to delete, to make a gap.
    """
    rng = np.random.default_rng(seed)
    minutes = pd.date_range(START, periods=DAYS * 1440, freq="1min")
    px = 40_000 * np.exp(np.cumsum(rng.normal(0, 5e-4, len(minutes))))

    def bump(times, values, known_at_offset=pd.Timedelta(0)):
        if perturb_after is None:
            return values
        later = (pd.DatetimeIndex(times) + known_at_offset) > perturb_after
        return np.where(later, values * 1.7 + 1.0, values)

    def bar_frame(price):
        vol = rng.uniform(1, 10, len(minutes))
        df = pd.DataFrame({"open_time_us": _us(minutes), "open": price, "high": price * 1.001,
                           "low": price * 0.999, "close": price, "volume": vol, "quote_volume": vol * price,
                           "count": 100, "taker_buy_volume": vol * rng.uniform(0.2, 0.8, len(minutes)),
                           "taker_buy_quote_volume": 0.0})
        for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"):
            df[c] = bump(minutes, df[c].to_numpy(), known_at_offset=pd.Timedelta(minutes=1))
        return df

    spot = bar_frame(px)
    if drop:
        t = pd.DatetimeIndex(minutes)
        spot = spot[~((t >= drop[0]) & (t < drop[1]))]
    _write(root, "spot/klines_1m", spot, DATASETS["spot/klines_1m"].schema)
    _write(root, "futures/klines_1m", bar_frame(px * 1.0005), DATASETS["futures/klines_1m"].schema)
    prem = bar_frame(np.full(len(minutes), 1e-4))
    _write(root, "futures/premiumIndex_1m", prem, DATASETS["futures/premiumIndex_1m"].schema)

    for name in ("spot/tradeBars_1m", "futures/tradeBars_1m"):
        b = bar_frame(px)
        b["buy_volume"] = b.taker_buy_volume
        b["trades"] = 50
        b["taker_orders"] = 20
        b["buy_taker_orders"] = 10
        for c, frac in (("max_order_qty", 0.3), ("small_volume", 0.1), ("small_buy_volume", 0.05),
                        ("large_volume", 0.4), ("large_buy_volume", 0.15)):
            b[c] = bump(minutes, b.volume * frac * rng.uniform(0.5, 1.5, len(b)), pd.Timedelta(minutes=1))
        _write(root, name, b, BAR_SCHEMA)

    t8 = pd.date_range(START, periods=DAYS * 3, freq="8h")
    _write(root, "futures/fundingRate", pd.DataFrame({
        "calc_time_us": _us(t8), "funding_interval_hours": 8,
        "last_funding_rate": bump(t8, rng.normal(1e-4, 5e-5, len(t8)))}), DATASETS["futures/fundingRate"].schema)

    t5 = pd.date_range(START, periods=DAYS * 288, freq="5min")
    metrics = {"create_time_us": _us(t5)}
    for c in ("sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
              "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"):
        metrics[c] = bump(t5, rng.uniform(0.5, 2, len(t5)) * (1e5 if "interest" in c else 1))
    _write(root, "futures/metrics", pd.DataFrame(metrics), DATASETS["futures/metrics"].schema)

    t30 = pd.date_range(START + pd.Timedelta(seconds=7), periods=DAYS * 2880, freq="30s")
    rows = []
    for p in (-5.0, -1.0, 1.0, 5.0, 2.0):
        rows.append(pd.DataFrame({"timestamp_us": _us(t30), "percentage": p, "depth": 1.0,
                                  "notional": bump(t30, rng.uniform(1e6, 2e6, len(t30)) * abs(p))}))
    _write(root, "futures/bookDepth", pd.concat(rows), DATASETS["futures/bookDepth"].schema)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("base")
    make_history(root)
    return build(root)


def test_every_feature_filled_once_warmed_up(base):
    warm = base[base.index >= START + pd.Timedelta(days=8)].iloc[:-24]    # 7d windows full, targets known
    trend = [c for c in base.columns if c.endswith(("_30d", "_90d", "_200d"))]   # longer than the test data
    assert len(trend) == 6 and base[trend].isna().all().all()
    empty = [c for c in base.columns if c not in trend and warm[c].isna().any()]
    assert not empty, empty
    assert len(base) == DAYS * 24 and (base.index.minute == 0).all()


def test_no_lookahead(base, tmp_path):
    cutoff = START + pd.Timedelta(days=8, hours=5)
    make_history(tmp_path, perturb_after=cutoff)
    changed = build(tmp_path)
    cols = feature_columns(base)
    before, after = base.index <= cutoff, base.index > cutoff
    pd.testing.assert_frame_equal(base.loc[before, cols], changed.loc[before, cols])
    pd.testing.assert_series_equal(base.loc[before, "close"], changed.loc[before, "close"])
    # The perturbation is visible straight after the cutoff, so the test can detect leaks.
    assert not np.allclose(base.loc[after, "ret_1h"].iloc[0], changed.loc[after, "ret_1h"].iloc[0])
    for c in cols:
        if not c.endswith(("_sin", "_cos", "_30d", "_90d", "_200d")):
            assert not base.loc[after, c].equals(changed.loc[after, c]), c


def test_targets_are_forward_returns(base, tmp_path):
    t = START + pd.Timedelta(days=2, hours=3)
    row = base.loc[t]
    later = base.loc[t + pd.Timedelta(hours=1)]
    assert row.fwd_ret_1h == pytest.approx(np.log(later.close / row.close))
    assert later.ret_1h == pytest.approx(row.fwd_ret_1h)
    assert set(target_columns(base)) == {"fwd_ret_1h", "fwd_ret_4h", "fwd_ret_24h"}


def test_gaps_give_nan_not_stale_values(tmp_path):
    gap = (START + pd.Timedelta(days=3), START + pd.Timedelta(days=3, hours=3))
    make_history(tmp_path, drop=gap)
    df = build(tmp_path)
    inside = df.loc[gap[0] + pd.Timedelta(hours=1)]
    assert np.isnan(inside.close) and np.isnan(inside.ret_1h)
    after = df.loc[gap[1] + pd.Timedelta(hours=2)]
    assert not np.isnan(after.ret_1h) and not np.isnan(after.rv_1h)    # 1h window is clear of the gap
    assert np.isnan(after.rv_4h)                    # 4h window is mostly gap: too few minutes
    assert np.isnan(after.ret_4h)                   # starts inside the gap
    # An hour earlier, the 4h return starts just before the gap: both endpoints exist, so it's real.
    assert not np.isnan(df.loc[gap[1] + pd.Timedelta(hours=1), "ret_4h"])
    assert np.isnan(df.loc[gap[0], "fwd_ret_1h"])


def test_start_trims_rows_but_keeps_warmup(tmp_path):
    make_history(tmp_path)
    df = build(tmp_path, start="2024-01-09")
    assert df.index[0] == pd.Timestamp("2024-01-09", tz="UTC")
    assert not np.isnan(df.ret_7d.iloc[0])


def test_trend_features_have_no_lookahead(tmp_path, monkeypatch):
    import importlib
    fb = importlib.import_module("features.build")     # the module, not the build() function
    monkeypatch.setattr(fb, "TREND_DAYS", (2,))           # short enough for the synthetic data
    cutoff = START + pd.Timedelta(days=6, hours=5)
    make_history(tmp_path / "a")
    make_history(tmp_path / "b", perturb_after=cutoff)
    a, b = build(tmp_path / "a"), build(tmp_path / "b")
    cols = ["ret_2d", "px_vs_ma_2d"]
    assert a[cols].iloc[60:].notna().all().all()
    pd.testing.assert_frame_equal(a.loc[a.index <= cutoff, cols], b.loc[b.index <= cutoff, cols])
    assert not a.loc[a.index > cutoff, "px_vs_ma_2d"].equals(b.loc[b.index > cutoff, "px_vs_ma_2d"])
