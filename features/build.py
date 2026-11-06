"""Feature table for an hourly-horizon spot strategy, built from the 1m history.

One row per sample time t (hourly by default):
- features: computed only from data that was public before t
- targets:  fwd_ret_<h> = log(spot close at t+h / spot close at t)
- close:    the spot close at t, for the backtester

Timing rules (no look-ahead):
- A 1m bar with open time T is complete at T+1m, so it is indexed at T+1m.
- Point samples (funding, open interest, book depth) are used as of the latest
  sample at or before t. Open interest also gets METRICS_LAG, because Binance
  publishes it with a delay.

Gaps stay as NaN instead of being papered over:
- Rolling windows need COVERAGE of their minutes present.
- Prices are carried forward at most MAX_FILL minutes.
- Point samples go stale after a dataset-specific tolerance.

Datasets that don't exist yet for a period (futures before 2020, book depth
before 2023, trade bars before --trades-start) just give NaN columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from history.datasets import in_range, load

MIN = pd.Timedelta(minutes=1)
WINDOWS = {"1h": 60, "4h": 240, "24h": 1440, "7d": 10080}
TREND_DAYS = (30, 90, 200)    # long-term trend: returns and distance from moving averages
HORIZONS = {"1h": 60, "4h": 240, "24h": 1440}
COVERAGE = 0.8
MAX_FILL = 5
METRICS_LAG = pd.Timedelta(minutes=5)
DEPTH_LEVELS = (1.0, 5.0)     # % from mid; bookDepth notional is cumulative out to each level

KLINE_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]
FLOW_COLS = ["volume", "quote_volume", "taker_orders", "max_order_qty",
             "small_volume", "small_buy_volume", "large_volume", "large_buy_volume"]


class _Sources:
    """Loads datasets for one symbol and date range and aligns them to a 1-minute grid."""

    def __init__(self, root: str | Path, symbol: str, start: str, end: str) -> None:
        self.root, self.symbol, self.start, self.end = Path(root), symbol, start, end
        self.grid: pd.DatetimeIndex | None = None

    def _load(self, name: str, columns: list[str]) -> pd.DataFrame:
        return load(name, self.root, self.symbol, self.start, self.end, columns)

    def bars(self, name: str, columns: list[str]) -> pd.DataFrame:
        df = self._load(name, ["open_time_us", *columns])
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("open_time_us").astype("int64"), unit="us", utc=True)) + MIN
        df = df[~df.index.duplicated(keep="last")].sort_index().astype(float)
        if self.grid is None:
            self.grid = pd.date_range(df.index[0], df.index[-1], freq=MIN)
        return df.reindex(self.grid)

    def points(self, df: pd.DataFrame, time_col: str, tolerance: pd.Timedelta,
               lag: pd.Timedelta = pd.Timedelta(0)) -> pd.DataFrame:
        """Point samples -> grid: each minute gets the latest sample at or before it, if fresh enough."""
        times = pd.to_datetime(df.pop(time_col).astype("int64"), unit="us", utc=True) + lag
        df = df.astype(float).set_axis(pd.DatetimeIndex(times).ceil(MIN)).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if df.empty:
            return pd.DataFrame(np.nan, index=self.grid, columns=df.columns)
        return df.reindex(self.grid, method="ffill", tolerance=tolerance)

    def depth(self) -> pd.DataFrame:
        """Futures book depth, pivoted to bid/ask notional at each of DEPTH_LEVELS."""
        pcts = [p * s for p in DEPTH_LEVELS for s in (-1, 1)]
        files = sorted(f for f in (self.root / "futures/bookDepth" / self.symbol).glob("*.parquet")
                       if in_range(f.stem, self.start, self.end))
        parts = []
        for f in files:    # file by file: the raw table has ~12 rows per 30s sample
            t = pq.read_table(f, columns=["timestamp_us", "percentage", "notional"],
                              filters=[("percentage", "in", pcts)]).to_pandas()
            if not t.empty:
                    parts.append(t.pivot_table(index="timestamp_us", columns="percentage", values="notional",
                                           aggfunc="last"))
        cols = {p: (f"bid_{p:g}" if p < 0 else f"ask_{p:g}").replace("-", "") for p in pcts}
        df = pd.concat(parts) if parts else pd.DataFrame(columns=pcts)
        df = df.reindex(columns=pcts).rename(columns=cols).rename_axis(None, axis=1).reset_index(names="t")
        return self.points(df, "t", tolerance=pd.Timedelta(minutes=5))


def _sum(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=int(n * COVERAGE)).sum()


def _imbalance(buy: pd.Series, total: pd.Series, n: int, denom: pd.Series | None = None) -> pd.Series:
    """(buy - sell) / denom over n minutes; denom defaults to the total."""
    d = _sum(total if denom is None else denom, n)
    return (2 * _sum(buy, n) - _sum(total, n)) / d.where(d > 0)


def _log(x: pd.Series) -> pd.Series:
    return np.log(x.where(x > 0))


class _Sampled(dict):
    """Keeps each feature only at the sample times: the 1m series are ~5M rows each, so holding
    ~60 of them for 9 years would need several GB. Features combined later (ratios, differences)
    are pointwise, so they come out the same from the sampled values."""

    def __init__(self, sample: pd.DatetimeIndex) -> None:
        super().__init__()
        self.sample = sample

    def __setitem__(self, key: str, value: pd.Series) -> None:
        super().__setitem__(key, value.reindex(self.sample))


def build(root: str | Path = "data/history", symbol: str = "BTCUSDT", start: str = "", end: str = "9999",
          freq: str = "1h") -> pd.DataFrame:
    """Build the feature table. start/end are YYYY-MM or YYYY-MM-DD labels, as for history.load."""
    # Load earlier than asked, so the 200-day trend windows are warmed up at `start`.
    load_start = (pd.Period(start[:7], "M") - 8).strftime("%Y-%m") if start else ""
    src = _Sources(root, symbol, load_start, end)
    spot = src.bars("spot/klines_1m", KLINE_COLS)
    grid = src.grid
    sample = grid[grid == grid.floor(freq)]
    if start:
        sample = sample[sample >= pd.Timestamp(start, tz="UTC")]
    f = _Sampled(sample)

    # Spot price and volatility (2017 onwards).
    close = spot.close.ffill(limit=MAX_FILL)
    logc = _log(close)
    r1 = logc.diff()
    for w, n in WINDOWS.items():
        f[f"ret_{w}"] = logc - logc.shift(n)
        f[f"rv_{w}"] = np.sqrt(_sum(r1 ** 2, n) * 1440 / n)   # daily units, comparable across windows
    f["rv_ratio_1h_24h"] = f["rv_1h"] / f["rv_24h"]
    f["range_1h"] = _log(spot.high.rolling(60, min_periods=48).max() / spot.low.rolling(60, min_periods=48).min())
    vwap_24h = _sum(spot.quote_volume, 1440) / _sum(spot.volume, 1440)
    f["px_vs_vwap_24h"] = logc - _log(vwap_24h)
    for w in ("24h", "7d"):
        n = WINDOWS[w]
        hi = spot.high.rolling(n, min_periods=int(n * COVERAGE)).max()
        lo = spot.low.rolling(n, min_periods=int(n * COVERAGE)).min()
        f[f"pos_in_range_{w}"] = (close - lo) / (hi - lo).where(hi > lo)
    f["volume_z_1h_7d"] = _log(_sum(spot.quote_volume, 60) / (_sum(spot.quote_volume, WINDOWS["7d"]) / 168))

    # Long-term trend.
    for days in TREND_DAYS:
        n = days * 1440
        f[f"ret_{days}d"] = logc - logc.shift(n)
        f[f"px_vs_ma_{days}d"] = logc - _log(close.rolling(n, min_periods=int(n * COVERAGE)).mean())

    # Spot taker flow: klines cover 2017 onwards; trade bars add order-size detail.
    for w in ("1h", "4h", "24h"):
        f[f"taker_imb_{w}"] = _imbalance(spot.taker_buy_volume, spot.volume, WINDOWS[w])
    flow = src.bars("spot/tradeBars_1m", FLOW_COLS)
    for w in ("1h", "4h", "24h"):
        n = WINDOWS[w]
        f[f"large_imb_{w}"] = _imbalance(flow.large_buy_volume, flow.large_volume, n, denom=flow.volume)
        f[f"small_imb_{w}"] = _imbalance(flow.small_buy_volume, flow.small_volume, n, denom=flow.volume)
    f["large_share_24h"] = _sum(flow.large_volume, 1440) / _sum(flow.volume, 1440)
    f["avg_order_usd_1h"] = _log(_sum(flow.quote_volume, 60) / _sum(flow.taker_orders, 60))
    f["max_order_usd_1h"] = _log((flow.max_order_qty * close).rolling(60, min_periods=48).max())

    # Futures: perp flow, basis and premium (2020 onwards).
    fut = src.bars("futures/klines_1m", ["close", "volume", "quote_volume", "taker_buy_volume"])
    fflow = src.bars("futures/tradeBars_1m", ["volume", "large_volume", "large_buy_volume"])
    for w in ("1h", "4h", "24h"):
        n = WINDOWS[w]
        f[f"fut_taker_imb_{w}"] = _imbalance(fut.taker_buy_volume, fut.volume, n)
        f[f"fut_large_imb_{w}"] = _imbalance(fflow.large_buy_volume, fflow.large_volume, n, denom=fflow.volume)
    f["fut_spot_volume_24h"] = _log(_sum(fut.quote_volume, 1440) / _sum(spot.quote_volume, 1440))
    f["basis"] = _log(fut.close.ffill(limit=MAX_FILL)) - logc
    prem = src.bars("futures/premiumIndex_1m", ["close"]).close
    f["premium_1h"] = prem.rolling(60, min_periods=48).mean()
    f["premium_chg_24h"] = f["premium_1h"] - prem.rolling(1440, min_periods=int(1440 * COVERAGE)).mean()

    # Funding (per 8h, as the interval changed from 8h to 4h at times).
    fund = src._load("futures/fundingRate", ["calc_time_us", "funding_interval_hours", "last_funding_rate"])
    hours = fund.pop("funding_interval_hours").astype(float).replace(0, 8).fillna(8)
    fund["rate"] = fund.pop("last_funding_rate").astype(float) * 8 / hours
    fund = fund.sort_values("calc_time_us")
    t = pd.DatetimeIndex(pd.to_datetime(fund.calc_time_us.astype("int64"), unit="us", utc=True))
    fund["mean_7d"] = fund.rate.set_axis(t).rolling("7D").mean().to_numpy()
    fund = src.points(fund, "calc_time_us", tolerance=pd.Timedelta(hours=12))
    f["funding"] = fund.rate
    f["funding_vs_7d"] = fund.rate - fund.mean_7d

    # Open interest and long/short ratios (5m samples, 2020-09 onwards).
    m = src._load("futures/metrics", ["create_time_us", "sum_open_interest", "count_long_short_ratio",
                                      "sum_toptrader_long_short_ratio", "sum_taker_long_short_vol_ratio"])
    m = src.points(m, "create_time_us", tolerance=pd.Timedelta(minutes=15), lag=METRICS_LAG)
    log_oi = _log(m.sum_open_interest)
    for w in ("1h", "4h", "24h"):
        f[f"oi_chg_{w}"] = log_oi - log_oi.shift(WINDOWS[w])
    f["ls_accounts"] = _log(m.count_long_short_ratio)
    f["ls_top_traders"] = _log(m.sum_toptrader_long_short_ratio)
    f["taker_ls_5m"] = _log(m.sum_taker_long_short_vol_ratio)

    # Futures book depth near mid (2023 onwards).
    d = src.depth()
    for p in DEPTH_LEVELS:
        bid, ask = d[f"bid_{p:g}"], d[f"ask_{p:g}"]
        imb = (bid - ask) / (bid + ask)
        f[f"depth_imb_{p:g}pct"] = imb
        f[f"depth_imb_{p:g}pct_1h"] = imb.rolling(60, min_periods=48).mean()
        f[f"depth_{p:g}pct"] = _log(bid + ask)

    # Targets.
    for h, n in HORIZONS.items():
        f[f"fwd_ret_{h}"] = logc.shift(-n) - logc

    out = pd.DataFrame({k: v.to_numpy() for k, v in f.items()}, index=sample)
    out.insert(0, "close", close.reindex(sample).to_numpy())
    hour = sample.hour + sample.minute / 60
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * sample.dayofweek / 7), np.cos(2 * np.pi * sample.dayofweek / 7)
    out.index.name = "time"
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "close" and not c.startswith("fwd_")]


def target_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("fwd_")]
