# btc-research

Short-horizon trading research on BTC/USDT, built from two data sources:

1. **History** (`python -m history`): years of Binance public data (1m bars, tick trades, funding,
   open interest, futures book depth) downloaded from data.binance.vision.
2. **Recorder** (`python -m recorder`): live spot trades and the full order book, plus futures
   trades, mark price/funding, open interest and liquidations. The order book and liquidations have
   no public history, so they have to be recorded going forward.

Spot is what we trade. Futures data is only used for signals: funding, open interest and the
perp premium often lead spot.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Historical data

```bash
python -m history --dry-run             # what would be downloaded, and how big
python -m history                       # download / update everything; re-run any time
python scripts/inspect_history.py       # coverage report: ranges, rows, missing minutes
```

| dataset | contents | from | download |
|---|---|---|---|
| `spot/klines_1m` | 1m OHLCV with taker-buy volume and trade count | 2017-08 | 0.24 GB |
| `spot/aggTrades` | every trade (aggregated by taker order) | 2017-08 | ~0.4 GB/month |
| `futures/klines_1m` | USDⓈ-M perp 1m OHLCV | 2020-01 | 0.15 GB |
| `futures/aggTrades` | every perp trade | 2020-01 | ~0.4 GB/month |
| `futures/premiumIndex_1m` | perp premium over the index (basis), 1m OHLC | 2020-01 | 0.07 GB |
| `futures/fundingRate` | funding rate at each settlement | 2020-01 | tiny |
| `futures/metrics` | 5m open interest and long/short ratios | 2020-09 | 0.03 GB |
| `futures/bookDepth` | perp depth within ±1–5% of mid, ~30s samples | 2023-01 | 0.66 GB |

### Tick trades: kept as 1m bars, raw only for recent months

Every downloaded `aggTrades` file is also summarised into **1-minute trade-flow bars**
(`spot/tradeBars_1m`, `futures/tradeBars_1m`), about 1% of the raw size. Each bar has:
- OHLC, volume, quote volume, and taker-buy volume
- trade and taker-order counts, and the largest taker order
- volume from small (< $1k) and large (≥ $100k) taker orders, total and buy side

A taker order is a run of trades with the same timestamp and side, i.e. one market order sweeping
several levels. See [history/bars.py](history/bars.py).

Raw ticks are kept only for the last `--keep-raw-months` months (default 3, including the current
one). Older raw files are deleted once their bars are built, because Binance's archive is a
permanent backup. To rebuild bars after adding columns, delete the `tradeBars_1m` folder and re-run:
each month is re-downloaded, summarised and pruned again. `--keep-raw-months 0` keeps bars only.

Downloading tick trades back to launch means about 105 GB of transfer, but it leaves only about
1 GB of bars on disk, plus a few GB of recent raw ticks. They start at `--trades-start` (default
`2026-01`). For the full history, run:

```bash
python -m history --trades-start 2017-08 --dry-run   # check the transfer size first
python -m history --trades-start 2017-08
```

Other options: `--datasets`, `--start`, `--end`, `--jobs`, `--out`.

Files land in `data/history/<market>/<dataset>/BTCUSDT/<YYYY-MM>.parquet`. The current month comes
from daily files (`<YYYY-MM-DD>.parquet`), which are replaced once Binance publishes the monthly file.
Every zip is checked against its published SHA-256. Every timestamp is converted to int64 UTC
microseconds (`*_us` columns), because Binance's raw files mix milliseconds and microseconds.
Integer columns are delta-encoded, which roughly halves raw tick files without losing anything.

```python
from history.datasets import load
bars = load("spot/tradeBars_1m", start="2024-01", end="2024-06")
```

Binance's archive has holes: for example, futures 1m bars are missing all of 2026-10-28. Run the
coverage report and handle gaps explicitly rather than forward-filling across them.

Liquidations aren't in the archive at all; they would have to be recorded live.

## Features

```bash
python -m features                      # -> data/features/BTCUSDT_1h.parquet, plus a coverage report
```

[features/build.py](features/build.py) turns the history into one row per hour (`--freq 15min` etc.
for denser sampling). Each row stamped `t` holds:
- features computed only from data public before `t`
- `close`, the spot close at `t`
- targets `fwd_ret_1h`, `fwd_ret_4h` and `fwd_ret_24h`, the forward log returns of spot

| group | features | from |
|---|---|---|
| price/volatility | returns and realised vol over 1h–7d, range, position in the 24h/7d range, distance from 24h VWAP, volume vs its 7d norm | 2017 |
| spot flow | taker buy/sell imbalance; large- and small-order imbalance, large-order share, average and largest order size | 2017 / trade bars |
| futures | perp taker and large-order imbalance, perp/spot volume, basis, premium index and its 24h change | 2020 |
| positioning | funding (per 8h) and vs its 7d mean; open-interest change over 1h–24h; account, top-trader and taker long/short ratios | 2020 |
| book depth | perp bid/ask imbalance and total notional within ±1% and ±5% | 2023 |
| calendar | hour of day, day of week | |

Datasets missing for a period give NaN, not zeros. Gaps aren't papered over: windows need 80% of
their minutes, and prices are carried forward at most 5 minutes. Open interest is lagged 5 minutes,
because Binance publishes it late. A test perturbs all data after a cutoff and checks that no
feature at or before it changes.

```python
from features import build, feature_columns
df = build(start="2023-01")
```

## Backtesting

```bash
python -m backtest                                  # ridge baseline on fwd_ret_4h, 2+ years of training
python -m backtest --target fwd_ret_24h --enter-bps 40 --delay 1min
```

- **Spot, long or flat only.** No shorting or leverage, since retail derivatives aren't available in
  Australia.
- **Execution at a realistic price.** Each decision at `t` fills at the VWAP of the first minute
  after `t`, not at the close the features were computed from.
- **Costs.** Fees (10 bps, Binance's base spot taker rate) plus slippage (1 bp) are charged on every
  change in position.
- **Data gaps.** If there's no price, nothing trades.
- **Walk-forward ([backtest/walkforward.py](backtest/walkforward.py)).** Every prediction comes from a
  model refit monthly on earlier data only. Training rows whose target wasn't yet known at the start of
  the test month are left out (the embargo), so there's no label overlap.
- **Trading rule.** Go long when the predicted return beats the round-trip cost (`--enter-bps`,
  default 22). Go flat when it drops below `--exit-bps` (default 0). The gap between the two keeps
  the position from flipping on noise.

The report shows:
- predictive IC, i.e. how well predictions line up with the actual returns, month by month, with a
  t-stat (see below)
- return, Sharpe, max drawdown, time in market, trades, turnover and costs, against buy and hold
- returns by year

Predictions and the equity curve are saved to `data/backtests/`. Any model with `fit(X, y)` and
`predict(X)` plugs into `walk_forward`. Two models are included in [backtest/models.py](backtest/models.py):
- `ridge`, a deliberately simple baseline
- `composite`, which each month picks the features with the most reliable past IC and averages
  their signed percentile ranks

### Measuring signals without fooling yourself

```bash
python -m backtest.signals              # every feature x horizon: IC, t-stat, stability, top-vs-bottom spread
python -m backtest.variants             # pre-registered daily long-only comparison (trend filter, sizing, costs)
python -m backtest.refine               # pre-registered model comparison under one sizing rule
python -m backtest.variants --final     # the locked final year: run once, at the end
```

**Locked final year.** Data from 2025-11-01 on is cut off before any model is fitted, so refinement
never sees it. Each strategy is also compared with its Sharpe before costs, and with 200 copies of its
own positions shifted to random dates. Those copies hold the same amount and trade as often, so a
strategy that doesn't beat them has no timing skill.

**Chosen strategy (D2).**
- Signal: baseline composite.
- Sizing: one decision a day at 00:00 UTC. The position is today's prediction as a percentile of all
  past ones, averaged over the last 5 days.
- Trend filter: flat while price is below its 200-day average.
- Band: rebalance only when the target moves 10 points or more.

On the development period, D2 had a Sharpe of 1.02 against 0.95 for buy and hold. Its max drawdown was
−43% against −77%. It beat 91% of random-timed copies, and fees came to 9% of account value. Broader,
t-weighted, grouped and two-horizon models were no better. Without the trend filter, Sharpe was higher
(1.15) but max drawdown was deeper (−52%). The filter was kept for drawdown protection.

Correlating a feature with returns *within* each month is biased for slow price-level features. Even on
a pure random walk it shows strong "mean reversion" (IC about −0.1 to −0.3 at 24h). The cause is the
Dickey-Fuller/Stambaugh small-sample bias.

[backtest/stats.py](backtest/stats.py) avoids it as follows:
- each feature is scored against its own trailing 90 days
- returns are scaled by past volatility
- nothing is demeaned within the window
- the t-stat is Newey-West, because slow features give runs of same-sign months

A test checks the statistic stays near zero on a random walk.

Treat the backtest's profit and loss as the final word. It can't be fooled this way.

## Live recorder

`python -m recorder` runs two recorders side by side, each on its own websocket connection
(`--market spot` or `--market futures` runs just one).

**Spot** (what we trade) is written to `data/binance/btcusdt/`:

| stream | contents |
|---|---|
| `trades` | every public trade: price, qty, aggressor side, trade id |
| `book_ticker` | every change to best bid / best ask |
| `depth_updates` | order-book diffs every 100ms, one row per changed price level (qty 0 = level removed) |
| `depth_snapshots` | full 1000-level book snapshots on connect, on any sequence gap, and every 15 min |

**USDⓈ-M futures** (signals only) is written to `data/binance_futures/btcusdt/`:

| stream | contents |
|---|---|
| `agg_trades` | every perp trade (aggregated by taker order) |
| `mark_price` | mark price, index price, estimated funding rate and next funding time, every second |
| `liquidations` | liquidation orders for **every** futures symbol (filter on `symbol`); `SELL` = a long was liquidated |
| `open_interest` | BTCUSDT open interest, polled over REST every `--oi-interval` seconds (default 10) |

Notes on the futures feed:
- **Liquidations are a sample, not a complete record.** Binance pushes at most one liquidation per symbol per
  second, and its archive has no liquidation history, so recording them live is the only way to get them.
- **Binance has split its futures websocket paths.** `/market` carries trades, mark price and liquidations;
  `/public` carries best bid/ask and depth. The legacy `/stream` path now silently delivers only best bid/ask,
  so the recorder uses `/market`. Perp best bid/ask (~400 msgs/s) isn't recorded: it isn't needed for signals.

Each row has `recv_ns`, the local receive time in nanoseconds, alongside the exchange timestamps. The
backtester will use both to model real latency. Keep your clock NTP-synced. On Windows the clock only ticks
every 1–16 ms, so back-to-back messages can share a `recv_ns`. Always order events by exchange ids (update id,
trade id), never by `recv_ns`. Run the recorder on Linux if you want precise latency measurements.

Files land in `<market>/btcusdt/<stream>/date=YYYY-MM-DD/<first_recv_ns>.parquet` under `data/` (UTC dates,
zstd-compressed, written atomically). Up to 60s of data is buffered in memory and flushed on shutdown.

```bash
python -m recorder                      # Ctrl-C to stop cleanly
python scripts/inspect_data.py          # health check: rows, latency, gaps, funding, liquidations
```

Options: `--market`, `--symbol`, `--out`, `--snapshot-interval`, `--depth-limit`, `--oi-interval`,
`--flush-seconds`, `--log-level`. If `stream.binance.com` is blocked on your network, use `--endpoint vision`
(Binance's market-data-only mirrors for spot).

### Running it for real

Recording only helps if it's continuous. A laptop that sleeps will leave gaps, which you'll see as new
sessions and `connect` snapshots. Run it on an always-on Ubuntu machine: a small cloud VM (Tokyo or
Singapore sit closest to Binance's servers) or a spare PC at home. Avoid US-based servers, because Binance
blocks US IP addresses (HTTP 451).

On a fresh Ubuntu 22.04+ server (x86 or ARM), as a normal user with sudo:

```bash
curl -fsSL https://raw.githubusercontent.com/Zephyr-Howson/BTC-HFT/main/deploy/setup.sh | bash
```

[deploy/setup.sh](deploy/setup.sh) installs Python and chrony (NTP time sync), clones the repo to `~/BTC-HFT`,
and installs two systemd units:
- `btc-recorder.service` runs the recorder, restarts it on failure, and starts it at boot.
- `btc-history.timer` updates the historical data daily at 02:30 UTC.

Re-run the script to pull new code and restart. Set `HISTORY_ARGS="--trades-start 2017-08"` before running it
for full tick history.

```bash
journalctl -u btc-recorder -f                              # live log
~/BTC-HFT/.venv/bin/python ~/BTC-HFT/scripts/inspect_data.py ~/BTC-HFT/data
df -h ~/BTC-HFT/data                                       # disk space
```

The recorder writes roughly 0.3 GB/day in quiet markets, and more on busy days, so plan on 100–200 GB a
year. Check `df` after the first week.

### Rebuilding the order book

For each session, start from a snapshot with `last_update_id = L`. Drop depth events with
`final_update_id <= L`, then apply events in order. The first applied event must satisfy
`first_update_id <= L + 1 <= final_update_id`. Every later event must start at the previous event's
`final_update_id + 1`; otherwise, resync from the next snapshot.
