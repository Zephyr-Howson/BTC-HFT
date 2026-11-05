# btc-research

Short-horizon trading research on BTC/USDT, built from two data sources:

1. **History** (`python -m history`): years of Binance public data (1m bars, tick trades, funding,
   open interest, futures book depth) downloaded from data.binance.vision.
2. **Recorder** (`python -m recorder`): a live websocket feed of trades and the full order book,
   which has no public history and so has to be recorded going forward.

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

## Live recorder

The recorder holds one websocket connection to Binance spot and writes four streams to Parquet:

| stream | contents |
|---|---|
| `trades` | every public trade: price, qty, aggressor side, trade id |
| `book_ticker` | every change to best bid / best ask |
| `depth_updates` | order-book diffs every 100ms, one row per changed price level (qty 0 = level removed) |
| `depth_snapshots` | full 1000-level book snapshots on connect, on any sequence gap, and every 15 min |

Each row has `recv_ns`, the local receive time in nanoseconds, alongside the exchange timestamps. The backtester will use both to model real latency. Keep your clock NTP-synced.

Files land in `data/binance/btcusdt/<stream>/date=YYYY-MM-DD/<first_recv_ns>.parquet` (UTC dates, zstd-compressed, written atomically). Up to 60s of data is buffered in memory and flushed on shutdown.

```bash
python -m recorder                      # Ctrl-C to stop cleanly
python scripts/inspect_data.py          # health check: row counts, latency, gaps
```

Options: `--symbol`, `--out`, `--snapshot-interval`, `--depth-limit`, `--flush-seconds`, `--log-level`.
If `stream.binance.com` is blocked on your network, use `--endpoint vision` (Binance's market-data-only mirrors).

### Running it for real

Recording only helps if it's continuous. A laptop that sleeps will leave gaps, which you'll see as new
sessions and `connect` snapshots. Run it on an always-on machine or a small cloud VM (Tokyo or Singapore
regions sit closer to Binance's servers) under `tmux` or a systemd service. Check disk usage after day one
and plan storage from there.

### Rebuilding the order book

For each session, start from a snapshot with `last_update_id = L`. Drop depth events with
`final_update_id <= L`, then apply events in order. The first applied event must satisfy
`first_update_id <= L + 1 <= final_update_id`. Every later event must start at the previous event's
`final_update_id + 1`; otherwise, resync from the next snapshot.
