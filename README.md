# btc-research

Short-horizon trading research on BTC/USDT. Step 1: a market-data recorder.

## What it records

The recorder holds one websocket connection to Binance spot and writes four streams to Parquet:

| stream | contents |
|---|---|
| `trades` | every public trade: price, qty, aggressor side, trade id |
| `book_ticker` | every change to best bid / best ask |
| `depth_updates` | order-book diffs every 100ms, one row per changed price level (qty 0 = level removed) |
| `depth_snapshots` | full 1000-level book snapshots on connect, on any sequence gap, and every 15 min |

Each row has `recv_ns`, the local receive time in nanoseconds, alongside the exchange timestamps. The backtester will use both to model real latency. Keep your clock NTP-synced.

Files land in `data/binance/btcusdt/<stream>/date=YYYY-MM-DD/<first_recv_ns>.parquet` (UTC dates, zstd-compressed, written atomically). Up to 60s of data is buffered in memory and flushed on shutdown.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m recorder                      # Ctrl-C to stop cleanly
python scripts/inspect_data.py          # health check: row counts, latency, gaps
```

Options: `--symbol`, `--out`, `--snapshot-interval`, `--depth-limit`, `--flush-seconds`, `--log-level`.
If `stream.binance.com` is blocked on your network, use `--endpoint vision` (Binance's market-data-only mirrors).

## Create the GitHub repo

```bash
git init && git add . && git commit -m "Market-data recorder"
gh repo create btc-research --private --source=. --push
```

## Running it for real

Recording only helps if it's continuous. A laptop that sleeps will leave gaps, which you'll see as new
sessions and `connect` snapshots. Run it on an always-on machine or a small cloud VM (Tokyo or Singapore
regions sit closer to Binance's servers) under `tmux` or a systemd service. Check disk usage after day one
and plan storage from there.

## Rebuilding the order book (next step)

For each session, start from a snapshot with `last_update_id = L`. Drop depth events with
`final_update_id <= L`, then apply events in order. The first applied event must satisfy
`first_update_id <= L + 1 <= final_update_id`. Every later event must start at the previous event's
`final_update_id + 1`; otherwise, resync from the next snapshot.
