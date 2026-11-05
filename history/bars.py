"""1-minute trade-flow bars built from aggTrades.

These keep what an hourly model needs from tick data at ~1% of the size:
OHLC, taker buy/sell split, and how much volume came from small vs large
taker orders. Raw ticks can always be re-downloaded to rebuild them with
new columns.

A "taker order" is a run of consecutive aggTrades with the same timestamp
and side: one market order sweeping several price levels. Sizes are bucketed
by order notional in USDT, so the buckets mean the same thing at $4k and $100k BTC.

Checked against Binance's own 1m klines: spot matches exactly. For recent
futures, ~8% of minutes differ slightly because trades within a few ms of a
minute boundary land in the neighbouring minute; daily totals match exactly.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import pyarrow as pa

BAR_US = 60_000_000
SMALL_USD = 1_000      # orders below this notional count as small
LARGE_USD = 100_000    # orders at or above this notional count as large

SCHEMA = pa.schema([
    ("open_time_us", pa.int64()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),             # base asset (BTC)
    ("quote_volume", pa.float64()),       # USDT; vwap = quote_volume / volume
    ("buy_volume", pa.float64()),         # taker-buy BTC; sell = volume - buy_volume
    ("trades", pa.int64()),               # aggTrades
    ("taker_orders", pa.int64()),
    ("buy_taker_orders", pa.int64()),
    ("max_order_qty", pa.float64()),      # largest single taker order, BTC
    ("small_volume", pa.float64()),       # BTC from orders < SMALL_USD
    ("small_buy_volume", pa.float64()),
    ("large_volume", pa.float64()),       # BTC from orders >= LARGE_USD
    ("large_buy_volume", pa.float64()),
])

_SUMS = ["volume", "quote_volume", "buy_volume", "trades", "taker_orders", "buy_taker_orders",
         "small_volume", "small_buy_volume", "large_volume", "large_buy_volume"]


def _partial(table: pa.Table) -> pd.DataFrame:
    """Per-minute stats for one batch of aggTrades (sorted by time)."""
    df = table.select(["price", "quantity", "transact_time_us", "is_buyer_maker"]).to_pandas()
    t, sell = df.transact_time_us, df.is_buyer_maker
    buy = ~sell
    minute = t // BAR_US * BAR_US
    quote = df.price * df.quantity

    # An order split across two batches counts as two orders; with ~1M-row batches that's negligible.
    order = (t.ne(t.shift()) | sell.ne(sell.shift())).cumsum()
    o = pd.DataFrame({"minute": minute, "buy": buy, "qty": df.quantity, "quote": quote}).groupby(order).agg(
        minute=("minute", "first"), buy=("buy", "first"), qty=("qty", "sum"), quote=("quote", "sum"))
    small, large = o.quote < SMALL_USD, o.quote >= LARGE_USD
    orders = pd.DataFrame({
        "taker_orders": 1,
        "buy_taker_orders": o.buy.astype("int64"),
        "max_order_qty": o.qty,
        "small_volume": o.qty.where(small, 0.0),
        "small_buy_volume": o.qty.where(small & o.buy, 0.0),
        "large_volume": o.qty.where(large, 0.0),
        "large_buy_volume": o.qty.where(large & o.buy, 0.0),
    }).groupby(o.minute.values).agg(
        {"taker_orders": "sum", "buy_taker_orders": "sum", "max_order_qty": "max", "small_volume": "sum",
         "small_buy_volume": "sum", "large_volume": "sum", "large_buy_volume": "sum"})

    trades = pd.DataFrame({
        "open": df.price, "high": df.price, "low": df.price, "close": df.price, "volume": df.quantity,
        "quote_volume": quote, "buy_volume": df.quantity.where(buy, 0.0), "trades": 1,
    }).groupby(minute.values).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
         "quote_volume": "sum", "buy_volume": "sum", "trades": "sum"})
    return trades.join(orders)


class BarBuilder:
    """Feed aggTrades batches in time order with add(), then call table()."""

    def __init__(self) -> None:
        self._parts: list[pd.DataFrame] = []

    def add(self, batch: pa.Table) -> None:
        if batch.num_rows:
            self._parts.append(_partial(batch))

    def table(self) -> pa.Table:
        if not self._parts:
            return SCHEMA.empty_table()
        # Minutes that straddle a batch boundary appear in two partials; merge them.
        df = pd.concat(self._parts).groupby(level=0, sort=True).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "max_order_qty": "max",
             **{c: "sum" for c in _SUMS}})
        df = df.rename_axis("open_time_us").reset_index()
        return pa.Table.from_pandas(df[SCHEMA.names], schema=SCHEMA, preserve_index=False)


def build(batches: Iterable[pa.Table]) -> pa.Table:
    builder = BarBuilder()
    for batch in batches:
        builder.add(batch)
    return builder.table()
