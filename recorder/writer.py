"""Buffered Parquet writer: collects rows in memory and flushes them to
date-partitioned files, e.g. data/binance/btcusdt/trades/date=2026-10-29/<ns>.parquet
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


class BufferedParquetWriter:
    def __init__(
        self,
        root: Path,
        stream: str,
        schema: pa.Schema,
        max_rows: int = 200_000,
        max_seconds: float = 60.0,
    ) -> None:
        self.dir = Path(root) / stream
        self.stream = stream
        self.schema = schema
        self.max_rows = max_rows
        self.max_seconds = max_seconds
        self._cols: dict[str, list] = self._empty()
        self._n = 0
        self._last_flush = time.monotonic()
        self._lock = asyncio.Lock()
        self.rows_written = 0

    def _empty(self) -> dict[str, list]:
        return {name: [] for name in self.schema.names}

    def append(self, row: dict) -> None:
        for name, col in self._cols.items():
            col.append(row[name])
        self._n += 1

    def extend(self, rows: list[dict]) -> None:
        for row in rows:
            self.append(row)

    def should_flush(self) -> bool:
        if self._n >= self.max_rows:
            return True
        return self._n > 0 and time.monotonic() - self._last_flush >= self.max_seconds

    async def maybe_flush(self) -> None:
        if self.should_flush():
            await self.flush()

    async def flush(self) -> None:
        async with self._lock:
            self._last_flush = time.monotonic()
            if self._n == 0:
                return
            # Swap buffers before awaiting so new rows keep arriving safely.
            cols, self._cols = self._cols, self._empty()
            n, self._n = self._n, 0
            await asyncio.to_thread(self._write, cols)
            self.rows_written += n

    def _write(self, cols: dict[str, list]) -> None:
        table = pa.table(cols, schema=self.schema)
        first_ns = cols["recv_ns"][0]
        day = datetime.fromtimestamp(first_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")
        out_dir = self.dir / f"date={day}"
        out_dir.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"{first_ns}.parquet"
        tmp = final.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, final)  # atomic: readers never see half-written files
        log.debug("wrote %d rows -> %s", table.num_rows, final)
