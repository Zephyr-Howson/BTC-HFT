"""List, plan and download files from data.binance.vision.

Monthly files are used wherever they exist. Months without one (normally the
current month) are filled from daily files, which are deleted once that
month's monthly file arrives. Every zip is checked against its .CHECKSUM.

Tick-trade files are also summarised into 1m trade-flow bars as they stream
in. Their raw Parquet is then kept only for recent months (keep_raw); older
raw ticks are deleted, since they can be re-downloaded at any time.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq

from .bars import BarBuilder, build
from .datasets import Dataset, convert, in_range, write_table

log = logging.getLogger(__name__)

LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILE_URL = "https://data.binance.vision/"
_LABEL = re.compile(r"-(\d{4}-\d{2}(?:-\d{2})?)\.zip$")
_XML_KEY = re.compile(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", re.S)

Lister = Callable[[str], list[tuple[str, int]]]


@dataclass(frozen=True)
class Item:
    ds: Dataset
    key: str     # bucket key of the zip
    label: str   # YYYY-MM (monthly) or YYYY-MM-DD (daily)
    size: int    # zip size in bytes
    keep_raw: bool = True  # False: keep only the bars once they're built

    @property
    def monthly(self) -> bool:
        return len(self.label) == 7

    def out_path(self, root: Path, symbol: str) -> Path:
        return self.ds.out_dir(root, symbol) / f"{self.label}.parquet"

    def bars_path(self, root: Path, symbol: str) -> Path | None:
        return Path(root) / self.ds.bars / symbol / f"{self.label}.parquet" if self.ds.bars else None

    def needs_download(self, root: Path, symbol: str) -> bool:
        if self.out_path(root, symbol).exists():
            return False
        bars = self.bars_path(root, symbol)
        return bars is None or not bars.exists() or self.keep_raw

    def needs_work(self, root: Path, symbol: str) -> bool:
        if self.needs_download(root, symbol):
            return True
        bars = self.bars_path(root, symbol)
        return bars is not None and (not bars.exists() or (not self.keep_raw and self.out_path(root, symbol).exists()))


def list_keys(prefix: str) -> list[tuple[str, int]]:
    """All (key, size) pairs under a bucket prefix, following S3 pagination."""
    out: list[tuple[str, int]] = []
    marker = ""
    while True:
        query = urllib.parse.urlencode({"prefix": prefix, "marker": marker})
        with urllib.request.urlopen(f"{LIST_URL}?{query}", timeout=30) as resp:
            xml = resp.read().decode()
        page = [(k, int(s)) for k, s in _XML_KEY.findall(xml)]
        out += page
        if "<IsTruncated>true</IsTruncated>" not in xml or not page:
            return out
        marker = page[-1][0]


def plan(ds: Dataset, symbol: str, start: str = "", end: str = "9999", keep_raw_from: str = "",
         lister: Lister = list_keys) -> list[Item]:
    """Files needed to cover [start, end]: monthly where available, daily for the rest.

    Raw tick files for months before keep_raw_from (YYYY-MM) are marked to be summarised then deleted.
    """
    found: dict[str, dict[str, Item]] = {"monthly": {}, "daily": {}}
    for period, items in found.items():
        for key, size in lister(ds.prefix(period, symbol)):
            m = _LABEL.search(key)
            if m and key.endswith(".zip") and in_range(m.group(1), start, end):
                label = m.group(1)
                items[label] = Item(ds, key, label, size, keep_raw=not ds.bars or label[:7] >= keep_raw_from)
    monthly = found["monthly"]
    daily = [i for label, i in found["daily"].items() if label[:7] not in monthly]
    return sorted([*monthly.values(), *daily], key=lambda i: i.label)


def _get(url: str, dest: Path, stop: threading.Event) -> str:
    """Stream url to dest; return the sha256 hex digest."""
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 20):
            if stop.is_set():
                raise InterruptedError("stopped")
            h.update(chunk)
            f.write(chunk)
    return h.hexdigest()


def fetch(item: Item, root: Path, symbol: str, stop: threading.Event, attempts: int = 3) -> int:
    """Download, verify and convert one file, building its bars if it has them. Returns rows written."""
    out, bars = item.out_path(root, symbol), item.bars_path(root, symbol)
    tmp_zip = out.parent / f".{item.label}.zip.tmp"
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(FILE_URL + item.key + ".CHECKSUM", timeout=30) as resp:
                expected = resp.read().decode().split()[0]
            if _get(FILE_URL + item.key, tmp_zip, stop) != expected:
                raise ValueError("checksum mismatch")
            builder = BarBuilder() if bars else None
            rows = convert(tmp_zip, out, item.ds, on_batch=builder.add if builder else None)
            if builder:
                write_table(builder.table(), bars)
            if item.monthly:  # the monthly file supersedes any daily files for that month
                for d in [out.parent] + ([bars.parent] if bars else []):
                    for f in d.glob(f"{item.label}-??.parquet"):
                        f.unlink()
            return rows
        except InterruptedError:
            raise
        except Exception as exc:
            if attempt == attempts:
                raise
            log.warning("%s %s attempt %d/%d failed: %r", item.ds.name, item.label, attempt, attempts, exc)
            time.sleep(5 * attempt)
        finally:
            tmp_zip.unlink(missing_ok=True)
    raise AssertionError("unreachable")


def process(item: Item, root: Path, symbol: str, stop: threading.Event) -> str:
    """Bring one item up to date: download, (re)build bars, prune raw. Returns a log message."""
    out, bars = item.out_path(root, symbol), item.bars_path(root, symbol)
    if item.needs_download(root, symbol):
        msg = f"{fetch(item, root, symbol, stop):,} rows ({item.size / 1e6:.0f} MB)"
    elif bars and not bars.exists():  # raw file from an earlier run: summarise it locally
        batches = (pa.Table.from_batches([b]) for b in pq.ParquetFile(out).iter_batches(1 << 20))
        write_table(build(batches), bars)
        msg = "bars built from local raw file"
    else:
        msg = "raw file pruned"
    if bars and not item.keep_raw:
        out.unlink(missing_ok=True)
    return msg


def sync(items: list[Item], root: Path, symbol: str, jobs: int = 4) -> list[Item]:
    """Bring every item up to date. Returns the items that failed."""
    todo = [i for i in items if i.needs_work(root, symbol)]
    mb = sum(i.size for i in todo if i.needs_download(root, symbol)) / 1e6
    log.info("%d files to process (%.0f MB to download), %d up to date", len(todo), mb, len(items) - len(todo))
    stop = threading.Event()
    failed: list[Item] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(process, i, root, symbol, stop): i for i in todo}
        try:
            for n, fut in enumerate(as_completed(futures), 1):
                item = futures[fut]
                try:
                    log.info("[%d/%d] %s %s: %s", n, len(todo), item.ds.name, item.label, fut.result())
                except Exception as exc:
                    failed.append(item)
                    log.error("[%d/%d] %s %s failed: %r", n, len(todo), item.ds.name, item.label, exc)
        except KeyboardInterrupt:
            stop.set()
            pool.shutdown(cancel_futures=True)
            raise
    return failed
