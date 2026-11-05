"""Run with: python -m history [--dry-run] [--datasets ...] [--trades-start 2026-01] [--keep-raw-months 3]"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .datasets import DATASETS
from .vision import plan, sync


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Binance historical data to Parquet.")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--out", default="data/history", help="output root directory")
    p.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    p.add_argument("--start", default="", help="YYYY-MM or YYYY-MM-DD (default: earliest available)")
    p.add_argument("--end", default="9999", help="YYYY-MM or YYYY-MM-DD (default: latest available)")
    p.add_argument("--trades-start", default="2026-01",
                   help="start for the large tick-trade datasets (~0.8 GB/month to download, spot + futures)")
    p.add_argument("--keep-raw-months", type=int, default=3,
                   help="months of raw ticks to keep (incl. the current one); older ones are kept only as 1m bars")
    p.add_argument("--jobs", type=int, default=4, help="parallel downloads")
    p.add_argument("--dry-run", action="store_true", help="show what would be downloaded, then exit")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def months_back(n: int) -> str:
    """YYYY-MM of the month n-1 months before this one (n=1: this month). n<=0: never."""
    if n <= 0:
        return "9999"
    now = datetime.now(timezone.utc)
    y, m = divmod(now.year * 12 + now.month - 1 - (n - 1), 12)
    return f"{y:04d}-{m + 1:02d}"


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = Path(args.out)
    keep_raw_from = months_back(args.keep_raw_months)

    items = []
    for name in args.datasets:
        ds = DATASETS[name]
        start = max(args.start, args.trades_start) if ds.bars else args.start
        found = plan(ds, args.symbol, start, args.end, keep_raw_from)
        todo = [i for i in found if i.needs_download(root, args.symbol)]
        span = f"{found[0].label} -> {found[-1].label}" if found else "nothing available"
        print(f"{name:24s} {span:28s} files={len(found):5d}  to fetch={len(todo):5d} "
              f"({sum(i.size for i in todo) / 1e9:.2f} GB)", flush=True)
        items += found

    if args.dry_run:
        return 0
    failed = sync(items, root, args.symbol, jobs=args.jobs)
    if failed:
        print(f"{len(failed)} files failed; re-run to retry them:", *(f"  {i.ds.name} {i.label}" for i in failed),
              sep="\n")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
