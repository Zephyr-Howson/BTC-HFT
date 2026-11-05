"""Run with: python -m recorder --symbol BTCUSDT --out data [--market spot futures]"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from .binance import ENDPOINTS, BinanceRecorder
from .futures import FuturesRecorder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record Binance spot and futures market data to Parquet.")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--out", default="data", help="output root directory")
    p.add_argument("--market", nargs="+", choices=["spot", "futures"], default=["spot", "futures"],
                   help="which recorders to run (default: both)")
    p.add_argument("--endpoint", choices=sorted(ENDPOINTS), default="default", help="spot endpoints")
    p.add_argument("--snapshot-interval", type=float, default=900.0, help="seconds between periodic spot book snapshots")
    p.add_argument("--depth-limit", type=int, default=1000, help="levels per side in spot REST snapshots")
    p.add_argument("--oi-interval", type=float, default=10.0, help="seconds between futures open-interest polls")
    p.add_argument("--flush-seconds", type=float, default=60.0, help="max seconds of data held in memory")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    recorders = []
    if "spot" in args.market:
        recorders.append(BinanceRecorder(
            symbol=args.symbol,
            out_dir=args.out,
            endpoint=args.endpoint,
            snapshot_interval=args.snapshot_interval,
            depth_limit=args.depth_limit,
            flush_seconds=args.flush_seconds,
        ))
    if "futures" in args.market:
        recorders.append(FuturesRecorder(
            symbol=args.symbol,
            out_dir=args.out,
            oi_interval=args.oi_interval,
            flush_seconds=args.flush_seconds,
        ))
    task = asyncio.gather(*(r.run() for r in recorders))
    # Ctrl-C is handled by asyncio.run; also handle SIGTERM (systemd, docker).
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    except (NotImplementedError, RuntimeError):
        pass  # Windows
    try:
        await task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
