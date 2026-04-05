"""
Standalone data collection script.

Connects to the PDAX market feed and stores ticks + candles to SQLite
WITHOUT placing any trades. Run this before training to accumulate
live PDAX market data on top of the historical data imported from CoinGecko.

The more live data you collect, the better the agent learns real PDAX
microstructure (spreads, order book depth, bid/ask imbalance) that
CoinGecko doesn't provide.

Usage
-----
  # Collect data for all configured pairs until Ctrl-C
  python scripts/collect.py

  # Collect only BTC data
  python scripts/collect.py --pair BTCPHP

  # Collect for a fixed duration (seconds)
  python scripts/collect.py --duration 3600
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from data.collector import DataCollector
from exchange.client import PDAXClient
from exchange.feed import MarketFeed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(cfg.log_dir, "collect.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PDAX live market data collector")
    parser.add_argument("--pair",     type=str,   default=None,  help="Collect only this pair.")
    parser.add_argument("--duration", type=float, default=None,  help="Stop after this many seconds.")
    args = parser.parse_args()

    os.makedirs(cfg.log_dir, exist_ok=True)

    # Override pairs if --pair given
    if args.pair:
        cfg.pairs = [args.pair]

    client    = PDAXClient()
    collector = DataCollector()
    feed      = MarketFeed(client=client)

    tick_counts: dict = {p: 0 for p in cfg.pairs}
    start_time = time.time()

    def on_tick(tick):
        pair = tick["pair"]
        collector.tick(tick)
        tick_counts[pair] = tick_counts.get(pair, 0) + 1

    feed.subscribe(on_tick)

    stop_event = threading.Event()

    def _shutdown(sig, frame):
        logger.info("Stopping collection...")
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    feed.start()
    logger.info("Collecting data for: %s", cfg.pairs)
    logger.info("Press Ctrl-C to stop.")

    try:
        while not stop_event.is_set():
            time.sleep(30)
            elapsed = time.time() - start_time
            total_ticks = sum(tick_counts.values())
            logger.info(
                "Elapsed: %.0fs | Total ticks: %d | %s",
                elapsed,
                total_ticks,
                "  ".join(f"{p}:{n}" for p, n in tick_counts.items()),
            )
            if args.duration and elapsed >= args.duration:
                logger.info("Collection duration reached. Stopping.")
                break
    finally:
        feed.stop()
        collector.close()
        elapsed = time.time() - start_time
        logger.info("Collection complete. Ran for %.0fs. Ticks: %s", elapsed, tick_counts)


if __name__ == "__main__":
    main()
