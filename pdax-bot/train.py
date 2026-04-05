"""
Entry point: train the PPO agent on historical data.

Usage
-----
  # Train on all configured pairs (uses data already in SQLite)
  python train.py

  # Train only BTC
  python train.py --pair BTCPHP

  # Override timesteps
  python train.py --timesteps 500000

The script:
  1. Loads candles from the SQLite database (must run data collection first).
  2. Trains a PPO agent per pair.
  3. Saves models to models/ directory.

If you have no collected data yet, seed the database by running the bot in
collection-only mode for a while, or import historical OHLCV from a CSV.
"""

import argparse
import logging
import os
import sys

from config import cfg
from agent.policy import TradingAgent
from data.collector import DataCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(cfg.log_dir, "train.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def train_pair(pair: str, timesteps: int, collector: DataCollector) -> None:
    candles = collector.get_candles(pair, limit=50_000)
    if len(candles) <= cfg.lookback_window:
        logger.warning(
            "[%s] Only %d candles available (need > %d). Skipping.",
            pair, len(candles), cfg.lookback_window,
        )
        return

    logger.info("[%s] Training on %d candles, %d timesteps.", pair, len(candles), timesteps)
    agent = TradingAgent(pair=pair)

    # Resume from checkpoint if available
    agent.load()

    agent.train(candles, total_timesteps=timesteps)
    logger.info("[%s] Training complete.", pair)


def main():
    os.makedirs(cfg.log_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train PDAX RL trading bot")
    parser.add_argument("--pair", type=str, default=None, help="Single pair to train (e.g. BTCPHP). Default: all pairs.")
    parser.add_argument("--timesteps", type=int, default=200_000, help="PPO training timesteps per pair.")
    args = parser.parse_args()

    collector = DataCollector()
    pairs = [args.pair] if args.pair else cfg.pairs

    for pair in pairs:
        try:
            train_pair(pair, args.timesteps, collector)
        except Exception as e:
            logger.error("[%s] Training failed: %s", pair, e, exc_info=True)

    collector.close()
    logger.info("All training complete.")


if __name__ == "__main__":
    main()
