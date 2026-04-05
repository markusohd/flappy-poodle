"""
Historical data importer.

Fetches OHLCV candle data from CoinGecko (free, no API key needed) and
converts prices to PHP using historical USD/PHP exchange rates from
the Open Exchange Rates free tier or a fixed fallback rate.

Supported coins mapped to PDAX pairs:
  bitcoin   → BTCPHP
  ethereum  → ETHPHP
  ripple    → XRPPHP
  solana    → SOLPHP
  binancecoin → BNBPHP
  cardano   → ADAPHP
  avalanche-2 → AVAXPHP
  polkadot  → DOTPHP
  dogecoin  → DOGEPHP
  litecoin  → LTCPHP
  bitcoin-cash → BCHPHP
  tron      → TRXPHP
  chainlink → LINKPHP

CoinGecko free tier returns up to 365 days of daily OHLCV for /coins/{id}/ohlc.
For hourly data (better for intraday trading) it returns up to 90 days.

Usage
-----
  # Import 90 days of hourly candles for all configured pairs
  python scripts/import_history.py

  # Import only BTC, 30 days
  python scripts/import_history.py --pair BTCPHP --days 30

  # Use a specific USD/PHP rate instead of live lookup
  python scripts/import_history.py --usdphp 57.0
"""

import argparse
import logging
import os
import sqlite3
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from data.collector import DataCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Coin ID mapping ────────────────────────────────────────────────────────────
# PDAX pair → CoinGecko coin ID
PAIR_TO_COINGECKO: Dict[str, str] = {
    "BTCPHP":  "bitcoin",
    "ETHPHP":  "ethereum",
    "XRPPHP":  "ripple",
    "SOLPHP":  "solana",
    "BNBPHP":  "binancecoin",
    "ADAPHP":  "cardano",
    "AVAXPHP": "avalanche-2",
    "DOTPHP":  "polkadot",
    "DOGEPHP": "dogecoin",
    "LTCPHP":  "litecoin",
    "BCHPHP":  "bitcoin-cash",
    "TRXPHP":  "tron",
    "LINKPHP": "chainlink",
    "USDTPHP": "tether",
    "USDCPHP": "usd-coin",
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Fallback USD/PHP rate if live lookup fails
FALLBACK_USDPHP = 57.0


def get_usd_php_rate() -> float:
    """Fetch current USD/PHP rate from a free public API."""
    try:
        resp = requests.get(
            "https://open.er-api.com/v6/latest/USD",
            timeout=8,
        )
        resp.raise_for_status()
        rate = resp.json()["rates"].get("PHP", FALLBACK_USDPHP)
        logger.info("USD/PHP rate: %.4f", rate)
        return float(rate)
    except Exception as e:
        logger.warning("Could not fetch USD/PHP rate (%s). Using fallback %.2f.", e, FALLBACK_USDPHP)
        return FALLBACK_USDPHP


def fetch_ohlcv_coingecko(coin_id: str, days: int) -> List[List]:
    """
    Fetch OHLCV from CoinGecko in USD.
    Returns list of [timestamp_ms, open, high, low, close].
    CoinGecko returns hourly candles for days <= 90, daily otherwise.
    """
    url = f"{COINGECKO_BASE}/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": str(days)}

    retries = 4
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                logger.warning("Rate limited by CoinGecko. Waiting %ds...", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            logger.info("Fetched %d candles for %s (%d days)", len(data), coin_id, days)
            return data
        except requests.RequestException as e:
            logger.warning("CoinGecko fetch attempt %d/%d failed: %s", attempt + 1, retries, e)
            time.sleep(5 * (2 ** attempt))

    logger.error("Failed to fetch data for %s after %d attempts.", coin_id, retries)
    return []


def import_pair(
    pair: str,
    days: int,
    usd_php: float,
    collector: DataCollector,
) -> int:
    """
    Fetch and store candles for one pair. Returns number of candles imported.
    """
    coin_id = PAIR_TO_COINGECKO.get(pair)
    if not coin_id:
        logger.warning("No CoinGecko mapping for %s. Skipping.", pair)
        return 0

    raw = fetch_ohlcv_coingecko(coin_id, days)
    if not raw:
        return 0

    # Stablecoins: price is ~1 USD, PHP value is just usd_php rate
    is_stable = pair in ("USDTPHP", "USDCPHP")

    conn = collector._conn
    inserted = 0
    for row in raw:
        if len(row) < 5:
            continue
        ts_ms, open_usd, high_usd, low_usd, close_usd = row[:5]
        ts_sec = ts_ms / 1000.0

        if is_stable:
            o, h, l, c = usd_php, usd_php, usd_php, usd_php
        else:
            o = open_usd  * usd_php
            h = high_usd  * usd_php
            l = low_usd   * usd_php
            c = close_usd * usd_php

        conn.execute(
            """
            INSERT OR IGNORE INTO candles
              (pair, timestamp, open, high, low, close, volume, num_ticks)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (pair, ts_sec, o, h, l, c, 0.0, 0),
        )
        inserted += 1

    conn.commit()

    # Add a unique index to prevent duplicate imports
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_candles_pair_ts_unique ON candles(pair, timestamp)"
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass

    logger.info("[%s] Imported %d candles (PHP rate=%.2f)", pair, inserted, usd_php)
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Import historical OHLCV data from CoinGecko")
    parser.add_argument("--pair", type=str, default=None,
                        help="Single PDAX pair (e.g. BTCPHP). Default: all configured pairs.")
    parser.add_argument("--days", type=int, default=90,
                        help="Number of days of history to import (max 365). Default: 90.")
    parser.add_argument("--usdphp", type=float, default=None,
                        help="Fixed USD/PHP exchange rate. Default: fetch live.")
    args = parser.parse_args()

    os.makedirs(cfg.log_dir, exist_ok=True)

    usd_php = args.usdphp if args.usdphp else get_usd_php_rate()
    pairs   = [args.pair] if args.pair else list(PAIR_TO_COINGECKO.keys())
    # Only import pairs that are in the bot's configured list
    pairs   = [p for p in pairs if p in cfg.pairs or (args.pair and p == args.pair)]

    if not pairs:
        logger.warning("No matching pairs to import. Check --pair or config.py pairs list.")
        return

    collector = DataCollector()
    total = 0
    for pair in pairs:
        n = import_pair(pair, args.days, usd_php, collector)
        total += n
        # Be polite to CoinGecko free tier (10–15 req/min)
        time.sleep(7)

    collector.close()
    logger.info("Import complete. Total candles stored: %d", total)


if __name__ == "__main__":
    main()
