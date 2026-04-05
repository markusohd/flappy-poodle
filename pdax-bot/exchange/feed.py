"""
Market data feed via REST polling (PDAX has no public WebSocket).

Continuously polls the PDAX ticker and order book for all configured pairs
and emits tick events to registered callbacks.

Tick dict structure:
  {
    "pair":        str,      # e.g. "BTCPHP"
    "timestamp":   float,    # unix epoch seconds
    "last":        float,    # last traded price
    "bid":         float,    # best bid
    "ask":         float,    # best ask
    "volume_24h":  float,    # 24-hour volume
    "bid_depth":   float,    # total bid volume (top of book)
    "ask_depth":   float,    # total ask volume (top of book)
    "spread":      float,    # ask - bid
    "mid":         float,    # (ask + bid) / 2
  }
"""

import logging
import threading
import time
from typing import Callable, Dict, List, Optional

from config import cfg
from exchange.client import PDAXClient

logger = logging.getLogger(__name__)

TickCallback = Callable[[Dict], None]


class MarketFeed:
    """
    Polls PDAX REST API on a fixed interval and fires tick callbacks.
    Runs in a background daemon thread.
    """

    def __init__(self, client: Optional[PDAXClient] = None):
        self._client = client or PDAXClient()
        self._callbacks: List[TickCallback] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Latest tick per pair – allows synchronous reads without waiting
        self._latest: Dict[str, Dict] = {}

    # ── Subscription ──────────────────────────────────────────────────────────

    def subscribe(self, callback: TickCallback) -> None:
        """Register a function to be called on every new tick."""
        self._callbacks.append(callback)

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start polling in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("MarketFeed started (interval=%.1fs, pairs=%d)", cfg.poll_interval_sec, len(cfg.pairs))

    def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=cfg.poll_interval_sec + 2)
        logger.info("MarketFeed stopped.")

    # ── Latest snapshot ───────────────────────────────────────────────────────

    def latest(self, pair: str) -> Optional[Dict]:
        """Return the most recent tick for a pair, or None if not yet received."""
        return self._latest.get(pair)

    def all_latest(self) -> Dict[str, Dict]:
        return dict(self._latest)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._fetch_and_emit()
            except Exception as e:
                logger.warning("Feed poll error: %s", e)
            time.sleep(cfg.poll_interval_sec)

    def _fetch_and_emit(self) -> None:
        now = time.time()

        # Fetch ticker for all pairs
        try:
            ticker_data = self._client.get_ticker()
        except Exception as e:
            logger.warning("Ticker fetch failed: %s", e)
            return

        # ticker_data is expected to be a list or dict keyed by pair
        # We normalise to {pair: ticker_entry}
        ticker_map: Dict[str, Dict] = {}
        if isinstance(ticker_data, list):
            for entry in ticker_data:
                pair = entry.get("pair") or entry.get("symbol", "")
                if pair:
                    ticker_map[pair] = entry
        elif isinstance(ticker_data, dict):
            ticker_map = ticker_data

        for pair in cfg.pairs:
            entry = ticker_map.get(pair)
            if not entry:
                continue

            # Build a normalised tick
            tick = self._normalise_ticker(pair, entry, now)

            # Optionally enrich with order book depth
            try:
                ob = self._client.get_order_book(pair)
                tick["bid_depth"] = sum(float(b[1]) for b in ob.get("bids", [])[:5])
                tick["ask_depth"] = sum(float(a[1]) for a in ob.get("asks", [])[:5])
            except Exception:
                tick["bid_depth"] = 0.0
                tick["ask_depth"] = 0.0

            tick["spread"] = tick["ask"] - tick["bid"]
            tick["mid"] = (tick["ask"] + tick["bid"]) / 2.0

            self._latest[pair] = tick

            for cb in self._callbacks:
                try:
                    cb(tick)
                except Exception as e:
                    logger.error("Tick callback error: %s", e)

    @staticmethod
    def _normalise_ticker(pair: str, entry: Dict, timestamp: float) -> Dict:
        """Map various PDAX ticker field names to our standard schema."""
        def _f(keys, default=0.0) -> float:
            for k in keys:
                v = entry.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        pass
            return default

        return {
            "pair":       pair,
            "timestamp":  timestamp,
            "last":       _f(["last", "last_price", "price"]),
            "bid":        _f(["bid", "best_bid"]),
            "ask":        _f(["ask", "best_ask"]),
            "volume_24h": _f(["volume", "volume_24h", "base_volume"]),
        }
