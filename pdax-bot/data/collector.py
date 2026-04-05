"""
Market data collector.

Receives ticks from MarketFeed and persists them to SQLite.
Also builds synthetic OHLCV candles from raw ticks (since PDAX has no
candlestick endpoint).

Schema
------
ticks table  : raw tick per pair per poll
candles table: 1-minute OHLCV aggregated from ticks
"""

import logging
import sqlite3
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

from config import cfg

logger = logging.getLogger(__name__)

# Candle aggregation window in seconds
CANDLE_WINDOW_SEC = 60


class DataCollector:
    """
    Persist ticks → SQLite and aggregate into 1-min candles.
    Thread-safe: tick() can be called from the feed thread.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or cfg.db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        # Buffer: pair → list of ticks in current candle window
        self._tick_buffer: Dict[str, List[Dict]] = defaultdict(list)
        self._candle_start: Dict[str, float] = {}
        self._init_db()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        import os
        os.makedirs(cfg.log_dir, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        c = self._conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                pair        TEXT    NOT NULL,
                timestamp   REAL    NOT NULL,
                last        REAL,
                bid         REAL,
                ask         REAL,
                mid         REAL,
                spread      REAL,
                volume_24h  REAL,
                bid_depth   REAL,
                ask_depth   REAL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_ticks_pair_ts ON ticks(pair, timestamp)")
        c.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                pair      TEXT  NOT NULL,
                timestamp REAL  NOT NULL,   -- candle open time
                open      REAL,
                high      REAL,
                low       REAL,
                close     REAL,
                volume    REAL,
                num_ticks INTEGER
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_candles_pair_ts ON candles(pair, timestamp)")
        self._conn.commit()
        logger.info("DataCollector: database ready at %s", self._db_path)

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def tick(self, tick: Dict) -> None:
        """
        Called by MarketFeed on every new tick.
        Persists the tick and updates the candle buffer.
        """
        pair = tick["pair"]
        ts = tick["timestamp"]

        with self._lock:
            # Persist raw tick
            self._insert_tick(tick)

            # Initialise candle window if first tick for this pair
            if pair not in self._candle_start:
                self._candle_start[pair] = ts

            # If current window has elapsed, flush the candle
            if ts - self._candle_start[pair] >= CANDLE_WINDOW_SEC:
                self._flush_candle(pair)
                self._candle_start[pair] = ts

            self._tick_buffer[pair].append(tick)

    def _insert_tick(self, tick: Dict) -> None:
        self._conn.execute(
            """
            INSERT INTO ticks
              (pair, timestamp, last, bid, ask, mid, spread, volume_24h, bid_depth, ask_depth)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                tick["pair"],
                tick["timestamp"],
                tick.get("last"),
                tick.get("bid"),
                tick.get("ask"),
                tick.get("mid"),
                tick.get("spread"),
                tick.get("volume_24h"),
                tick.get("bid_depth"),
                tick.get("ask_depth"),
            ),
        )
        self._conn.commit()

    def _flush_candle(self, pair: str) -> None:
        buf = self._tick_buffer.get(pair, [])
        if not buf:
            return

        prices = [t["last"] for t in buf if t.get("last")]
        if not prices:
            self._tick_buffer[pair] = []
            return

        candle_ts = self._candle_start[pair]
        open_  = prices[0]
        high   = max(prices)
        low    = min(prices)
        close  = prices[-1]
        volume = sum(t.get("volume_24h", 0) for t in buf) / len(buf)  # approximate

        self._conn.execute(
            """
            INSERT INTO candles (pair, timestamp, open, high, low, close, volume, num_ticks)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (pair, candle_ts, open_, high, low, close, volume, len(buf)),
        )
        self._conn.commit()
        self._tick_buffer[pair] = []
        logger.debug("Candle flushed: %s @ %.0f  O=%.4f H=%.4f L=%.4f C=%.4f", pair, candle_ts, open_, high, low, close)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def get_candles(self, pair: str, limit: int = 200) -> List[Dict]:
        """
        Return the last `limit` 1-min candles for a pair as a list of dicts,
        ordered oldest → newest.
        """
        rows = self._conn.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE pair = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (pair, limit),
        ).fetchall()

        candles = [
            {"timestamp": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
            for r in reversed(rows)
        ]
        return candles

    def get_ticks(self, pair: str, limit: int = 500) -> List[Dict]:
        """Return the last `limit` raw ticks for a pair."""
        rows = self._conn.execute(
            """
            SELECT timestamp, last, bid, ask, mid, spread, volume_24h, bid_depth, ask_depth
            FROM ticks
            WHERE pair = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (pair, limit),
        ).fetchall()

        keys = ["timestamp", "last", "bid", "ask", "mid", "spread", "volume_24h", "bid_depth", "ask_depth"]
        return [dict(zip(keys, r)) for r in reversed(rows)]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
