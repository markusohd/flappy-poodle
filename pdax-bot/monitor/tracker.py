"""
Trade & P&L tracker.

Logs every trade to SQLite and prints a live dashboard to stdout.
Also tracks running statistics used to trigger fine-tuning.
"""

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import cfg

logger = logging.getLogger(__name__)

TRADES_DB = os.path.join(cfg.log_dir, "trades.db")


@dataclass
class TradeSummary:
    pair:          str
    side:          str        # 'buy' or 'sell'
    quantity:      float
    price:         float
    php_value:     float      # quantity * price
    fee_php:       float
    net_php:       float      # php_value ± fee
    timestamp:     float
    order_id:      str = ""
    action_idx:    int = -1   # RL action that triggered this
    obs_vector:    Optional[List[float]] = None


@dataclass
class SessionStats:
    start_time:      float = field(default_factory=time.time)
    trades_total:    int   = 0
    trades_winning:  int   = 0
    trades_losing:   int   = 0
    gross_pnl:       float = 0.0   # PHP
    total_fees:      float = 0.0   # PHP
    net_pnl:         float = 0.0   # PHP
    peak_value:      float = 0.0
    trough_value:    float = 0.0
    max_drawdown:    float = 0.0


class TradeTracker:
    """Records trades, computes PnL, and decides when to trigger fine-tuning."""

    def __init__(self, initial_php: float = 0.0):
        self._lock     = threading.Lock()
        self._stats    = SessionStats()
        self._stats.peak_value   = initial_php
        self._stats.trough_value = initial_php
        self._trades_since_finetune = 0
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ── DB setup ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        self._conn = sqlite3.connect(TRADES_DB, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                pair        TEXT,
                side        TEXT,
                quantity    REAL,
                price       REAL,
                php_value   REAL,
                fee_php     REAL,
                net_php     REAL,
                timestamp   REAL,
                order_id    TEXT,
                action_idx  INTEGER
            )
        """)
        self._conn.commit()

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, trade: TradeSummary) -> None:
        """Log a completed trade and update running stats."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO trades
                  (pair, side, quantity, price, php_value, fee_php, net_php, timestamp, order_id, action_idx)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    trade.pair, trade.side, trade.quantity, trade.price,
                    trade.php_value, trade.fee_php, trade.net_php,
                    trade.timestamp, trade.order_id, trade.action_idx,
                ),
            )
            self._conn.commit()

            self._stats.trades_total  += 1
            self._stats.total_fees    += trade.fee_php
            self._stats.net_pnl       += trade.net_php

            if trade.side == "sell":
                if trade.net_php > 0:
                    self._stats.trades_winning += 1
                else:
                    self._stats.trades_losing += 1
                self._stats.gross_pnl += trade.net_php + trade.fee_php

            self._trades_since_finetune += 1

            logger.info(
                "TRADE [%s] %s %.6f @ %.2f  net=%.2f PHP  total_net_pnl=%.2f PHP",
                trade.pair, trade.side.upper(), trade.quantity, trade.price,
                trade.net_php, self._stats.net_pnl,
            )

    def update_portfolio_value(self, value: float) -> None:
        """Call after each step with the current total portfolio value."""
        with self._lock:
            if value > self._stats.peak_value:
                self._stats.peak_value = value
            if value < self._stats.trough_value:
                self._stats.trough_value = value
            if self._stats.peak_value > 0:
                drawdown = (self._stats.peak_value - value) / self._stats.peak_value
                self._stats.max_drawdown = max(self._stats.max_drawdown, drawdown)

    # ── Fine-tune trigger ─────────────────────────────────────────────────────

    def should_finetune(self) -> bool:
        """Return True when enough new trades have accumulated to justify fine-tuning."""
        return self._trades_since_finetune >= cfg.retrain_every_n_trades

    def reset_finetune_counter(self) -> None:
        with self._lock:
            self._trades_since_finetune = 0

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        s = self._stats
        win_rate = (
            s.trades_winning / max(s.trades_winning + s.trades_losing, 1) * 100
        )
        elapsed = (time.time() - s.start_time) / 3600
        print(
            f"\n{'─'*50}\n"
            f"  Session P&L      : {s.net_pnl:+.2f} PHP\n"
            f"  Gross P&L        : {s.gross_pnl:+.2f} PHP\n"
            f"  Total fees       : {s.total_fees:.2f} PHP\n"
            f"  Trades           : {s.trades_total}  (W:{s.trades_winning} / L:{s.trades_losing})\n"
            f"  Win rate         : {win_rate:.1f}%\n"
            f"  Max drawdown     : {s.max_drawdown*100:.1f}%\n"
            f"  Elapsed          : {elapsed:.1f}h\n"
            f"{'─'*50}\n"
        )

    def get_stats(self) -> Dict:
        s = self._stats
        return {
            "trades_total":   s.trades_total,
            "net_pnl":        s.net_pnl,
            "gross_pnl":      s.gross_pnl,
            "total_fees":     s.total_fees,
            "trades_winning": s.trades_winning,
            "trades_losing":  s.trades_losing,
            "max_drawdown":   s.max_drawdown,
        }

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        cols = ["id", "pair", "side", "quantity", "price", "php_value",
                "fee_php", "net_php", "timestamp", "order_id", "action_idx"]
        return [dict(zip(cols, r)) for r in rows]
