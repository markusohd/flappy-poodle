"""
Entry point: run the trading bot live against PDAX.

Usage
-----
  # Dry-run (no real orders — just logs what it would do)
  python run.py --dry-run

  # Live trading (sandbox by default; set PDAX_SANDBOX=false for real money)
  python run.py

  # Trade only one pair
  python run.py --pair BTCPHP

The main loop per pair:
  1. MarketFeed polls PDAX REST API every poll_interval_sec.
  2. Each tick is stored by DataCollector.
  3. Every tick, the agent observes the current state and picks an action.
  4. If action is BUY/SELL, place a market order via PDAXClient.
  5. Record the trade in TradeTracker.
  6. After every retrain_every_n_trades trades, fine-tune the agent.

One agent instance per pair runs concurrently in separate threads.
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
import uuid
from typing import Dict, Optional

import numpy as np

from config import cfg
from agent.policy import TradingAgent
from data.collector import DataCollector
from data.features import build_state
from exchange.client import PDAXClient
from exchange.feed import MarketFeed
from memory.replay import ReplayBuffer
from monitor.tracker import TradeTracker, TradeSummary
from env.trading_env import BUY_FRACTIONS, SELL_FRACTIONS, HOLD, N_ACTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(cfg.log_dir, "run.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Per-pair bot state ─────────────────────────────────────────────────────────

class PairBot:
    """
    Manages a single trading pair: observes ticks, decides actions, places orders.
    """

    def __init__(
        self,
        pair: str,
        client: PDAXClient,
        collector: DataCollector,
        replay: ReplayBuffer,
        tracker: TradeTracker,
        dry_run: bool = True,
    ):
        self.pair      = pair
        self.client    = client
        self.collector = collector
        self.replay    = replay
        self.tracker   = tracker
        self.dry_run   = dry_run

        self.agent = TradingAgent(pair=pair)
        if not self.agent.load():
            logger.warning("[%s] No trained model found. Bot will act randomly until trained.", pair)

        # Live position state
        self.crypto_qty:  float = 0.0
        self.entry_price: float = 0.0
        self.last_obs:    Optional[np.ndarray] = None

    def on_tick(self, tick: Dict) -> None:
        """Called by MarketFeed on every new tick for this pair."""
        if tick["pair"] != self.pair:
            return

        # Persist tick
        self.collector.tick(tick)

        # Build observation
        candles = self.collector.get_candles(self.pair, limit=cfg.lookback_window + 5)
        if len(candles) < cfg.lookback_window:
            return  # not enough history yet

        price = tick.get("last") or tick.get("mid") or 0.0
        if price <= 0:
            return

        entry = self.entry_price if self.entry_price > 0 else price
        unrealised_pnl = (price - entry) / entry if self.crypto_qty > 0 else 0.0

        obs = build_state(
            candles,
            current_position=1.0 if self.crypto_qty > 0 else 0.0,
            unrealised_pnl_pct=unrealised_pnl,
            bid_depth=tick.get("bid_depth", 0.0),
            ask_depth=tick.get("ask_depth", 0.0),
            spread_pct=tick.get("spread", 0.0) / (price or 1.0),
        )

        # Agent decision
        action = self.agent.predict(obs, deterministic=True)

        if action != HOLD:
            self._execute_action(action, price, obs, tick["timestamp"])

        self.last_obs = obs

        # Fine-tune trigger
        if self.tracker.should_finetune():
            self._finetune()

    def _execute_action(self, action: int, price: float, obs: np.ndarray, ts: float) -> None:
        side = "buy" if action in BUY_FRACTIONS else "sell"

        try:
            balances = self.client.get_balances()
        except Exception as e:
            logger.warning("[%s] Could not fetch balances: %s", self.pair, e)
            return

        php_bal = balances.get("PHP", 0.0)
        asset   = self.pair.replace("PHP", "")
        crypto_bal = balances.get(asset, 0.0)

        quantity: float = 0.0

        if side == "buy":
            fraction  = BUY_FRACTIONS[action]
            php_spend = php_bal * fraction
            if php_spend < cfg.min_order_php:
                logger.debug("[%s] BUY skipped – insufficient PHP (%.2f)", self.pair, php_spend)
                return
            quantity = php_spend / price

        elif side == "sell":
            fraction = SELL_FRACTIONS[action]
            quantity = crypto_bal * fraction
            if quantity * price < cfg.min_order_php:
                logger.debug("[%s] SELL skipped – quantity too small", self.pair)
                return

        order_id = f"rl_{self.pair}_{int(ts)}_{uuid.uuid4().hex[:6]}"
        fee_php  = quantity * price * cfg.fee_rate
        net_php  = quantity * price * (1 - cfg.fee_rate) if side == "sell" else -(quantity * price * (1 + cfg.fee_rate))

        if self.dry_run:
            logger.info("[DRY-RUN][%s] %s %.6f @ %.2f  fee=%.2f  net=%.2f PHP",
                        self.pair, side.upper(), quantity, price, fee_php, net_php)
        else:
            try:
                resp = self.client.place_market_order(self.pair, side, quantity, custom_id=order_id)
                logger.info("[%s] Order placed: %s", self.pair, resp)
            except Exception as e:
                logger.error("[%s] Order failed: %s", self.pair, e)
                return

        # Update local position state
        if side == "buy":
            total = self.crypto_qty + quantity
            self.entry_price = (
                (self.crypto_qty * self.entry_price + quantity * price) / total
                if total > 0 else price
            )
            self.crypto_qty = total
        else:
            self.crypto_qty = max(0.0, self.crypto_qty - quantity)
            if self.crypto_qty == 0:
                self.entry_price = 0.0

        # Record in tracker
        trade = TradeSummary(
            pair=self.pair, side=side, quantity=quantity, price=price,
            php_value=quantity * price, fee_php=fee_php, net_php=net_php,
            timestamp=ts, order_id=order_id, action_idx=action,
        )
        self.tracker.record(trade)

        # Store in replay buffer
        if self.last_obs is not None:
            self.replay.push({
                "pair":      self.pair,
                "timestamp": ts,
                "obs":       self.last_obs,
                "action":    action,
                "reward":    net_php,
                "next_obs":  obs,
                "done":      False,
                "info":      {"price": price, "side": side},
            })

    def _finetune(self) -> None:
        candles = self.collector.get_candles(self.pair, limit=5_000)
        if len(candles) > cfg.lookback_window:
            logger.info("[%s] Fine-tuning agent...", self.pair)
            self.agent.finetune(candles, timesteps=5_000)
        self.tracker.reset_finetune_counter()


# ── Main orchestrator ──────────────────────────────────────────────────────────

def main():
    os.makedirs(cfg.log_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run PDAX RL trading bot")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Simulate orders without hitting the exchange (default: True)")
    parser.add_argument("--live", action="store_true", default=False,
                        help="Execute real orders (overrides --dry-run)")
    parser.add_argument("--pair", type=str, default=None,
                        help="Run only this pair (e.g. BTCPHP)")
    args = parser.parse_args()

    dry_run = not args.live
    if not dry_run:
        logger.warning("LIVE MODE: real orders will be placed on PDAX!")
        if cfg.use_sandbox:
            logger.info("(sandbox endpoint active — not real money)")

    pairs = [args.pair] if args.pair else cfg.pairs
    client    = PDAXClient()
    collector = DataCollector()
    replay    = ReplayBuffer()

    # Fetch initial balances for the tracker
    try:
        balances  = client.get_balances()
        init_php  = balances.get("PHP", 0.0)
        logger.info("Starting PHP balance: %.2f", init_php)
    except Exception as e:
        logger.warning("Could not fetch initial balance: %s", e)
        init_php = 0.0

    tracker = TradeTracker(initial_php=init_php)

    # Build one PairBot per pair
    bots: Dict[str, PairBot] = {}
    for pair in pairs:
        bots[pair] = PairBot(pair, client, collector, replay, tracker, dry_run=dry_run)

    # Set up market feed
    feed = MarketFeed(client=client)
    for bot in bots.values():
        feed.subscribe(bot.on_tick)

    # Graceful shutdown
    stop_event = threading.Event()

    def _shutdown(sig, frame):
        logger.info("Shutdown signal received.")
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    feed.start()
    logger.info("Bot running on pairs: %s", pairs)

    try:
        while not stop_event.is_set():
            time.sleep(60)
            tracker.print_summary()
    finally:
        feed.stop()
        collector.close()
        tracker.print_summary()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
