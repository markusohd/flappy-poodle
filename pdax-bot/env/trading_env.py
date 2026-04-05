"""
Custom Gymnasium trading environment for PDAX.

Observation space : flat float32 vector from data/features.py (FEATURE_DIM)
Action space      : Discrete(7)
  0  HOLD
  1  BUY  small  (10% of PHP balance)
  2  BUY  medium (25% of PHP balance)
  3  BUY  large  (50% of PHP balance)  ← aggressive
  4  SELL small  (10% of holdings)
  5  SELL medium (50% of holdings)
  6  SELL all

Reward            : step PnL (in PHP) after fees, minus a small holding penalty
                    to discourage perpetual inaction.

Episode ends when:
  - We've consumed all candles (backtesting mode), or
  - PHP balance + position value drops below 20% of starting capital (ruin stop).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import cfg
from data.features import build_state, FEATURE_DIM

logger = logging.getLogger(__name__)

# Action constants
HOLD        = 0
BUY_SMALL   = 1
BUY_MED     = 2
BUY_LARGE   = 3
SELL_SMALL  = 4
SELL_MED    = 5
SELL_ALL    = 6

N_ACTIONS = 7

# Fraction of PHP balance to spend per buy action
BUY_FRACTIONS  = {BUY_SMALL: 0.10, BUY_MED: 0.25, BUY_LARGE: 0.50}
# Fraction of crypto holdings to sell per sell action
SELL_FRACTIONS = {SELL_SMALL: 0.10, SELL_MED: 0.50, SELL_ALL: 1.00}


class TradingEnv(gym.Env):
    """
    Backtesting RL environment.
    Feed it a list of candles (from DataCollector) and it simulates trading.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candles: List[Dict],
        pair: str = "BTCPHP",
        initial_php: float = 10_000.0,
        fee_rate: Optional[float] = None,
        lookback: int = 60,
    ):
        super().__init__()
        assert len(candles) > lookback, "Need more candles than lookback window"

        self.candles      = candles
        self.pair         = pair
        self.initial_php  = initial_php
        self.fee_rate     = fee_rate if fee_rate is not None else cfg.fee_rate
        self.lookback     = lookback

        # ── Spaces ─────────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEATURE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # ── State (reset in reset()) ───────────────────────────────────────────
        self.step_idx    = 0
        self.php_balance = 0.0
        self.crypto_qty  = 0.0
        self.entry_price = 0.0
        self.trades_done = 0

    # ── Gym interface ──────────────────────────────────────────────────────────

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.step_idx    = self.lookback
        self.php_balance = self.initial_php
        self.crypto_qty  = 0.0
        self.entry_price = 0.0
        self.trades_done = 0
        return self._obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        candle      = self.candles[self.step_idx]
        price       = candle["close"]
        prev_value  = self._total_value(price)

        reward = self._execute_action(action, price)

        self.step_idx += 1
        terminated = self.step_idx >= len(self.candles)

        # Ruin stop: if portfolio value < 20% of starting capital
        cur_value = self._total_value(self.candles[self.step_idx - 1]["close"])
        truncated = cur_value < self.initial_php * 0.20

        info = {
            "pair":         self.pair,
            "step":         self.step_idx,
            "price":        price,
            "php_balance":  self.php_balance,
            "crypto_qty":   self.crypto_qty,
            "total_value":  cur_value,
            "trades_done":  self.trades_done,
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        price = self.candles[self.step_idx - 1]["close"]
        print(
            f"[{self.pair}] step={self.step_idx}  price={price:.2f}"
            f"  PHP={self.php_balance:.2f}  crypto={self.crypto_qty:.6f}"
            f"  value={self._total_value(price):.2f}"
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        window = self.candles[self.step_idx - self.lookback: self.step_idx]
        price  = window[-1]["close"] if window else 1.0
        entry  = self.entry_price if self.entry_price > 0 else price
        unrealised_pnl = (price - entry) / entry if self.crypto_qty > 0 else 0.0

        # Order book proxies from candle spread (we don't have live OB here)
        last = window[-1] if window else {}
        bid_depth   = last.get("bid_depth", 0.0) or 0.0
        ask_depth   = last.get("ask_depth", 0.0) or 0.0
        spread_pct  = last.get("spread", 0.0) / (price or 1.0)

        position = 1.0 if self.crypto_qty > 0 else 0.0

        return build_state(
            window,
            current_position=position,
            unrealised_pnl_pct=unrealised_pnl,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            spread_pct=spread_pct,
        )

    def _total_value(self, price: float) -> float:
        return self.php_balance + self.crypto_qty * price

    def _execute_action(self, action: int, price: float) -> float:
        """
        Execute the chosen action, update balances, compute reward.
        Returns the step reward (PHP PnL after fees).
        """
        fee = self.fee_rate
        prev_value = self._total_value(price)
        holding_penalty = 0.0

        if action == HOLD:
            # Small penalty for holding to encourage decisiveness
            holding_penalty = -0.01

        elif action in BUY_FRACTIONS:
            fraction  = BUY_FRACTIONS[action]
            php_spend = self.php_balance * fraction
            if php_spend >= cfg.min_order_php:
                cost       = php_spend * (1 + fee)
                if cost <= self.php_balance:
                    bought           = php_spend / price
                    self.php_balance -= cost
                    # Update average entry price
                    total_qty        = self.crypto_qty + bought
                    if total_qty > 0:
                        self.entry_price = (
                            (self.crypto_qty * self.entry_price + bought * price) / total_qty
                        )
                    self.crypto_qty  = total_qty
                    self.trades_done += 1

        elif action in SELL_FRACTIONS:
            fraction   = SELL_FRACTIONS[action]
            qty_sell   = self.crypto_qty * fraction
            if qty_sell > 0:
                gross            = qty_sell * price
                net              = gross * (1 - fee)
                self.php_balance += net
                self.crypto_qty  -= qty_sell
                if self.crypto_qty <= 0:
                    self.crypto_qty  = 0.0
                    self.entry_price = 0.0
                self.trades_done += 1

        new_value = self._total_value(price)
        reward    = (new_value - prev_value) + holding_penalty
        return float(reward)
