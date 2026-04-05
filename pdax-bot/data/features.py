"""
Feature engineering for the RL state vector.

Takes a list of candle dicts (oldest→newest) and returns a flat numpy array
suitable for the RL environment's observation space.

Features computed per candle window:
  Price-based   : returns, log-returns, price position within range
  Momentum      : RSI(14), MACD line, MACD signal, MACD histogram
  Trend         : EMA(9), EMA(21), EMA(50) (normalised vs current price)
  Volatility    : Bollinger Band width, %B position
  Volume        : volume normalised by rolling mean
  Order book    : bid/ask depth ratio, spread (normalised)
  Position info : injected externally (current position, unrealised PnL)
"""

import numpy as np
from typing import List, Dict


def _prices(candles: List[Dict]) -> np.ndarray:
    return np.array([c["close"] for c in candles], dtype=np.float64)


def _volumes(candles: List[Dict]) -> np.ndarray:
    return np.array([c.get("volume", 0.0) for c in candles], dtype=np.float64)


# ── Individual indicators ──────────────────────────────────────────────────────

def rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def ema(prices: np.ndarray, period: int) -> np.ndarray:
    if len(prices) == 0:
        return np.array([])
    alpha = 2.0 / (period + 1)
    result = np.empty(len(prices))
    result[0] = prices[0]
    for i in range(1, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]
    return result


def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram) as scalars (last value)."""
    if len(prices) < slow:
        return 0.0, 0.0, 0.0
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line[-1] - signal_line[-1]
    return macd_line[-1], signal_line[-1], histogram


def bollinger_bands(prices: np.ndarray, period: int = 20, num_std: float = 2.0):
    """Returns (upper, middle, lower, %B, bandwidth) as scalars."""
    if len(prices) < period:
        p = prices[-1] if len(prices) > 0 else 0.0
        return p, p, p, 0.5, 0.0
    window = prices[-period:]
    mid = window.mean()
    std = window.std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    current = prices[-1]
    pct_b = (current - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    bandwidth = (upper - lower) / mid if mid > 0 else 0.0
    return upper, mid, lower, pct_b, bandwidth


# ── Main feature builder ───────────────────────────────────────────────────────

def build_state(
    candles: List[Dict],
    current_position: float = 0.0,
    unrealised_pnl_pct: float = 0.0,
    bid_depth: float = 0.0,
    ask_depth: float = 0.0,
    spread_pct: float = 0.0,
) -> np.ndarray:
    """
    Build the RL observation vector from recent candles + live context.

    Parameters
    ----------
    candles             : list of candle dicts, oldest first, length >= lookback
    current_position    : 1.0 = fully long, 0.0 = flat, -1.0 = fully short
    unrealised_pnl_pct  : unrealised PnL as fraction of entry value
    bid_depth / ask_depth : order book depth ratio context
    spread_pct          : (ask-bid)/mid as a fraction

    Returns
    -------
    1-D numpy float32 array
    """
    prices = _prices(candles)
    vols   = _volumes(candles)

    if len(prices) == 0:
        # Return zeros if no data yet
        return np.zeros(30, dtype=np.float32)

    p_now = prices[-1] if prices[-1] != 0 else 1.0

    # ── Log returns (last 10) ──────────────────────────────────────────────────
    log_ret = np.diff(np.log(np.maximum(prices, 1e-9)))
    log_ret_window = log_ret[-10:] if len(log_ret) >= 10 else np.pad(log_ret, (10 - len(log_ret), 0))

    # ── EMAs (normalised by current price) ────────────────────────────────────
    ema9_val  = ema(prices, 9)[-1]  / p_now - 1.0
    ema21_val = ema(prices, 21)[-1] / p_now - 1.0 if len(prices) >= 21 else 0.0
    ema50_val = ema(prices, 50)[-1] / p_now - 1.0 if len(prices) >= 50 else 0.0

    # ── RSI ────────────────────────────────────────────────────────────────────
    rsi_val = rsi(prices) / 100.0  # normalise to [0,1]

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_line, macd_sig, macd_hist = macd(prices)
    # Normalise by price level
    macd_line_n = macd_line / p_now
    macd_sig_n  = macd_sig  / p_now
    macd_hist_n = macd_hist / p_now

    # ── Bollinger ─────────────────────────────────────────────────────────────
    _, _, _, pct_b, bb_width = bollinger_bands(prices)

    # ── Volume (ratio to rolling mean) ────────────────────────────────────────
    vol_mean = vols[-20:].mean() if vols[-20:].mean() > 0 else 1.0
    vol_ratio = vols[-1] / vol_mean if vol_mean > 0 else 1.0
    vol_ratio = np.clip(vol_ratio, 0.0, 5.0) / 5.0  # normalise to [0,1]

    # ── Order book ────────────────────────────────────────────────────────────
    total_depth = bid_depth + ask_depth
    ob_imbalance = bid_depth / total_depth if total_depth > 0 else 0.5
    spread_feat  = np.clip(spread_pct, 0.0, 0.05) / 0.05  # normalise to [0,1]

    # ── Position / PnL ────────────────────────────────────────────────────────
    pos_feat   = np.clip(current_position, -1.0, 1.0)
    pnl_feat   = np.clip(unrealised_pnl_pct, -1.0, 1.0)

    # ── Assemble vector ───────────────────────────────────────────────────────
    features = np.concatenate([
        log_ret_window,            # 10 features
        [ema9_val, ema21_val, ema50_val],  # 3
        [rsi_val],                 # 1
        [macd_line_n, macd_sig_n, macd_hist_n],  # 3
        [pct_b, bb_width],         # 2
        [vol_ratio],               # 1
        [ob_imbalance, spread_feat],  # 2
        [pos_feat, pnl_feat],      # 2
    ]).astype(np.float32)          # total = 24 features

    # Safety: replace NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    return features


# Dimension constant – used by env to declare observation_space
FEATURE_DIM = 24
