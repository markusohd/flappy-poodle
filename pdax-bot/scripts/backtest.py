"""
Backtest a trained PPO agent against held-out historical candles.

Runs the agent through the TradingEnv in deterministic mode and reports:
  - Total return %
  - Annualised return %
  - Sharpe ratio
  - Max drawdown %
  - Win rate %
  - Number of trades
  - Comparison vs buy-and-hold

Usage
-----
  # Backtest all trained pairs (uses last 20% of stored candles as test set)
  python scripts/backtest.py

  # Backtest a specific pair
  python scripts/backtest.py --pair BTCPHP

  # Use a specific number of candles
  python scripts/backtest.py --pair BTCPHP --candles 1000
"""

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from agent.policy import TradingAgent
from data.collector import DataCollector
from env.trading_env import TradingEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INITIAL_PHP = 10_000.0


def run_backtest(pair: str, candles: list, initial_php: float = INITIAL_PHP) -> dict:
    """
    Run the trained agent through candles and collect episode metrics.
    Returns a metrics dict.
    """
    agent = TradingAgent(pair=pair)
    if not agent.load():
        logger.warning("[%s] No saved model found. Cannot backtest.", pair)
        return {}

    env = TradingEnv(candles, pair=pair, initial_php=initial_php, lookback=cfg.lookback_window)
    obs, _ = env.reset()

    portfolio_values = [initial_php]
    actions_taken    = []
    rewards          = []

    done = False
    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append(info["total_value"])
        actions_taken.append(action)
        rewards.append(reward)

    # ── Metrics ───────────────────────────────────────────────────────────────
    values   = np.array(portfolio_values)
    final    = values[-1]
    total_ret = (final - initial_php) / initial_php * 100.0

    # Annualised return (assume 1-min candles → minutes per year = 525,600)
    n_candles  = len(candles) - cfg.lookback_window
    years      = n_candles / 525_600
    ann_ret    = ((final / initial_php) ** (1.0 / max(years, 1e-6)) - 1) * 100.0 if years > 0 else 0.0

    # Sharpe (daily returns approximation)
    step_rets = np.diff(values) / values[:-1]
    sharpe    = (step_rets.mean() / (step_rets.std() + 1e-9)) * np.sqrt(525_600)

    # Max drawdown
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / (peak + 1e-9)
    max_dd = drawdowns.max() * 100.0

    # Trade stats
    trades     = info["trades_done"]
    non_hold   = [a for a in actions_taken if a != 0]
    buy_count  = sum(1 for a in non_hold if a in (1, 2, 3))
    sell_count = sum(1 for a in non_hold if a in (4, 5, 6))

    # Buy-and-hold baseline
    buy_hold_ret = (candles[-1]["close"] - candles[cfg.lookback_window]["close"]) / candles[cfg.lookback_window]["close"] * 100.0

    return {
        "pair":            pair,
        "initial_php":     initial_php,
        "final_php":       final,
        "total_return_%":  total_ret,
        "ann_return_%":    ann_ret,
        "sharpe":          sharpe,
        "max_drawdown_%":  max_dd,
        "trades":          trades,
        "buys":            buy_count,
        "sells":           sell_count,
        "buy_hold_ret_%":  buy_hold_ret,
        "alpha_%":         total_ret - buy_hold_ret,
        "n_candles":       n_candles,
    }


def print_results(m: dict) -> None:
    if not m:
        return
    print(f"\n{'═'*55}")
    print(f"  Backtest: {m['pair']}")
    print(f"{'═'*55}")
    print(f"  Initial capital     : {m['initial_php']:>10,.2f} PHP")
    print(f"  Final value         : {m['final_php']:>10,.2f} PHP")
    print(f"  Total return        : {m['total_return_%']:>+9.2f} %")
    print(f"  Annualised return   : {m['ann_return_%']:>+9.2f} %")
    print(f"  Sharpe ratio        : {m['sharpe']:>10.3f}")
    print(f"  Max drawdown        : {m['max_drawdown_%']:>9.2f} %")
    print(f"  Trades              : {m['trades']:>10}")
    print(f"    Buys              : {m['buys']:>10}")
    print(f"    Sells             : {m['sells']:>10}")
    print(f"  Buy-and-hold return : {m['buy_hold_ret_%']:>+9.2f} %")
    print(f"  Alpha vs B&H        : {m['alpha_%']:>+9.2f} %")
    print(f"  Candles evaluated   : {m['n_candles']:>10}")
    print(f"{'═'*55}\n")


def main():
    parser = argparse.ArgumentParser(description="Backtest trained PDAX PPO agent")
    parser.add_argument("--pair", type=str, default=None)
    parser.add_argument("--candles", type=int, default=None,
                        help="Number of candles to use as test set. Default: last 20% of stored data.")
    parser.add_argument("--capital", type=float, default=INITIAL_PHP,
                        help=f"Starting PHP capital. Default: {INITIAL_PHP}")
    args = parser.parse_args()

    os.makedirs(cfg.log_dir, exist_ok=True)
    collector = DataCollector()
    pairs = [args.pair] if args.pair else cfg.pairs

    for pair in pairs:
        all_candles = collector.get_candles(pair, limit=50_000)
        if len(all_candles) <= cfg.lookback_window + 10:
            logger.warning("[%s] Not enough candles (%d). Run import_history.py first.", pair, len(all_candles))
            continue

        if args.candles:
            test_candles = all_candles[-args.candles:]
        else:
            split = int(len(all_candles) * 0.8)
            test_candles = all_candles[split:]

        logger.info("[%s] Backtesting on %d candles...", pair, len(test_candles))
        metrics = run_backtest(pair, test_candles, initial_php=args.capital)
        print_results(metrics)

    collector.close()


if __name__ == "__main__":
    main()
