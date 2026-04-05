"""
Central configuration for the PDAX trading bot.
All values can be overridden via environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── API credentials ────────────────────────────────────────────────────────
    api_key: str = field(default_factory=lambda: os.getenv("PDAX_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("PDAX_SECRET_KEY", ""))

    # ── Endpoints ──────────────────────────────────────────────────────────────
    prod_url: str = "https://services.pdax.ph/api/exchange/v1"
    sandbox_url: str = "https://services-stage.pdax.ph/api/exchange/v1"
    # Set PDAX_SANDBOX=false to go live
    use_sandbox: bool = field(
        default_factory=lambda: os.getenv("PDAX_SANDBOX", "true").lower() != "false"
    )

    @property
    def base_url(self) -> str:
        return self.sandbox_url if self.use_sandbox else self.prod_url

    # ── Trading pairs ──────────────────────────────────────────────────────────
    # All available PDAX pairs (PHP-quoted). Add/remove as needed.
    pairs: List[str] = field(default_factory=lambda: [
        "BTCPHP", "ETHPHP", "XRPPHP", "SOLPHP", "BNBPHP",
        "ADAPHP", "AVAXPHP", "DOTPHP", "DOGEPHP", "LTCPHP",
        "BCHPHP", "TRXPHP", "LINKPHP", "USDTPHP", "USDCPHP",
    ])

    # ── Risk / position sizing (aggressive mode) ───────────────────────────────
    # Max fraction of total balance to commit per trade (0.0–1.0)
    max_position_fraction: float = 0.5
    # Minimum order size in PHP
    min_order_php: float = 100.0
    # Trading fee rate (PDAX ~0.1% maker, ~0.2% taker – use taker to be safe)
    fee_rate: float = 0.002

    # ── Data / feature engineering ─────────────────────────────────────────────
    # Number of historical ticks included in the RL state vector
    lookback_window: int = 60
    # Seconds between REST poll cycles
    poll_interval_sec: float = 5.0
    # SQLite database path
    db_path: str = os.getenv("PDAX_DB_PATH", "logs/market_data.db")

    # ── RL hyperparameters (PPO) ───────────────────────────────────────────────
    learning_rate: float = 3e-4
    n_steps: int = 2048       # steps per PPO rollout
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99       # discount factor
    gae_lambda: float = 0.95  # GAE smoothing
    clip_range: float = 0.2
    ent_coef: float = 0.01    # entropy bonus – encourages exploration

    # ── Continuous learning ────────────────────────────────────────────────────
    # Fine-tune the model after every N completed trades
    retrain_every_n_trades: int = 50
    # Keep up to this many past experiences in the replay buffer
    replay_buffer_size: int = 100_000

    # ── Paths ──────────────────────────────────────────────────────────────────
    model_dir: str = "models"
    log_dir: str = "logs"


# Singleton – import this everywhere
cfg = Config()
