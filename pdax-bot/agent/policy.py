"""
PPO agent wrapper.

Uses stable-baselines3's PPO with a custom MlpPolicy.
Handles:
  - Initial training on historical candles
  - Continuous fine-tuning after every N trades (online learning)
  - Model persistence (save/load checkpoints)
"""

import logging
import os
from typing import List, Dict, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from config import cfg
from env.trading_env import TradingEnv

logger = logging.getLogger(__name__)


class TradingAgent:
    """
    Wraps a stable-baselines3 PPO model with helpers for:
      - training from scratch on historical data
      - fine-tuning (continuous learning) on new experiences
      - action prediction for live trading
    """

    def __init__(self, pair: str = "BTCPHP"):
        self.pair      = pair
        self.model: Optional[PPO] = None
        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.log_dir,   exist_ok=True)

    # ── Model path helpers ─────────────────────────────────────────────────────

    def _model_path(self) -> str:
        return os.path.join(cfg.model_dir, f"ppo_{self.pair}")

    def _checkpoint_path(self) -> str:
        return os.path.join(cfg.model_dir, "checkpoints", self.pair)

    # ── Build ──────────────────────────────────────────────────────────────────

    def _make_env(self, candles: List[Dict]) -> TradingEnv:
        return TradingEnv(candles, pair=self.pair, lookback=cfg.lookback_window)

    def _build_model(self, env: TradingEnv) -> PPO:
        return PPO(
            policy        = "MlpPolicy",
            env           = env,
            learning_rate = cfg.learning_rate,
            n_steps       = cfg.n_steps,
            batch_size    = cfg.batch_size,
            n_epochs      = cfg.n_epochs,
            gamma         = cfg.gamma,
            gae_lambda    = cfg.gae_lambda,
            clip_range    = cfg.clip_range,
            ent_coef      = cfg.ent_coef,
            verbose       = 1,
            tensorboard_log = cfg.log_dir,
        )

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, candles: List[Dict], total_timesteps: int = 200_000) -> None:
        """
        Train from scratch (or fine-tune if a model already exists) on
        historical candles.

        Splits candles 80/20: train / eval.
        """
        split = int(len(candles) * 0.8)
        train_candles = candles[:split]
        eval_candles  = candles[split:]

        if len(train_candles) <= cfg.lookback_window:
            logger.warning("Not enough candles to train (%d). Need > %d.", len(train_candles), cfg.lookback_window)
            return

        train_env = self._make_env(train_candles)

        if self.model is None:
            logger.info("Building new PPO model for %s", self.pair)
            self.model = self._build_model(train_env)
        else:
            # Keep existing weights, just update the environment
            self.model.set_env(DummyVecEnv([lambda: train_env]))

        callbacks = []

        # Periodic checkpoint
        os.makedirs(self._checkpoint_path(), exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, total_timesteps // 10),
                save_path=self._checkpoint_path(),
                name_prefix="ppo",
            )
        )

        # Eval callback (if we have enough eval data)
        if len(eval_candles) > cfg.lookback_window:
            eval_env = DummyVecEnv([lambda: self._make_env(eval_candles)])
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=self._checkpoint_path(),
                    log_path=cfg.log_dir,
                    eval_freq=max(1, total_timesteps // 20),
                    deterministic=True,
                    render=False,
                )
            )

        logger.info("Training %s for %d timesteps...", self.pair, total_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,  # continue from last step count
            tb_log_name=f"PPO_{self.pair}",
        )
        self.save()
        logger.info("Training complete. Model saved.")

    def finetune(self, candles: List[Dict], timesteps: int = 10_000) -> None:
        """
        Lightweight fine-tune on recent candles — called after every N trades.
        Keeps existing weights; just runs a few more rollouts.
        """
        if self.model is None:
            logger.warning("No model to fine-tune. Call train() first.")
            return
        if len(candles) <= cfg.lookback_window:
            logger.warning("Not enough candles to fine-tune.")
            return

        env = self._make_env(candles)
        self.model.set_env(DummyVecEnv([lambda: env]))
        logger.info("Fine-tuning %s for %d timesteps...", self.pair, timesteps)
        self.model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"PPO_{self.pair}_finetune",
        )
        self.save()
        logger.info("Fine-tune complete.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Given an observation vector, return the action index.
        deterministic=True during live trading for more stable behaviour;
        deterministic=False during training for exploration.
        """
        if self.model is None:
            raise RuntimeError("Model not initialised. Call train() or load() first.")
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        if self.model:
            self.model.save(self._model_path())
            logger.debug("Model saved: %s", self._model_path())

    def load(self) -> bool:
        """Load a previously saved model. Returns True if successful."""
        path = self._model_path() + ".zip"
        if not os.path.exists(path):
            logger.info("No saved model found at %s", path)
            return False
        self.model = PPO.load(path)
        logger.info("Model loaded from %s", path)
        return True
