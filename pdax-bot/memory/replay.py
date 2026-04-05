"""
Experience replay buffer.

Stores completed trade experiences so the agent can learn from them
beyond what PPO's own rollout buffer captures.

Each experience:
  {
    "pair":        str,
    "timestamp":   float,
    "obs":         np.ndarray,   # state at decision time
    "action":      int,
    "reward":      float,        # PHP PnL of the trade
    "next_obs":    np.ndarray,   # state after trade settled
    "done":        bool,
    "info":        dict          # price, qty, side, etc.
  }

Persisted to a JSON-lines file so learning survives restarts.
"""

import json
import logging
import os
import threading
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from config import cfg

logger = logging.getLogger(__name__)

REPLAY_FILE = "logs/replay_buffer.jsonl"


class ReplayBuffer:
    """
    Fixed-size circular buffer with disk persistence.
    Thread-safe for concurrent push/sample.
    """

    def __init__(self, maxlen: Optional[int] = None, path: Optional[str] = None):
        self._maxlen = maxlen or cfg.replay_buffer_size
        self._path   = path or REPLAY_FILE
        self._lock   = threading.Lock()
        self._buffer: deque = deque(maxlen=self._maxlen)
        self._load()

    # ── Write ──────────────────────────────────────────────────────────────────

    def push(self, experience: Dict) -> None:
        """Add one experience to the buffer and append it to disk."""
        # Convert numpy arrays to lists for JSON serialisation
        exp = _serialise(experience)
        with self._lock:
            self._buffer.append(exp)
            self._append_to_disk(exp)

    def _append_to_disk(self, exp: Dict) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(exp) + "\n")
        except OSError as e:
            logger.warning("Replay buffer write error: %s", e)

    # ── Read ───────────────────────────────────────────────────────────────────

    def sample(self, n: int) -> List[Dict]:
        """Return n random experiences (with numpy arrays restored)."""
        with self._lock:
            buf = list(self._buffer)
        if not buf:
            return []
        indices = np.random.choice(len(buf), size=min(n, len(buf)), replace=False)
        return [_deserialise(buf[i]) for i in indices]

    def recent(self, n: int) -> List[Dict]:
        """Return the n most recent experiences."""
        with self._lock:
            buf = list(self._buffer)
        return [_deserialise(e) for e in buf[-n:]]

    def __len__(self) -> int:
        return len(self._buffer)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Reload buffer from disk on startup."""
        if not os.path.exists(self._path):
            return
        loaded = 0
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._buffer.append(json.loads(line))
                        loaded += 1
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Replay buffer load error: %s", e)
        logger.info("ReplayBuffer: loaded %d experiences from %s", loaded, self._path)


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _serialise(exp: Dict) -> Dict:
    out = {}
    for k, v in exp.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _deserialise(exp: Dict) -> Dict:
    out = dict(exp)
    for key in ("obs", "next_obs"):
        if key in out and isinstance(out[key], list):
            out[key] = np.array(out[key], dtype=np.float32)
    return out
