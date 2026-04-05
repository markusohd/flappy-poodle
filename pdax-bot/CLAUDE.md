# PDAX Trading Bot

Self-learning RL trading bot for the Philippine Digital Asset Exchange (PDAX).
Uses PPO (Proximal Policy Optimization) that fine-tunes itself after every 50 trades.

## Project layout

```
pdax-bot/
├── config.py              # All settings — edit here or use env vars
├── train.py               # Train the agent on historical data
├── run.py                 # Run the live bot (dry-run by default)
├── exchange/
│   ├── client.py          # PDAX REST API (HMAC-SHA384 auth, retry logic)
│   └── feed.py            # Polls ticker + order book every 5s
├── data/
│   ├── collector.py       # Stores ticks → SQLite, builds 1-min candles
│   └── features.py        # RSI, MACD, EMA, Bollinger Bands → state vector
├── env/
│   └── trading_env.py     # Gymnasium env: 7 actions (BUY/SELL/HOLD × 3 sizes)
├── agent/
│   └── policy.py          # PPO wrapper: train / finetune / predict / save / load
├── memory/
│   └── replay.py          # Experience replay buffer (disk-persisted)
├── monitor/
│   └── tracker.py         # Trade log, P&L, drawdown, fine-tune trigger
└── scripts/
    ├── import_history.py  # Seed DB with 90 days of CoinGecko OHLCV → PHP
    ├── collect.py         # Live data collection only (no trading)
    └── backtest.py        # Evaluate trained agent, report Sharpe/drawdown/alpha
```

## Setup (first time)

```bash
cd pdax-bot
bash setup.sh
cp .env.example .env
# Fill in PDAX_API_KEY and PDAX_SECRET_KEY in .env
# (email institutions@pdax.ph to get API credentials)
```

## Typical workflow

```bash
source .venv/bin/activate

# 1. Seed the database with historical data (no API key needed)
python scripts/import_history.py

# 2. Train the agent
python train.py

# 3. Backtest before going anywhere near real money
python scripts/backtest.py

# 4. Collect live PDAX data (run for a few hours to enrich training)
python scripts/collect.py

# 5. Retrain with live data mixed in
python train.py

# 6. Dry-run (sandbox, no real orders)
python run.py

# 7. Go live on PDAX sandbox
PDAX_SANDBOX=true python run.py --live

# 8. Go live with real money (only after thorough testing)
PDAX_SANDBOX=false python run.py --live
```

## Key environment variables

| Variable | Default | Description |
|---|---|---|
| `PDAX_API_KEY` | — | Your PDAX API key |
| `PDAX_SECRET_KEY` | — | Your PDAX secret key |
| `PDAX_SANDBOX` | `true` | `false` = real money |
| `PDAX_DB_PATH` | `logs/market_data.db` | SQLite database path |

## How the learning loop works

```
Live tick (every 5s)
  → DataCollector (SQLite)
  → build_state() → 24-dim observation vector
  → PPO agent.predict() → action (BUY / SELL / HOLD)
  → PDAXClient.place_market_order()
  → TradeTracker.record() → ReplayBuffer.push()
  → every 50 trades → agent.finetune() (5,000 timesteps)
  → model saved to models/ppo_{PAIR}.zip
```

## Action space

| Index | Action | Size |
|---|---|---|
| 0 | HOLD | — |
| 1 | BUY | 10% of PHP balance |
| 2 | BUY | 25% of PHP balance |
| 3 | BUY | 50% of PHP balance |
| 4 | SELL | 10% of holdings |
| 5 | SELL | 50% of holdings |
| 6 | SELL ALL | 100% of holdings |

## Important notes

- **PDAX API is institutional-only** — you must email `institutions@pdax.ph`
- **Sandbox first** — always test on sandbox before enabling `PDAX_SANDBOX=false`
- **No OHLCV endpoint** on PDAX — the bot builds its own candles from tick polls
- **No WebSocket** — all data comes from REST polling (5s interval)
- Models are saved to `models/ppo_{PAIR}.zip` and auto-loaded on restart
- The replay buffer survives restarts via `logs/replay_buffer.jsonl`
