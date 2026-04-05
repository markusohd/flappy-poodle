#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# PDAX Trading Bot — one-time setup script
# Run: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "==> Checking Python version..."
python3 --version

echo "==> Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "==> Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Creating directories..."
mkdir -p models logs models/checkpoints

echo "==> Setting up environment file..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    Created .env — please fill in your PDAX_API_KEY and PDAX_SECRET_KEY"
else
  echo "    .env already exists, skipping."
fi

echo ""
echo "✓ Setup complete."
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your PDAX API credentials"
echo "  2. Import historical data:"
echo "       source .venv/bin/activate"
echo "       python scripts/import_history.py"
echo "  3. Train the agent:"
echo "       python train.py"
echo "  4. Run a backtest:"
echo "       python scripts/backtest.py"
echo "  5. Run the bot (dry-run by default):"
echo "       python run.py"
echo "  6. Go live (only after testing!):"
echo "       PDAX_SANDBOX=false python run.py --live"
