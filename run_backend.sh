#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

export PYTHONPATH="$DIR"
echo "🚀 Starting Infographix Backend on http://0.0.0.0:8000..."
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

