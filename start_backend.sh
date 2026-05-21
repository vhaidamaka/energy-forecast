#!/bin/bash
# Start the Energy Predictor backend
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo "⚡ Energy Predictor — Backend"
echo ""

# Create venv if not exists
if [ ! -d "$BACKEND_DIR/.venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$BACKEND_DIR/.venv"
fi

source "$BACKEND_DIR/.venv/bin/activate"

echo "Installing dependencies..."
pip install -q -r "$BACKEND_DIR/requirements.txt"

echo ""
echo "Starting FastAPI server on http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""

cd "$BACKEND_DIR"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
