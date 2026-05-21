#!/bin/bash
# Start the Energy Predictor frontend
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "⚡ Energy Predictor — Frontend"
echo ""

cd "$FRONTEND_DIR"

if [ ! -d "node_modules" ]; then
  echo "Installing npm dependencies..."
  npm install
fi

echo "Starting React dev server on http://localhost:5173"
echo ""

npm run dev
