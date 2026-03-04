#!/bin/bash
# qwen-ane-llm Server Install Script

set -e

echo "=== qwen-ane-llm Server Install Script ==="
echo ""

# Check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env 2>/dev/null || true
fi

# Navigate to server directory
cd "$(dirname "$0")/.."

echo "Installing server dependencies..."
uv sync --project server

echo ""
echo "=== Install Complete ==="
echo ""
echo "To start the server:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  cd server"
echo "  uv run python -m server"
