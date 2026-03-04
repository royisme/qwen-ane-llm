#!/bin/bash
# qwen-ane-llm Run Script

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

cd "$PROJECT_DIR"

# Copy .env.example to .env if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
    fi
fi

# Source .env file
export $(cat .env | grep -v '^#' | xargs)

# Set binary path to the dylib (required for ANE inference)
export ANE_BINARY_PATH="${ANE_BINARY_PATH:-$PROJECT_DIR/build/libane-lm.dylib}"

echo "Starting qwen-ane-llm server..."
echo "Model: $ANE_MODEL_ID"
echo "Binary: $ANE_BINARY_PATH"

# Check if binary exists
if [ ! -f "$ANE_BINARY_PATH" ]; then
    echo "Error: Binary not found at $ANE_BINARY_PATH"
    echo "Run ./scripts/build.sh first"
    exit 1
fi

# Run server directly
cd "$PROJECT_DIR/server"
exec .venv/bin/python main.py
