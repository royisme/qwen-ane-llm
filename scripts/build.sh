#!/bin/bash
# qwen-ane-llm Build Script

set -e

echo "=== qwen-ane-llm Build Script ==="
echo ""

# Check prerequisites
command -v cmake >/dev/null 2>&1 || { echo "Error: cmake is required but not installed."; exit 1; }

# Build directory
BUILD_DIR="build"

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

# Configure with CMake
echo "Configuring with CMake..."
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
cmake --build "$BUILD_DIR" -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Binary location: $BUILD_DIR/ane-lm"
echo ""
echo "To run the server:"
echo "  1. Download a Qwen3.5 model"
echo "  2. Update config.yaml with model path"
echo "  3. Run: ./run.sh"
