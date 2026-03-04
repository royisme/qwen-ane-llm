# qwen-ane-llm

High-performance LLM inference engine running Qwen3.5 on Apple Neural Engine (ANE), with OpenAI-compatible API for seamless integration.

## Overview

qwen-ane-llm leverages Apple's private `AppleNeuralEngine.framework` to run large language model inference directly on Apple Silicon. Unlike cloud-based solutions, all inference happens locally on your device — ensuring privacy, low latency, and zero API costs.

### Key Features

- **Native ANE Acceleration**: Utilizes Apple's Neural Engine for efficient LLM inference
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API with minimal code changes
- **Streaming Support**: Real-time token-by-token streaming via Server-Sent Events
- **Tool Calling**: Built-in function calling support for interactive AI agents
- **Zero Cloud Dependency**: 100% offline inference — your data never leaves the device

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    qwen-ane-llm                        │
├─────────────────────────────────────────────────────────┤
│  Python Server (FastAPI)                               │
│  ├── OpenAI-compatible REST API                        │
│  ├── Tool calling orchestration                        │
│  └── Response streaming (SSE)                          │
├─────────────────────────────────────────────────────────┤
│  C++ Core                                              │
│  ├── ANE Runtime (private framework)                  │
│  ├── Qwen3.5 Model Implementation                      │
│  ├── KV Cache Management                               │
│  └── Sampling & Tokenization                           │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Build the Engine

```bash
# Clone and build
git clone https://github.com/royisme/qwen-ane-llm.git
cd qwen-ane-llm

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 2. Download Model

Download Qwen3.5-0.8B in safetensors format and place it in your project directory.

### 3. Run the Server

```bash
# Set environment variables
export ANE_BINARY_PATH="./build/ane-lm"
export ANE_MODEL_ID="./path/to/Qwen3.5-0.8B"

# Start the API server
cd server
uv sync
uv run python -m server --host 127.0.0.1 --port 11222
```

### 4. Make Your First Request

```bash
curl -sS http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-0.8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }' | jq
```

## API Reference

### List Available Models

```bash
GET /v1/models
```

### Chat Completions

```bash
POST /v1/chat/completions
```

**Request Body:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model ID (e.g., `qwen3_5-0.8B`) |
| `messages` | array | Conversation history |
| `temperature` | float | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | Maximum tokens to generate |
| `stream` | boolean | Enable streaming response |
| `tools` | array | Available functions for tool calling |

### Tool Calling Example

```bash
curl -sS http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-0.8B",
    "messages": [{"role": "user", "content": "What time is it now?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "run_date",
        "description": "Get current system time",
        "parameters": {"type": "object", "properties": {}}
      }
    }],
    "tool_choice": "auto"
  }'
```

## Performance

Running Qwen3.5-0.8B on Apple Neural Engine delivers:

- **~20 tokens/sec** generation speed on M1/M2/M3
- **Zero** network latency — all inference local
- **Privacy-first**: No data leaves your machine

## Requirements

- **macOS** 13.0+ (Ventura or later)
- **Apple Silicon** (M1, M2, M3, M4, M5)
- **Python** 3.9+ (for API server)

## Use Cases

- **Privacy-sensitive applications**: Medical, legal, financial AI assistants
- **Offline AI agents**: Local-first autonomous agents
- **Development & prototyping**: Fast iteration without API costs
- **Edge deployment**: Embeddable AI on Apple devices

## Tech Stack

- **C++17**: Core inference engine
- **FastAPI**: Python API server
- **Apple Neural Engine**: Hardware-accelerated inference
- **Qwen3.5**: Language model

## Roadmap

- [ ] Support more Qwen variants
- [ ] Multi-turn conversation optimization
- [ ] GPU fallback for non-ANE Macs
- [ ] Quantization support (Q4, Q8)

## License

MIT License — see LICENSE file.

## Acknowledgments

- [maderix/ANE](https://github.com/maderix/ANE) — Reverse-engineered ANE framework
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — LLM inference patterns
- [Qwen](https://github.com/QwenLM/Qwen) — Language model weights

---

**Star us on GitHub** if you find this project useful!
