# qwen-ane-llm

High-performance LLM inference engine running Qwen3.5 on Apple Neural Engine (ANE), with OpenAI-compatible REST API.

## Overview

qwen-ane-llm leverages Apple's private `AppleNeuralEngine.framework` to run large language model inference directly on Apple Silicon. All inference happens locally on your device — ensuring privacy, low latency, and zero API costs.

## Features

- **Native ANE Acceleration**: Hardware-accelerated LLM inference on Apple Silicon
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI Chat Completions API
- **Streaming Support**: Real-time token-by-token streaming via Server-Sent Events
- **Tool Calling**: Built-in function calling support for interactive AI agents
- **100% Offline**: No cloud dependencies — your data never leaves the device

## Quick Start

### 1. Build

```bash
./scripts/build.sh
```

### 2. Configure

Copy `.env.example` to `.env` and set your model path:

```bash
cp .env.example .env
# Edit .env with your model path
```

### 3. Install Server Dependencies

```bash
./scripts/install.sh
```

### 4. Run

```bash
./run.sh
```

The server will start at `http://127.0.0.1:11222`

## API Usage

### List Models

```bash
curl http://127.0.0.1:11222/v1/models
```

### Chat Completion

```bash
curl -X POST http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-2B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming

```bash
curl -X POST http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-2B",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

## Configuration

Edit `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANE_MODEL_ID` | Path to model directory | `./models/Qwen3.5-2B` |
| `ANE_MODEL_NAME` | Model name in API responses | (basename of model path) |
| `ANE_BINARY_PATH` | Path to ANE library | `./build/libane-lm.dylib` |
| `ANE_HOST` | Server host | `127.0.0.1` |
| `ANE_PORT` | Server port | `11222` |
| `ANE_TEMPERATURE` | Default temperature | `0.7` |
| `ANE_MAX_TOKENS` | Default max tokens | `2048` |

## Requirements

- **macOS** 13.0+ (Ventura or later)
- **Apple Silicon** (M1/M2/M3/M4/M5)
- **Python** 3.9+ (for API server)

## Architecture

```
┌─────────────────────────────────────────┐
│         qwen-ane-llm                   │
├─────────────────────────────────────────┤
│  Python Server (FastAPI)               │
│  ├── OpenAI-compatible REST API        │
│  ├── Tool calling orchestration        │
│  └── Response streaming (SSE)          │
├─────────────────────────────────────────┤
│  C++ Core                              │
│  ├── ANE Runtime (private framework)   │
│  ├── Qwen3.5 Model Implementation      │
│  ├── KV Cache Management               │
│  └── Sampling & Tokenization           │
└─────────────────────────────────────────┘
```

## License

MIT License
