# qwen-ane-llm OpenAI API Server

OpenAI-compatible REST API server for qwen-ane-llm, built with FastAPI.

## Overview

This server provides a drop-in replacement for OpenAI's Chat Completions API, enabling:

- Standard OpenAI client compatibility
- Real-time streaming responses via Server-Sent Events (SSE)
- Function calling / tool use for interactive AI agents
- Built-in safe shell command execution

## Features

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion (streaming supported) |

### Supported Capabilities

- **Streaming**: Token-by-token streaming with `stream: true`
- **Tool Calling**: Function calling with automatic tool execution loop
- **System Prompt**: System message support for persona/instruction setting

## Installation

### Prerequisites

- Python 3.9+
- Built `ane-lm` binary (see main README)
- Qwen3.5 model in safetensors format

### Setup

```bash
# Using uv (recommended)
cd qwen-ane-llm
uv sync --project server

# Or using pip
pip install -e server/
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANE_BINARY_PATH` | Yes | Path to compiled `ane-lm` binary |
| `ANE_MODEL_ID` | Yes | Path to model directory |

### Example

```bash
export ANE_BINARY_PATH="./build/ane-lm"
export ANE_MODEL_ID="./models/Qwen3.5-0.8B"

# Start server
uv run python -m server --host 127.0.0.1 --port 11222
```

## API Usage

### 1. List Models

```bash
curl -sS http://127.0.0.1:11222/v1/models | jq
```

### 2. Simple Chat

```bash
curl -sS http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-0.8B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Apple Neural Engine?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }' | jq
```

### 3. Streaming Response

```bash
curl -sS http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-0.8B",
    "messages": [{"role": "user", "content": "Tell me a short story"}],
    "stream": true
  }'
```

### 4. Tool Calling

```bash
curl -sS http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-0.8B",
    "messages": [{"role": "user", "content": "What is the current system time?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "run_date",
        "description": "Get current system date and time",
        "parameters": {"type": "object", "properties": {}}
      }
    }],
    "tool_choice": "auto",
    "temperature": 0
  }' | jq
```

### 5. Shell Command (Allowlisted)

```bash
curl -sS http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-0.8B",
    "messages": [{"role": "user", "content": "Run `uname -a` for me"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "run_shell",
        "description": "Execute a shell command from allowlist",
        "parameters": {
          "type": "object",
          "properties": {
            "cmd": {"type": "string", "description": "Command to execute"}
          },
          "required": ["cmd"]
        }
      }
    }],
    "tool_choice": "auto"
  }' | jq
```

## Built-in Tools

| Tool | Description |
|------|-------------|
| `run_date` | Returns current system date and time |
| `run_shell` | Executes allowlisted shell commands |

### Shell Allowlist

For security, `run_shell` only permits these commands:

- `date`
- `uname -a`
- `uptime`

## Error Handling

Standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request format
- `500 Internal Server Error`: Engine execution failure

## Security

- Server binds to `127.0.0.1` by default (localhost only)
- Shell commands are strictly allowlisted
- No external network calls — fully offline

## Performance Tips

1. **Pre-convert weights**: Run `./build/ane-lm convert --model /path/to/model` to convert BF16→FP16 for faster loading
2. **Disable ANE cache**: Use `--no-ane-cache` flag if experiencing stale compilation issues
3. **Enable thinking mode**: Use `--enable-thinking` for reasoning-heavy tasks

## License

MIT License — see project root LICENSE file.
