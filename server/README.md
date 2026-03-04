# ANE-LM OpenAI Server

This is an OpenAI-compatible API wrapper for **ANE-LM**, implemented using FastAPI.

## Features
- `GET /v1/models`
- `POST /v1/chat/completions` (Non-streaming)
- **Tool Calling**: Minimal closed-loop implementation using Route A (Strict JSON output).
- Built-in tools: `run_date`, `run_shell` (allowlisted).

## Prerequisites
- Python 3.9+
- ANE-LM binary (built in `build/ane-lm`)
- ANE-LM compatible model (e.g., Qwen3.5-0.8B)

## Installation
Navigate to the root directory and ensure you have `uv` installed, or use `pip`:

```bash
# Using uv (recommended)
uv sync --project server

# Or using pip
pip install -e server/
```

## Running the Server
Set the environment variables for your model and binary path:

```bash
export ANE_BINARY_PATH="./build/ane-lm"
export ANE_MODEL_ID="models/llm/qwen3_5-0.8B" # Update to your actual model path

uv run python -m server --host 127.0.0.1 --port 11222
```

## API Usage Examples

### 1. List Models
```bash
curl -sS http://127.0.0.1:11222/v1/models | jq
```

### 2. Basic Chat
```bash
curl -sS http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen3_5-0.8B",
    "messages":[{"role":"user","content":"你好，回复 OK"}],
    "temperature":0.1
  }' | jq
```

### 3. Tool Calling (Get Current Time)
```bash
curl -sS http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen3_5-0.8B",
    "messages":[{"role":"user","content":"现在几点了？你必须调用 run_date 工具后再回答。"}],
    "tools":[{
      "type":"function",
      "function":{
        "name":"run_date",
        "description":"Return current system time",
        "parameters":{"type":"object","properties":{},"required":[]}
      }
    }],
    "tool_choice":"auto",
    "temperature":0
  }' | jq
```

## Security
- `run_shell` is limited to an allowlist: `date`, `uname -a`, `uptime`.
- The server listens on `127.0.0.1` by default.
