import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from server.api.protocol import ChatCompletionRequest, ModelList, ModelCard
from server.adapters.ane_binding import ANEBindingAdapter
from server.core.engine import LLMEngine

from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start engine worker
    engine.start()
    yield
    # Shutdown: cancel worker task
    if engine.worker_task:
        engine.worker_task.cancel()

app = FastAPI(title="qwen-ane-llm OpenAI Compatible Server", lifespan=lifespan)

# Model configuration
MODEL_DIR = os.environ.get("ANE_MODEL_ID", "models/Qwen3.5-2B")
MODEL_NAME = os.environ.get("ANE_MODEL_NAME", os.path.basename(MODEL_DIR))

# Server configuration
HOST = os.environ.get("ANE_HOST", "127.0.0.1")
PORT = int(os.environ.get("ANE_PORT", "11222"))

# Generation defaults
DEFAULT_TEMPERATURE = float(os.environ.get("ANE_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.environ.get("ANE_MAX_TOKENS", "2048"))

# Binary path - check multiple locations
DEFAULT_BINARY_PATHS = [
    "./build/ane-lm",
    "./build/libane-lm.dylib",
    "../build/ane-lm",
    "../build/libane-lm.dylib",
]

def find_binary():
    """Find the ANE binary"""
    # Check environment
    env_path = os.environ.get("ANE_LIBRARY_PATH") or os.environ.get("ANE_BINARY_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Search default locations
    base_dir = os.path.dirname(os.path.dirname(__file__))
    for rel_path in DEFAULT_BINARY_PATHS:
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            return full_path

    return DEFAULT_BINARY_PATHS[0]

# Initialize adapter once at startup
BINARY_PATH = find_binary()
print(f"Loading model from: {MODEL_DIR}")
print(f"Using binary: {BINARY_PATH}")

adapter = ANEBindingAdapter(BINARY_PATH, MODEL_DIR)
engine = LLMEngine(adapter)

@app.get("/v1/models")
async def list_models():
    return ModelList(data=[ModelCard(id=MODEL_NAME)])

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        response = await engine.generate(request)
        if request.stream:
            return StreamingResponse(response, media_type="text/event-stream")
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "param": None,
                    "code": None
                }
            }
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
