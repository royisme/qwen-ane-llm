import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from server.api.protocol import ChatCompletionRequest, ModelList, ModelCard
from server.adapters.ane_binding import ANEBindingAdapter
from server.core.engine import LLMEngine

app = FastAPI(title="qwen-ane-llm OpenAI Compatible Server")

# Initialize Engine with Shared Library Binding
BINARY_PATH = os.environ.get("ANE_LIBRARY_PATH", "./build/libane-lm.dylib")
MODEL_DIR = os.environ.get("ANE_MODEL_ID", "models/llm/qwen3_5-0.8B")
MODEL_NAME = os.path.basename(MODEL_DIR)

# Initialize adapter once at startup
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11222)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
