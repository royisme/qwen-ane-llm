import json
import re
import uuid
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator, Union
from server.api.protocol import (
    ChatMessage, ChatCompletionRequest, ChatCompletionResponse, 
    ChatCompletionResponseChoice, UsageInfo, ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice, DeltaMessage
)
from server.adapters.ane_binding import ANEBindingAdapter
from server.tools import AVAILABLE_TOOLS

logger = logging.getLogger(__name__)

QWEN_SYSTEM_TOOLS = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
%s
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

@dataclass
class EngineRequest:
    messages: List[Dict[str, Any]]
    request: ChatCompletionRequest
    queue: asyncio.Queue
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

class LLMEngine:
    def __init__(self, adapter: ANEBindingAdapter):
        self.adapter = adapter
        self.request_queue = asyncio.Queue()
        self.worker_task = None

    def start(self):
        """Start the worker loop. Must be called after event loop is running."""
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker_loop())

    def format_qwen_tools(self, tools: List[Dict[str, Any]]) -> str:
        schemas = [json.dumps(t["function"], ensure_ascii=False) for t in tools if t.get("type") == "function"]
        return QWEN_SYSTEM_TOOLS % "\n".join(schemas)

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        tool_calls = []
        matches = re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match.group(1).strip())
                if "name" in data and "arguments" in data:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {"name": data["name"], "arguments": json.dumps(data["arguments"])}
                    })
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as err:
                logger.warning("Failed to parse tool_call block: %s", err)
                continue
        return tool_calls

    async def _worker_loop(self):
        """Background worker to process requests one by one using the ANE model."""
        while True:
            req: EngineRequest = await self.request_queue.get()
            try:
                # Process the request iteratively for tool calls
                current_messages = req.messages
                max_iters = 5
                
                final_result_queued = False
                last_text = ""
                last_iter_stats = {}

                for _ in range(max_iters):
                    # Call the synchronous C++ adapter
                    # To not block the whole event loop during heavy inference,
                    # we wrap the generator in an async context
                    #
                    # Inside the actual generate, we use the callback to push into req.queue
                    # but since the C library is synchronous, we'll collect the whole result
                    # for this iteration.
                    text, iter_stats = await asyncio.to_thread(
                        self.adapter.generate,
                        current_messages,
                        temperature=req.request.temperature,
                        max_tokens=req.request.max_tokens or 1024
                    )
                    last_text = text
                    last_iter_stats = iter_stats

                    tool_calls = self.parse_tool_calls(text)

                    if tool_calls and req.request.tools:
                        # Handle tool loop (don't send to user yet if we are in tool mode)
                        current_messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})
                        for tc in tool_calls:
                            name = tc["function"]["name"]
                            args = json.loads(tc["function"]["arguments"])
                            result = AVAILABLE_TOOLS[name](**args) if name in AVAILABLE_TOOLS else f"Error: Tool {name} not found."
                            current_messages.append({"role": "tool", "content": str(result), "tool_call_id": tc["id"]})
                        continue

                    # Final natural language output
                    # Put the result into the request's own queue
                    await req.queue.put((text, iter_stats))
                    final_result_queued = True
                    break

                if not final_result_queued:
                    # Fallback: if loop exhausted while still requesting tools, return last model output
                    await req.queue.put((last_text, last_iter_stats))
            except Exception as e:
                await req.queue.put(e)
            finally:
                self.request_queue.task_done()

    async def generate(self, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        
        # Tools setup
        if request.tools:
            tool_prompt = self.format_qwen_tools(request.tools)
            sys_msg = next((m for m in messages if m["role"] == "system"), None)
            if sys_msg: sys_msg["content"] = f"{sys_msg['content']}\n\n{tool_prompt}"
            else: messages.insert(0, {"role": "system", "content": tool_prompt})

        # Create a private queue for this specific request's result
        res_queue = asyncio.Queue()
        req = EngineRequest(messages=messages, request=request, queue=res_queue)
        
        # Enqueue for the global worker
        await self.request_queue.put(req)

        if not request.stream:
            # Wait for the worker to finish this request
            result = await res_queue.get()
            if isinstance(result, Exception): raise result
            text, stats = result
            return self._create_response(request, text, stats)
        else:
            # Streaming implementation
            return self._generate_stream(request, res_queue)

    async def _generate_stream(self, request: ChatCompletionRequest, res_queue: asyncio.Queue):
        """Streaming generator to be used with FastAPI StreamingResponse."""
        # Note: In this v1.5 implementation, the worker collects the whole text for an iteration
        # before pushing to the res_queue to handle Tool Loop cleanly.
        # A true per-token stream would require the C++ callback to directly communicate with res_queue.

        # For now, we simulate the stream from the collected chunk to align with protocol.
        result = await res_queue.get()
        if isinstance(result, Exception):
            yield f"data: {json.dumps({'error': str(result)})}\n\n"
            return

        text, stats = result
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        tool_calls = self.parse_tool_calls(text)
        finish_reason = "tool_calls" if tool_calls else "stop"

        # Split into words/chunks for simulated streaming feel if needed,
        # or just send the whole block as one chunk for now.
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            model=request.model,
            choices=[ChatCompletionStreamResponseChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=text, tool_calls=tool_calls or None),
                finish_reason=finish_reason
            )]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    def _create_response(self, request: ChatCompletionRequest, text: str, stats: Dict) -> ChatCompletionResponse:
        usage = UsageInfo(
            prompt_tokens=stats.get("prompt_tokens", 0),
            completion_tokens=stats.get("generation_tokens", 0),
            total_tokens=stats.get("prompt_tokens", 0) + stats.get("generation_tokens", 0)
        )
        tool_calls = self.parse_tool_calls(text)
        finish_reason = "tool_calls" if tool_calls else "stop"
        return ChatCompletionResponse(
            model=request.model,
            choices=[ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text, tool_calls=tool_calls or None),
                finish_reason=finish_reason
            )],
            usage=usage
        )
