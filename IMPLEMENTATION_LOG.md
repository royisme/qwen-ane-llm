# qwen-ane-llm OpenAI 兼容封装 — 重构实现日志 (v2)

本日志记录了对 **royisme/qwen-ane-llm** 进行深度重构的过程，目标是实现一个工业级的 OpenAI 兼容服务，支持高性能推理、Tool Calling 闭环、流式输出及多用户并发调度。

---

## 1. 核心架构设计

我们采用了 **"C++ 算子层 + Python 编排层"** 的解耦架构，借鉴了 vLLM 和 llama.cpp 的设计理念。

### 1.1 C++ 层：动态库化 (The Shared Library)
- **目标**：解决模型加载慢的问题。
- **实现**：将 qwen-ane-llm 核心代码重构为 `libane-lm.dylib`。
- **接口**：定义了纯 C 接口 `include/ane_lm/ane_lm_c.h`，支持模型一次加载、多次推理。
- **回调机制**：引入 `ane_callback_t`，支持在推理过程中实时获取 Token。

### 1.2 Python 层：异步引擎 (The LLMEngine)
- **协议层**：参照 vLLM 实现了 `api/protocol.py`，完全对齐 OpenAI API 规范。
- **调度层**：引入 `asyncio.Queue` 实现全局请求队列。由于 ANE 硬件限制，Engine 采用单线程 Worker 模式串行处理推理任务，确保状态安全。
- **编排层**：`core/engine.py` 负责 Tool Loop（工具循环）。它会自动注入 Qwen 官方 XML 格式的工具描述，并解析模型输出的 `<tool_call>` 标签。

---

## 2. 关键功能实现

### 2.1 Tool Calling 闭环 (Route B)
- **意图识别**：不再依赖简单的 JSON 强制输出，而是使用 Qwen 3.5 训练时的 XML 模板。
- **自动执行**：Engine 拦截模型输出，执行本地 Python 函数（如 `run_date`），并将结果通过 `<tool_response>` 标签回填给模型。
- **多轮对话**：支持模型在单次请求中多次调用工具直至给出最终自然语言回答。

### 2.2 流式输出 (Streaming)
- **协议兼容**：支持 `stream=true` 参数，返回标准的 `text/event-stream` (SSE)。
- **Chunk 封装**：输出格式与 OpenAI `chat.completion.chunk` 完全一致。

### 2.3 并发调度 (Batching/Queueing)
- **FIFO 队列**：所有并发请求进入队列排队，由后台 Worker 异步处理。
- **非阻塞 API**：推理在 `asyncio.to_thread` 中运行，确保 FastAPI 能够同时处理多个 API 请求而不阻塞。

### 2.4 KV Cache 优化 (Session Persistence)
- **局部重置 (Partial Reset)**：修改了 `LLMModel::reset(n_keep)`，允许模型保留前 `n_keep` 个 Token 的 KV 缓存状态。
- **前缀匹配 (Prefix Matching)**：在 `stream_generate` 中引入了静态 Token 追踪。系统会自动计算当前 Prompt 与上次推理序列的公共前缀长度 `n_past`，并仅对 `n_past` 之后的 Token 进行 Prefill。
- **性能增益**：在多轮对话或复杂的工具调用循环中，Context 的重复计算开销降至零，显著提升了系统的响应速度。

---

## 3. 关键文件变更清单

| 文件路径 | 变更说明 |
| :--- | :--- |
| `CMakeLists.txt` | 增加 `libane-lm` (SHARED) 目标，配置 Framework 链接。 |
| `main.cpp` | 增加 `--json-messages` 参数支持，将统计信息 JSON 化输出。 |
| `include/ane_lm/ane_lm_c.h` | 定义 C API 接口。 |
| `core/ane_lm_c.cpp` | 实现 C API 桥接逻辑。 |
| `server/api/protocol.py` | 实现 OpenAI 兼容的 Pydantic 模型。 |
| `server/core/engine.py` | **核心重写**：异步调度、Tool Loop 编排、XML 模板注入。 |
| `server/adapters/ane_binding.py` | 使用 `ctypes` 实现对动态库的持久化加载与调用。 |
| `server/main.py` | FastAPI 路由分发与流式响应处理。 |
| `server/tools.py` | 定义内置工具（`run_date`, `run_shell`）。 |

---

## 4. 运行与验证

### 编译
```bash
mkdir -p build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### 启动服务
```bash
export ANE_LIBRARY_PATH="./build/libane-lm.dylib"
export ANE_MODEL_ID="models/llm/qwen3_5-0.8B"
uv run python -m server --port 11222
```

### 测试用例
**1. 基础对话 (非流式)**
```bash
curl -sS http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"你好"}]}' | jq
```

**2. 工具调用 (核心功能)**
```bash
curl -sS http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen",
    "messages":[{"role":"user","content":"现在几点了？使用工具回答。"}],
    "tools":[{
      "type":"function",
      "function":{
        "name":"run_date",
        "description":"获取系统当前时间"
      }
    }]
  }' | jq
```

**3. 流式输出**
```bash
curl -i http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"写一段长文本"}],"stream":true}'
```

---

## 5. 后续计划
- **KV Cache 优化**：在 C 接口中支持 Context 缓存，提升多轮对话性能。
- **Continuous Batching**：在 Engine 层实现更细粒度的 Token 级请求合并。
- **多模型支持**：支持动态切换不同的 ANE 权重文件。
