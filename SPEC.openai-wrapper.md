# ANE-LM OpenAI 兼容封装 — SPEC（非流式 v0）

目标：在 **royisme/ANE-LM** 上实现一个“可被 OpenAI SDK / curl / litellm 调用”的本地 HTTP 服务，提供 **OpenAI API 兼容（subset）**，并实现 **最小 tool/function call 闭环**。

> 结论先行：当前你看到的“模型说它不能调用外部工具”并不是模型本身的问题，而是**服务层缺少 tool schema 传递 + tool call 解析/执行 + tool loop**。

---

## 0. 范围与非目标

### In-scope（必须）
1. OpenAI-compatible endpoint（subset）：
   - `GET /v1/models`
   - `POST /v1/chat/completions`
2. Non-streaming responses（**不做 SSE/流式**）
3. 支持 **tools/function calling（最小闭环）**：
   - 服务端接收 `tools` 定义
   - 把 tools 注入 prompt（或 model 可理解的格式）
   - 解析模型输出中的工具调用意图（严格产出 OpenAI `tool_calls`）
   - 执行本地工具（首版只内置一个 `run_shell` 或 `run_date`）
   - 将 tool result 作为 `role=tool` message 再次喂给模型，得到最终自然语言回答
4. 端口固定：`11222`

### Out-of-scope（先不做）
- `stream=true` 流式输出（SSE chunk）
- `POST /v1/completions`（旧接口）
- embeddings / images / audio
- 多用户鉴权、速率限制、配额
- 复杂工具沙盒（首版只做 allowlist）

---

## 1. 参考实现（抄作业清单）

实现行为优先对齐：
- **vLLM OpenAI server**：作为“OpenAI 兼容行为”的对标（request/response、错误码、字段名、tool_calls 结构）
- **llama.cpp server**：作为“最小可用实现”的对标（实现短、易读，方便快速闭环）

> 注：首版不要求完全一致，但字段命名与 JSON 形状要兼容 OpenAI SDK。

---

## 2. 服务形态与启动方式

### 推荐结构（可在 repo 内新增）
- `server/`：OpenAI 兼容 HTTP 服务（Python FastAPI 或 Node/Fastify 二选一）
- `server/adapters/ane.py`：把 ANE-LM 的推理能力封成统一接口

### 启动命令（示例）
- Python：
  ```bash
  uv run python -m server --host 127.0.0.1 --port 11222
  ```
- Node：
  ```bash
  bun run server --host 127.0.0.1 --port 11222
  ```

### 环境变量（可选）
- `ANE_MODEL_ID=ane-qwen3.5-2b`
- `ANE_MAX_TOKENS=512`

---

## 3. API 契约

### 3.1 `GET /v1/models`

#### Response
```json
{
  "object": "list",
  "data": [
    {
      "id": "ane-qwen3.5-2b",
      "object": "model",
      "created": 0,
      "owned_by": "local"
    }
  ]
}
```

### 3.2 `POST /v1/chat/completions`

#### Request（subset）
支持字段：
- `model: string`
- `messages: Array<{role: 'system'|'user'|'assistant'|'tool', content: string, tool_call_id?: string}>`
- `temperature?: number`
- `max_tokens?: number`
- `stream?: false`（若 true 返回 400，不支持）
- `tools?: Array<ToolDef>`（OpenAI tools 结构）
- `tool_choice?: 'auto'|'none'|{type:'function',function:{name:string}}`

ToolDef（OpenAI 风格，function only）：
```json
{
  "type": "function",
  "function": {
    "name": "run_date",
    "description": "Return current system time",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  }
}
```

#### Response（non-stream）
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 123,
  "model": "ane-qwen3.5-2b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "...",
        "tool_calls": [
          {
            "id": "call_...",
            "type": "function",
            "function": {
              "name": "run_date",
              "arguments": "{}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

> 说明：当返回了 `tool_calls` 时，`finish_reason` 应为 `"tool_calls"`；正常自然语言结束时为 `"stop"`。usage 统计可先填 0（v0 可接受）。

---

## 4. Tool calling：最小闭环（必须实现）

### 4.1 目标行为
当用户问：
> “现在几点了？用工具回答。”

服务端应：
1) 第一次让模型输出 tool_calls（例如 `run_date`）
2) 服务端执行 `run_date` 得到结果
3) 将结果作为 `role=tool` message 加回 messages
4) 再调用模型，得到最终自然语言：
   - “当前时间是 …”

### 4.2 Tool 调用协议（内部约定）
由于多数本地模型不一定原生输出 OpenAI tool_calls JSON，需要服务端实现“**可解析的工具调用标记**”。建议两种路线二选一：

#### 路线 A（推荐，最简单可靠）：强约束输出 JSON
系统提示词注入：
- 当需要调用工具时，必须输出严格 JSON：
  ```json
  {"tool_call": {"name": "run_date", "arguments": {}}}
  ```
服务端检测到顶层 JSON 后执行工具。

#### 路线 B（更像 OpenAI，但更难）：让模型直接输出 tool_calls 结构
让模型按 OpenAI 风格输出 tool_calls；服务端做 JSON 提取。

v0 推荐路线 A，稳定后再升级路线 B。

### 4.3 内置工具（v0）
先只实现一个，确保闭环：

1) `run_date`：无参，返回 `date` 输出
2)（可选）`run_shell`：
   - 参数：`cmd: string`
   - **安全**：只允许 allowlist，比如 `date`, `uname -a`, `uptime`。禁止任意命令。

工具返回给模型时，用：
- `role=tool`
- `tool_call_id` 对应第一次返回的 `tool_calls[i].id`
- `content` 为工具 stdout（必要时附 stderr/exitCode）

---

## 5. 适配 ANE-LM 推理层（Adapter 约束）

需要抽象一个函数：
```ts
generate({messages, temperature, max_tokens}): Promise<string>
```
或 Python 等价。

要求：
- 把 OpenAI messages 转为 ANE-LM 可用的 prompt（Chat template / jinja）
- 支持 system/user/assistant/tool 四种 role（tool 作为文本注入）
- 可在 v0 先用简单拼接模板（后续再对齐官方 chat template）

---

## 6. 错误处理（最小）

- model 不存在：400
- stream=true：400，message: "streaming not supported"
- tool_choice 指定了未知 tool：400
- 工具执行失败：
  - 仍返回 200，但 tool message content 包含错误（更易调试）
  - 或返回 500（v0 可先 200）

错误 JSON 形状尽量像 OpenAI：
```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "param": "...",
    "code": null
  }
}
```

---

## 7. 验收测试（Roy 当前需要的结果）

### 7.1 基础可用
```bash
curl -sS http://127.0.0.1:11222/v1/models | jq
```

### 7.2 非 tool 的最小对话
```bash
curl -sS http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"ane-qwen3.5-2b",
    "messages":[{"role":"user","content":"你好，回复 OK"}],
    "temperature":0
  }' | jq
```
期望：assistant content 含 "OK"。

### 7.3 Tool call 闭环（关键）
```bash
curl -sS http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"ane-qwen3.5-2b",
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
期望：最终 assistant content 给出当前时间（或至少包含 date 输出）。

---

## 8. 实现建议（给 coding agent 的任务拆分）

1. 建一个最小 HTTP server（FastAPI/Fastify）
2. 实现 `/v1/models`
3. 实现 `/v1/chat/completions`（不含 tools）
4. 加入 tool schema 注入 prompt（路线 A）
5. 完成 tool 解析与执行（run_date）
6. 实现 tool loop（二次调用模型输出最终答案）
7. 补齐错误 JSON 结构
8. 写 README：启动、curl 示例

---

## 9. 安全备注（必须写进 README）

- v0 若提供 `run_shell`，必须 allowlist（仅 `date` 等），避免远程执行任意命令。
- 默认仅监听 `127.0.0.1`，不要暴露到局域网/公网。

