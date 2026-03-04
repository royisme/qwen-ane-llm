# Known Issues

## Tool Calling Not Working (as of 2026-03-03)

**Symptom**: Qwen3.5-2B model directly answers questions instead of calling tools, even when tools are provided and explicitly requested.

**Example**:
```bash
curl -X POST http://127.0.0.1:11222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-2B",
    "messages": [{"role": "user", "content": "please run_date tool to get current time"}],
    "tools": [{"type": "function", "function": {"name": "run_date", "description": "get current time"}}],
    "temperature": 0
  }'
```

**Expected**: Response with `tool_calls` field populated
**Actual**: Plain text answer without tool call

### Possible Causes

1. **Model size**: 2B model may have weak tool calling capability
2. **Model not fine-tuned**: Base model vs. instruction-tuned for tool use
3. **Prompt format**: May need different XML/tool format

### Possible Solutions

1. **Use larger model**: Try Qwen3.5-4B or 8B
2. **Add more useful tools**: Implement Read/Write/Edit/Bash file operations - the model may be more inclined to use tools if they are more powerful and useful
3. **System prompt tuning**: Adjust the tool description format
4. **Fine-tune**: Fine-tune model for tool calling

---

*Last updated: 2026-03-03*
