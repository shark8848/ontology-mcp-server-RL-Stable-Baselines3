# ChromaDB + MCP：支撑多轮上下文记忆与工具编排的语义骨架

> 参考文件：`src/agent/chroma_memory.py`、`src/agent/memory.py`、`src/agent/mcp_adapter.py`、`src/agent/react_agent.py`、`docs/interaction_sequence_diagrams.md`
>
> 目标：解析项目中“记忆系统 + MCP 工具协议”协同的技术实现，说明它如何支撑多轮语义对齐、上下文注入与可追踪的工具编排。

## 1. 架构定位：记忆是 Agent 的语义中枢

```
User ↔ Gradio UI ↔ ReAct Agent ↔ (Intent Tracker + Query Rewriter)
                                      ↓
                                  Memory Layer (ChromaDB)
                                      ↓
                              MCP Adapter → MCP Server → Tools
```

- **Memory Layer**：保存“最近摘要 + 语义相似片段 + 结构化事实”，对齐后续 Prompt 的上下文。
- **MCP Adapter**：把 Agent 的工具调用映射为统一的 HTTP payload，包含参数、trace_id、错误处理策略。
- 两者协同，使 Agent 在多轮会话中“既记得历史，也能调用正确的能力”。

## 2. Chroma 记忆流：写入与检索

### 2.1 写入逻辑

`src/agent/chroma_memory.py` 的 `ChromaMemory.add_turn()` 负责记录每轮对话的用户输入、Agent 思考、工具调用与回复摘要。

```python
class ChromaMemory:
    def add_turn(self, user_message: str, assistant_message: str, tool_calls: list[ToolCall]) -> None:
        # 1. 生成摘要
        summary = self.summary_generator.generate_summary(user_message, assistant_message)
        # 2. 拼装 Metadata（意图、工具、订单状态等）
        metadata = self._build_metadata(tool_calls, assistant_message)
        # 3. 写入 Chroma Collection
        self.collection.add(
            ids=[self._build_turn_id()],
            documents=[summary],
            metadatas=[metadata],
            embeddings=[self.embedder.embed(summary)]
        )
```

- `summary_generator` 默认基于 LLM（DeepSeek）或规则模板，确保每条摘要长度可控。
- Metadata 包含 `intent`, `tools_used`, `order_state`, `user_level` 等字段，方便后续过滤。
- `embedder` 使用指定的 embedding 模型（Ollama/DeepSeek/OpenAI），保证所有摘要可进行相似度搜索。

### 2.2 检索策略

`get_context_for_prompt()` 会按照优先级拼接三类内容：

1. **Recent turns**：最近 N 轮原文，通常为 2~3 轮，保证连续性。
2. **Semantic matches**：基于 embedding 的相似度检索，使用 `collection.query()`。
3. **Structured facts**：例如用户 ID、订单号、历史意图，直接从 metadata 或 `memory.py` 的状态缓存中拉取。

```python
def get_context_for_prompt(self, query: str, max_tokens: int = 1200) -> list[str]:
    bullets = []
    bullets.extend(self._fetch_recent_turns())
    bullets.extend(self._fetch_semantic_matches(query))
    bullets.extend(self._fetch_structured_facts())
    return self._truncate_to_tokens(bullets, max_tokens)
```

- `_truncate_to_tokens` 会根据模型上下文限制做截断，避免 prompt 爆炸。
- `query` 由 Agent 在每轮调用 `self.memory.get_context_for_prompt(user_message)` 时传入。

## 3. MCP Adapter：把记忆衍生的意图映射为工具调用

Memory 层提供上下文后，Agent 会决定调用哪个 MCP 工具。`src/agent/mcp_adapter.py` 管理请求构造、日志记录与错误抛出。

```python
class MCPAdapter:
    async def invoke_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        payload = {
            "tool": tool_name,
            "params": params,
            "session_id": self.session_id,
            "trace_id": str(uuid.uuid4()),
        }
        resp = await self._client.post("/invoke", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return ToolResult.from_response(resp.json())
```

- `trace_id` 对应 `Execution Log` 中的唯一标识，用于把工具调用与记忆写入/LLM 输出关联起来。
- 一旦工具返回错误，会抛出异常，由 `react_agent.py` 捕获并在 LLM 输出中解释。

## 4. ReactAgent 如何消费记忆 + 调度工具

`src/agent/react_agent.py` 中的 `handle_user_message()` 是核心入口。

```python
class ReactAgent:
    async def handle_user_message(self, user_message: str) -> AsyncGenerator[StreamChunk, None]:
        context = self.memory.get_context_for_prompt(user_message)
        plan = self.plan_builder.build_plan(user_message, context)
        async for chunk in self._execute_plan(plan, context):
            yield chunk
```

- Plan 中包含“思考步骤 + 工具调用 + 最终回答”。
- `_execute_plan` 会根据每一步的 `tool_name` 调用 `MCPAdapter.invoke_tool()`。
- 工具结果再被注入下一次思考的 prompt 中，形成“Thought → Action → Observation → Thought”闭环。

## 5. 记忆信号如何影响工具选择

1. **意图引导**：Memory 里保存的 `intent_history` 会作为附加特征传给 `IntentTracker`，降低短时抖动。
2. **实体回忆**：地址、电话、订单号等字段写入 metadata 后，Agent 在需要填写参数时优先读取，避免重复询问。
3. **冲突检测**：当用户要撤销刚才的订单，Memory 会说明“上一轮刚创建 ORD...”，Agent 便可调用 `commerce.cancel_order`。

## 6. 结合交互图理解控制流

- Conversation 1/2：Memory 提供“上轮推荐内容 + 最低价/最高价”供参考，Agent 决定是否继续搜索。
- Conversation 3：购物车重建和订单创建后的摘要记录了订单号、支付渠道，使后续售后查询有据可依。
- Conversation 4/5：历史摘要中包含“支付完成”“尚未生成物流”“总消费金额”，对应的工具调用因此带着精准的参数。

## 7. 常见调试路径

1. **记忆缺失**：检查 `data/chroma_memory/` 中的 sqlite/parquet 文件是否更新；或在 `ChromaMemory.add_turn` 打日志。
2. **检索不准**：确认 embedding 模型与 `collection.metadata` 中的 `embedding_model` 匹配；必要时重建 collection。
3. **MCP 请求异常**：观察 `trace_id` 在 `logs/server_*.log` 中的链路，确认工具是否收到正确参数。
4. **Prompt 爆炸**：调低 `recent_turns` 数量或 `max_tokens`，同时优化摘要生成的长度。

## 8. 结论：语义记忆 + 工具协议 = 可控的多轮智能体

- Chroma 让多轮对话具备可查询的“语义记忆库”。
- MCP Adapter 让工具调用标准化、可审计，并与记忆 metadata 互相印证。
- 两者在 `react_agent.py` 中汇合，使 Agent 能“先理解上下文，再做出可解释的行动”。

未来的优化方向包括：为 Memory 增加“置信度/时间衰减”权重、为 MCP 添加断路器/重试策略，以及把记忆摘要同步到 Analytics 以增强可观察性。