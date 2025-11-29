# Ontology RL Commerce Agent：交互时序图的解读

> 对照 [`docs/interaction_sequence_diagrams.md`](../interaction_sequence_diagrams.md) 可获取完整 Mermaid 时序图与截图；本文聚焦解释每个阶段背后的模块分工与技术要点。

## 0. 阅读指南

- 以 5 段对话作为切分单元，覆盖推荐、搜索、下单、售后与分析链路。
- 每段输出“核心组件”“数据/控制流关键节点”“失效信号”，方便直接映射到日志或调试会话。
- 专注于系统级逻辑，不再展开教学式操作步骤。

## 1. 顶层处理链路速览

| 阶段 | 输入/触发 | Agent 内部动作 | 输出/验证 |
| --- | --- | --- | --- |
| 入口 | Gradio UI → `forward_message` | Session 写入、启动 ReAct Agent | Streaming 思考块触发 | 
| 感知 | Intent Tracker、Query Rewriter | 意图打分、关键词派生、检索策略拼装 | 结构化任务（tool + params） |
| 执行 | MCP Adapter → MCP Server → Commerce/Analytics | 受控工具调用、本体推理、SHACL 校验、记忆写入 | 工具响应或错误码 |
| 思考 | DeepSeek LLM（`run_stream`） | 结合工具响应实时生成 token delta | UI 层逐 token 渲染 |
| 记忆 | Chroma Memory | `upsert_turn` 写最近摘要与事实 | 后续轮可直接引用 |

> 开发者可把 Agent 视作 orchestrator：Intent/Rewrite/LLM/Memory 负责“认知”，MCP/Commerce/Analytics 负责“行动”。跨域问题往往发生在 orchestrator 与工具层交界处。

## 2. Conversation 1 —— Streaming

- **核心组件**：Intent Tracker、Query Rewriter、`commerce.search_products`、DeepSeek streaming。
- **控制流**：UI → Agent → Intent 判定 `recommendation` → Query Rewrite 生成多类别关键词 → FTS5 检索循环（不足即类别 fallback）→ LLM streaming → Memory 记要点。
- **技术要点**：
  - 批量检索通过 `loop` 表示，说明 Agent 会根据 LLM 中间结果动态拉取更多候选，直到达到 10 条或命中率下降。
  - `run_stream` 将“thought + final answer”拆成 token delta，开发者可在日志中看到 `analysis` 与 `response` 两个 channel。
  - Memory 写入节点紧跟 streaming 结束，确保后续轮可引用商品列表。
- **失效信号**：UI 停留在“思考中”大于 5 秒，多数是 MCP 请求阻塞或 DeepSeek 流中断，可在 `logs/server_*.log` 检查同时间戳的工具调用耗时。

## 3. Conversation 2 —— 意图切换驱动的多轮检索

- **核心组件**：Intent Tracker（`search ↔ price_inquiry`）、Hybrid Query Rewriter、`commerce.search_products` with price bounds。
- **控制流**：每次用户追问都重新跑 Intent → Query Rewrite 输出类别/关键词/价格范围 → tool payload 携带 `max_price` 或 `min_price` → FTS5 命中失败即回退 LIKE + 类别过滤。
- **技术要点**：
  - 置信度 0.60 左右时，Agent 依靠历史记忆辅助决策；这是 `loop` 中 Memory 调用的意义。
  - 价格区间由 rewrite 引擎负责，工具层不再做推断，简化了 Commerce Service 的逻辑。
  - Fallback 顺序（FTS5 → LIKE → category）在图里以 `loop 追加咨询` 表示，可对应到 `commerce_service.py` 的多策略调用。
- **失效信号**：`loop` 中连续的 200 OK 但返回列表为空，意味着索引未覆盖或 rewrite 未生成价格条件。抓取 payload 即可定位。

## 4. Conversation 3 —— 订单执行闭环

- **核心组件**：`view_cart`、`remove_from_cart`、`add_to_cart`、`create_order`、`process_payment`；Ontology/SHACL pipeline。
- **控制流**：Agent 先获取购物车 → `loop` 删除旧条目 → `loop` 添加 MagSafe + iPhone → 用户提交地址/电话 → `create_order`（触发 ontology + SHACL + 库存/累计消费更新）→ `process_payment`。
- **技术要点**：
  - 购物车操作以两层 `loop` 形式展示，可定位具体商品 ID（如 659/18）。
  - 橙色矩形标识“受本体/校验保护”的步骤，开发者可据此快速找到必须保证字段完整的 API。
  - `commerce.process_payment` 仅在 `create_order` 成功后才执行，防止未生成 ORD 编号时提前扣款。
- **失效信号**：若橙色段直接返回 400/422，多半是 SHACL 校验未过或库存锁失败；抓取 `server.log` 中的校验消息即可定位字段。

## 5. Conversation 4 —— 售后链路的读写分离

- **核心组件**：`get_shipment_status`、`get_user_orders`、`get_order_detail`、`get_user_profile`。
- **控制流**：用户声明已支付 → Agent 尝试 `get_shipment_status`（因未发货收到 400）→ 连续调用订单/画像相关工具 → LLM streaming 输出核对信息。
- **技术要点**：
  - 物流接口返回 400 时附带 `ValueError` 描述，有助于前端直接提示“尚未生成物流”。
  - 任一查询命中用户画像或订单详情都会触发本体推理（橙色段），确保 VIP/SVIP 等派生属性在返回结果中保持一致。
  - 读接口均保持幂等，可反复调用而不污染状态。
- **失效信号**：如果多次查询地址为空，先确认 `get_user_profile` 的 payload，再检查 Memory 是否覆盖了更旧的用户信息。

## 6. Conversation 5 —— 分析服务与 UI 渲染链路

- **核心组件**：`commerce.get_user_orders`、Analytics Service `render_chart`（饼图/柱状图）。
- **控制流**：Intent → 获取订单与累计金额 → 两次调用 Analytics Service（pie → bar）→ 返回 JSON + Base64 → LLM 拼出 Markdown → UI 渲染图表。
- **技术要点**：
  - Analytics Service 响应里既含数据也含 Base64，Agent 只需内联即可展示，无需额外文件托管。
  - 若同一请求生成多图，需注意 LLM 顺序，避免写入 Markdown 时互相覆盖。
  - Intent 仍为 `price_inquiry`，但从工具调用可看出 Agent 已根据上下文切换到统计视角。
- **失效信号**：图像无法显示时，先用日志检查返回的 Base64 长度，再确认 UI 允许 `data:image/png;base64,...`；若 Markdown 渲染失败，会在 UI 控制台看到报错。

## 7. 面向开发者的延伸要点

1. **脚本回放**：用 CLI 模拟 5 段会话，可复现 streaming、intent 切换、订单闭环等关键路径，便于回归测试。
2. **策略训练落地**：把每段的状态切片注入 RL replay buffer，可验证“状态-动作-奖励”是否覆盖关键工具序列。
