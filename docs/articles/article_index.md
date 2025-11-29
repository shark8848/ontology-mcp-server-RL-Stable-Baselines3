# Ontology RL Commerce Agent 宣传文章目录

> 本目录汇总 11 篇面向技术读者的深度文章，覆盖从顶层架构、核心能力到落地案例的完整链路，可按顺序逐步发布。

## 1. [Ontology RL Commerce Agent：为何将本体推理与强化学习结合，重构电商智能体栈](./01-ontology-rl-commerce-agent.md)
- 目标：解释项目愿景与差异化价值，阐述“本体 + RL”组合的动机。
- 要点：整体架构、业务痛点、传统方案局限、关键突破。

## 2. [交互时序图：Agent 核心链路与技术要点](./02-agent-sequence-walkthrough.md)
- 目标：拆解 `docs/interaction_sequence_diagrams.md` 中五段会话的关键环节，向开发者阐明 orchestrator、工具层、Ontology/SHACL、Analytics 的协同方式。
- 要点：`run_stream` streaming、Intent/Rewrite 切换、购物车与订单闭环、售后查询的读写分离、Analytics Base64 渲染、故障信号与日志定位。文章源文件：`docs/articles/02-agent-sequence-walkthrough.md`。

## 3. [ChromaDB + MCP：支撑多轮上下文记忆与工具编排的语义骨架](./03-chromadb-mcp-memory-tooling.md)
- 目标：展示记忆与工具协议如何协同。
- 要点：ChromaDB 模式、MCP tool manifest、记忆读写策略、上下文注入。文章源文件：`docs/articles/03-chromadb-mcp-memory-tooling.md`。

## 4. [22 款 MCP 工具的可组合设计：从库存校验到 SHACL 规则闭环](./04-mcp-tools-and-shacl.md)
- 目标：拆解工具层的工程实践与治理。
- 要点：工具分层、错误处理、SHACL 校验、扩展指引。

## 5. [RL 训练闭环实录：Stable Baselines3 如何驱动 Agent 发现最优任务策略](./05-rl-training-loop.md)
- 目标：剖析强化学习训练与评估链路。
- 要点：Gym 环境、奖励设计、训练基线、上线回滚流程。

## 6. [多模态洞察看板：实时趋势图、销量榜与个性画像的工程落地](./06-analytics-dashboard.md)
- 目标：介绍图表服务与实时洞察实现。
- 要点：analytics 服务、图表过滤、隐私策略、可视化栈。

## 7. [Gradio 五页签电商 UI：将复杂状态机透明化给运营与专家用户](./07-gradio-ui-observability.md)
- 目标：分享 UI/UX 设计如何反映智能体内部状态。
- 要点：Tab 划分、状态提示、快捷短语、可观测性。

## 8. [对话状态、意图与质量评分管道：LangChain + 自研监控的实现细节](./08-conversation-intent-quality.md)
- 目标：讲解多维监控指标如何提升对话可靠性。
- 要点：ConversationState 管理、Intent Tracker、Quality Metrics、告警触发。

## 9. [记忆配置与数据统一层：从 YAML 到 SQLite/Chroma 的多环境可移植性](./09-memory-config-data-layer.md)
- 目标：强调配置化与数据层抽象带来的交付效率。
- 要点：config.yaml、ENV 覆盖策略、数据初始化脚本、迁移方案。

## 10. [执行日志、Tool Calls 与治理：让每一次 Agent 决策都可审计](./10-execution-log-governance.md)
- 目标：突出可观测性与合规性设计。
- 要点：Execution Log 结构、工具追踪、异常回放、治理面板。

## 11. [从 VIP 案例到生产部署：如何扩展为真实商业场景的可观测智能体](./11-vip-case-to-production.md)
- 目标：结合 `docs/VIP_Customer_Case.md` 展示端到端闭环。
- 要点：案例复盘、部署拓扑、监控指标、未来迭代路线。

> 后续可在每篇文章下继续细化章节小节、配图清单与引用代码段，以便团队成员快速分工撰写。
