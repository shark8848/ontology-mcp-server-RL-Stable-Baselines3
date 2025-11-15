# 执行日志功能说明

## 概述

执行日志功能为 Agent 提供了完整的运行时可观察性,记录每个执行环节的输入输出,包括:
- 用户输入和最终答案
- 记忆检索和保存
- LLM 输入输出(含完整消息和工具列表)
- 工具调用及其结果
- 推理迭代过程

## 功能特性

### 1. 全面的日志记录

记录 15 种执行步骤类型:

| 步骤类型 | 图标 | 说明 |
|---------|-----|------|
| `user_input` | 📝 | 用户输入的原始问题 |
| `memory_retrieval` | 🧠 | 记忆检索(模式和结果) |
| `memory_context` | 📚 | 检索到的历史上下文 |
| `enhanced_prompt` | 🎯 | 增强后的提示词(含上下文) |
| `iteration_start` | 🔄 | 推理迭代开始标记 |
| `llm_input` | 📤 | LLM 输入(消息+工具) |
| `llm_output` | 📥 | LLM 输出(内容+工具调用) |
| `tool_call` | 🔧 | 工具调用(名称+参数) |
| `tool_result` | ✅ | 工具执行结果 |
| `final_answer` | 🎉 | 最终答案 |
| `memory_save` | 💾 | 保存记忆操作 |
| `memory_saved` | ✅ | 记忆保存成功 |
| `execution_complete` | 🏁 | 执行完成总结 |
| `max_iterations` | ⚠️ | 达到最大迭代次数 |

### 2. 详细的元数据

每条日志包含:
- **时间戳**: ISO 8601 格式的精确时间
- **步骤类型**: 标识日志类型
- **内容**: 主要内容(文本/JSON)
- **元数据**: 上下文信息(如迭代次数、工具名称等)

### 3. 智能格式化

- 使用 Emoji 图标增强可读性
- Markdown 格式化(代码块、引用、列表)
- 智能截断(避免过长内容)
- JSON 高亮显示

## 使用示例

### 1. 在代码中使用

```python
from agent.react_agent import LangChainAgent

agent = LangChainAgent()
result = agent.run("查询VIP用户折扣")

# 获取执行日志
execution_log = result['execution_log']

# 统计日志类型
log_types = {}
for log in execution_log:
    step_type = log['step_type']
    log_types[step_type] = log_types.get(step_type, 0) + 1

print(f"执行日志条目: {len(execution_log)}")
print(f"工具调用次数: {log_types.get('tool_call', 0)}")
```

### 2. 在 Gradio UI 中查看

启动 UI:
```bash
python3 -m agent.gradio_ui
```

执行日志会在聊天界面下方的 **"运行日志"** 面板中实时显示,包括:
- 用户输入和系统响应
- 每轮 LLM 的输入输出
- 工具调用的完整参数和结果
- 记忆操作的详细信息

### 3. 格式化显示

```python
from agent.gradio_ui import format_execution_log

# 格式化日志为 Markdown
formatted = format_execution_log(execution_log)
print(formatted)
```

## 日志结构

### 基础结构

```json
{
  "step_type": "llm_output",
  "timestamp": "2025-11-10T14:03:38.488469",
  "content": "LLM 生成的文本内容",
  "metadata": {
    "iteration": 1,
    "tool_calls_count": 1
  }
}
```

### 常见日志类型示例

#### 用户输入
```json
{
  "step_type": "user_input",
  "timestamp": "2025-11-10T14:03:35.151089",
  "content": "我是VIP客户，订单1000元能打几折？"
}
```

#### LLM 输入
```json
{
  "step_type": "llm_input",
  "timestamp": "2025-11-10T14:03:35.151114",
  "content": "完整的 messages 列表",
  "metadata": {
    "iteration": 1,
    "messages_count": 4,
    "tools_count": 3,
    "tools": ["ontology_explain_discount", "ontology_normalize_product", "ontology_validate_order"]
  }
}
```

#### 工具调用
```json
{
  "step_type": "tool_call",
  "timestamp": "2025-11-10T14:03:38.488507",
  "content": {
    "name": "ontology_explain_discount",
    "arguments": {
      "is_vip": true,
      "amount": 1000
    }
  },
  "metadata": {
    "iteration": 1
  }
}
```

#### 工具结果
```json
{
  "step_type": "tool_result",
  "timestamp": "2025-11-10T14:03:38.491206",
  "content": "{\"@type\": \"DiscountExplanation\", \"discount_applied\": false, \"discount_rate\": 0.0}",
  "metadata": {
    "iteration": 1,
    "tool_name": "ontology_explain_discount"
  }
}
```

## 性能优化

### 1. 内容截断

为避免日志过大,自动截断长内容(默认 4000 字符)。可通过 `TOOL_LOG_MAX_CHARS` 环境变量调整上限:

```bash
export TOOL_LOG_MAX_CHARS=8000
```

此外,执行日志历史中的文字摘要默认为 500 字符,可通过 `EXEC_LOG_SNIPPET_CHARS` 调整:

```bash
export EXEC_LOG_SNIPPET_CHARS=1200
```

该参数影响 LLM 输出摘要、工具参数片段、最终回答等短文本渲染。

也可以直接在 `src/agent/config.yaml` 中的 `ui` 段落设置:

```yaml
ui:
  tool_log_max_chars: 8000
  execution_log_snippet_chars: 1200
```

### 2. 选择性记录

可通过配置控制日志记录级别:
```python
# 在 react_agent.py 中自定义
add_log("tool_result", observation[:500])  # 只记录前500字符
```

### 3. 异步写入(未来增强)

当前日志在内存中累积,未来可考虑:
- 异步写入文件
- 流式输出到日志系统
- 定期清理旧日志

## 故障排查

### 问题 1: 日志为空

**症状**: `execution_log` 为空列表

**原因**:
- Agent 执行失败
- 日志记录代码被跳过

**解决**:
```python
# 检查 agent.run() 是否成功
result = agent.run(query)
if not result.get('execution_log'):
    print("警告: 执行日志为空")
    print(f"最终答案: {result.get('final_answer')}")
```

### 问题 2: UI 显示不完整

**症状**: Gradio UI 中日志显示被截断

**原因**:
- Markdown 组件有长度限制
- 日志内容过长

**解决**:
- 减小日志记录的内容长度
- 使用分页显示
- 添加折叠/展开功能

### 问题 3: 时间戳不准确

**症状**: 时间戳与实际时间不符

**原因**:
- 系统时区设置问题
- datetime 使用不当

**解决**:
```python
from datetime import datetime

# 使用 UTC 时间
timestamp = datetime.utcnow().isoformat()

# 或使用本地时间
timestamp = datetime.now().isoformat()
```

## 最佳实践

### 1. 分析日志

```python
# 统计各步骤耗时(需增强日志记录持续时间)
from datetime import datetime

def analyze_timing(logs):
    timings = []
    for i in range(len(logs) - 1):
        start = datetime.fromisoformat(logs[i]['timestamp'])
        end = datetime.fromisoformat(logs[i+1]['timestamp'])
        duration = (end - start).total_seconds()
        timings.append({
            'step': logs[i]['step_type'],
            'duration': duration
        })
    return timings
```

### 2. 导出日志

```python
import json
from datetime import datetime

def export_log(logs, filename=None):
    if filename is None:
        filename = f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    
    print(f"日志已导出: {filename}")
```

### 3. 过滤日志

```python
def filter_logs(logs, step_types):
    """只保留指定类型的日志"""
    return [log for log in logs if log['step_type'] in step_types]

# 示例: 只看工具调用相关日志
tool_logs = filter_logs(execution_log, ['tool_call', 'tool_result'])
```

## 未来增强

1. **日志持久化**
   - 保存到文件/数据库
   - 支持日志查询和分析

2. **高级过滤**
   - 按时间范围过滤
   - 按步骤类型过滤
   - 按关键词搜索

3. **可视化**
   - 执行时间线图
   - 工具调用关系图
   - 性能瓶颈分析

4. **导出功能**
   - 导出为 JSON/CSV
   - 生成 HTML 报告
   - 集成到日志系统

5. **实时监控**
   - WebSocket 流式传输
   - 实时性能指标
   - 异常告警

## 相关文档

- [配置指南](MEMORY_CONFIG_GUIDE.md) - 记忆系统配置
- [API 文档](API.md) - Agent API 说明
- [故障排查](TROUBLESHOOTING.md) - 常见问题解决

## 示例输出

### 控制台输出

```
================================================================================
📋 UI 格式化的执行日志
================================================================================
## 运行日志

**记录数**: 15 条

### 📝 步骤 1: 用户输入
```
我是VIP客户，订单1000元能打几折？
```

<small>时间戳: 2025-11-10T14:03:35.151089</small>

---

### 🧠 步骤 2: 记忆检索
- **模式**: recent
- **结果长度**: 0 字符

<small>时间戳: 2025-11-10T14:03:35.151104</small>

---

### 🔧 步骤 7: 工具调用 (轮次 1)
- **工具**: `ontology_explain_discount`
- **参数**:
```json
{
  "is_vip": true,
  "amount": 1000
}
```

<small>时间戳: 2025-11-10T14:03:38.488507</small>

---

### ✅ 步骤 8: 工具结果 (轮次 1)
- **工具**: `ontology_explain_discount`
- **结果**:
```
{"@type": "DiscountExplanation", "discount_applied": false, "discount_rate": 0.0}
```

<small>时间戳: 2025-11-10T14:03:38.491206</small>

---
```

### Gradio UI 显示

在 Web 界面中,日志会以富文本形式显示,包括:
- 图标增强的步骤标题
- 格式化的代码块
- 清晰的时间戳
- 分隔线区分不同步骤

## 技术实现

### 日志收集 (react_agent.py)

```python
def run(self, user_input: str) -> dict:
    execution_log = []
    
    def add_log(step_type, content, metadata=None):
        execution_log.append({
            "step_type": step_type,
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "metadata": metadata or {}
        })
    
    # 记录用户输入
    add_log("user_input", user_input)
    
    # 记录记忆检索
    add_log("memory_retrieval", f"使用{mode}检索", {"mode": mode})
    
    # ... 更多日志记录
    
    return {
        "final_answer": answer,
        "execution_log": execution_log
    }
```

### 日志格式化 (gradio_ui.py)

```python
def format_execution_log(logs: list) -> str:
    if not logs:
        return "暂无执行日志"
    
    lines = ["## 运行日志\n", f"**记录数**: {len(logs)} 条\n"]
    
    for i, log in enumerate(logs, 1):
        step_type = log.get("step_type", "unknown")
        timestamp = log.get("timestamp", "N/A")
        content = log.get("content", "")
        metadata = log.get("metadata", {})
        
        # 根据类型选择图标和格式
        if step_type == "user_input":
            lines.append(f"### 📝 步骤 {i}: 用户输入\n")
            lines.append(f"```\n{content}\n```\n")
        elif step_type == "tool_call":
            lines.append(f"### 🔧 步骤 {i}: 工具调用\n")
            # ... 格式化工具调用
        # ... 更多步骤类型
        
        lines.append(f"<small>时间戳: {timestamp}</small>\n\n---\n\n")
    
    return "".join(lines)
```

## 总结

执行日志功能为 Agent 提供了完整的运行时可观察性,帮助开发者和用户:
- **理解**: Agent 的决策过程
- **调试**: 快速定位问题
- **优化**: 识别性能瓶颈
- **审计**: 追踪完整执行历史

通过结合 Gradio UI 的实时显示,用户可以清晰地看到 Agent 每一步的思考和行动,极大提升了系统的透明度和可信度。
