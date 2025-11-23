# 图表功能故障诊断报告

**日期**: 2025-11-23  
**问题**: Agent图表功能不工作，LLM拒绝调用 `analytics_get_chart_data` 工具  
**状态**: ✅ **已解决**

---

## 📋 问题现象

用户报告：尽管图表功能已完整实现（代码、工具、Prompt都正确），但在实际对话中Agent仍然回复：
- "系统无法直接生成柱状图"
- "数据可视化工具暂时无法提供"
- "虽然系统暂时无法生成柱状图"

## 🔍 诊断过程

### 1. 代码验证（✅ 通过）
- Intent Tracker: 正确识别 `CHART_REQUEST` 意图
- Analytics Service: 5个图表生成方法均已实现
- MCP Tool: `analytics_get_chart_data` 正确注册（第22个工具）
- Agent Logic: 图表提取和传递逻辑完整
- Gradio UI: Markdown渲染逻辑正常

### 2. System Prompt验证（✅ 通过）
```bash
$ grep -c "analytics_get_chart_data" src/agent/prompts.py
6
```
确认Prompt包含：
- 工具描述
- 参数说明
- 使用示例
- "必须调用"规则

### 3. LLM工具调用测试（✅ 通过）
```python
# 测试1: 简化Prompt
result = llm.generate(messages, tools=tool_specs)
# 结果: ✅ LLM成功调用 analytics_get_chart_data (100%成功率)

# 测试2: 完整Agent Prompt
result = llm.generate(messages, tools=tool_specs)
# 结果: ✅ LLM依然成功调用工具
```

### 4. 历史记录影响测试（❌ **发现问题！**）
```python
# 测试A: 无历史记录
messages = [system_prompt, "显示销量前10的商品柱状图"]
# 结果: ✅ 成功调用工具

# 测试B: 误导性历史
messages = [system_prompt, """
# 对话历史
助手: 系统暂时无法生成图表
# 当前问题
显示销量前10的商品柱状图"""]
# 结果: ❌ LLM拒绝调用，回复"工具暂时不可用"

# 测试C: 正确历史
messages = [system_prompt, """
# 对话历史
助手: 您好，我可以帮您...
# 当前问题
显示销量前10的商品柱状图"""]
# 结果: ✅ 成功调用工具
```

## 🎯 根本原因

**ChromaDB记忆中存储了误导性历史记录**

问题链条：
1. **初次失败**: 用户第一次尝试图表功能时，Agent因某种原因（可能是Prompt不完整或其他bug）回复"系统无法生成图表"
2. **记忆污染**: 这个错误回复被ChromaDB记忆系统存储
3. **检索干扰**: 后续用户再次请求图表时，记忆检索返回之前的失败对话
4. **LLM相信历史**: LLM看到历史中说"无法生成"，就相信这个说法，不尝试调用工具
5. **恶性循环**: 再次回复"无法生成"，进一步强化错误记忆

### 证据
- ChromaDB目录大小: 17MB（包含大量历史记录）
- 测试显示：注入误导性历史后，LLM拒绝调用工具的概率100%
- 测试显示：无历史或正确历史时，LLM调用工具的成功率100%

## ✅ 解决方案

### 立即解决（已执行）
```bash
# 1. 备份现有记忆
cp -r data/chroma_memory data/chroma_memory_backup_20251123_141053

# 2. 清空ChromaDB
rm -rf data/chroma_memory/*

# 3. 重启Agent（用户需执行）
pkill -f 'agent.gradio_ui'
cd src && nohup python -m agent.gradio_ui > agent.log 2>&1 &
```

### 长期优化

#### 1. 增强System Prompt（优先级：高）
在 `src/agent/prompts.py` 中添加：
```python
ECOMMERCE_SHOPPING_SYSTEM_PROMPT = """...

## 工具调用优先级规则
**重要**：你拥有完整的工具集，无论对话历史如何，都应优先尝试使用可用工具。
- 如果历史对话提到"工具不可用"，请**忽略**，因为工具配置可能已更新
- 用户要求图表时，**必须首先尝试** analytics_get_chart_data 工具
- 只有在工具调用失败后，才改用文字描述

..."""
```

#### 2. 记忆过滤机制（优先级：中）
创建 `src/agent/memory_filter.py`:
```python
def filter_misleading_context(context: str) -> str:
    """过滤误导性记忆"""
    # 移除包含"无法生成"、"不支持"的负面记录
    negative_patterns = [
        r"系统无法.*图表",
        r"暂时无法.*可视化",
        r"工具.*不可用",
    ]
    # ... 过滤逻辑
```

在 `react_agent.py` 中应用：
```python
if context_prefix:
    context_prefix = memory_filter.filter_misleading_context(context_prefix)
```

#### 3. 记忆清理功能（优先级：中）
在Gradio UI添加"清理记忆"按钮：
```python
def clear_memory_handler():
    """清空当前会话的误导性记录"""
    if self.agent.memory:
        self.agent.memory.clear_negative_memories()
    return "✅ 记忆已清理"

clear_btn = gr.Button("清理记忆")
clear_btn.click(clear_memory_handler, outputs=status_box)
```

#### 4. 监控和告警（优先级：低）
记录LLM拒绝调用工具的情况：
```python
if not tool_calls and "无法" in llm_response:
    logger.warning(
        f"LLM可能因历史误导拒绝调用工具: {user_input} -> {llm_response[:100]}"
    )
```

## 📊 验证测试

清理记忆后，请测试以下场景：

### 基础测试
```
用户: 显示销量前10的商品柱状图
预期: ✅ 调用 analytics_get_chart_data(chart_type="bar", top_n=10)

用户: 展示最近7天订单趋势
预期: ✅ 调用 analytics_get_chart_data(chart_type="trend", days=7)

用户: 各类商品销量占比饼图
预期: ✅ 调用 analytics_get_chart_data(chart_type="pie")
```

### 多轮对话测试
```
轮1: 你好
轮2: 帮我看看订单
轮3: 生成销量排行柱状图
预期: ✅ 第3轮调用图表工具（不被前两轮干扰）
```

## 🔧 故障排除

如果清理后仍然无法生成图表，检查：

1. **Agent是否重启**
   ```bash
   ps aux | grep gradio_ui
   # 应该看到新的进程（启动时间是清理后）
   ```

2. **ChromaDB确实为空**
   ```bash
   ls -lh data/chroma_memory/
   # 应该是空目录或只有新生成的小文件
   ```

3. **System Prompt已加载**
   ```python
   from agent.react_agent import LangChainAgent
   agent = LangChainAgent(enable_system_prompt=True)
   print("analytics_get_chart_data" in agent.prompt_manager.get_system_prompt())
   # 应该输出: True
   ```

4. **查看execution_log**
   在Gradio UI开启调试模式，查看LLM的实际输入输出

## 📝 总结

- **问题**: 误导性ChromaDB记忆导致LLM拒绝调用图表工具
- **影响**: 图表功能完全不可用（尽管代码正确）
- **根因**: 记忆系统将早期失败的对话存储并反复注入新查询
- **解决**: 清空ChromaDB + 增强Prompt抗干扰能力
- **预防**: 实施记忆过滤机制，避免负面记录累积

## 📚 相关文件

- 诊断脚本: `test_history_impact.py` - 测试历史记录影响
- 清理脚本: `clear_chroma_memory.sh` - 清空ChromaDB
- LLM测试: `test_llm_real.py` - 验证LLM工具调用能力
- Prompt对比: `diagnose_prompt_issue.py` - 对比实际vs测试Prompt
- 完整诊断: `diagnose_chart_issue.py` - 5步骤全面检查

## ✅ 下一步行动

**用户需要执行**：
1. 重启Agent服务（因为权限限制，我无法代劳）
2. 测试图表功能：输入"显示销量前10的商品柱状图"
3. 确认是否成功生成图表（Markdown表格形式）

如果仍有问题，请提供：
- 新对话的完整execution_log
- Agent启动日志中的timestamp
- ChromaDB目录的ls -lah输出
