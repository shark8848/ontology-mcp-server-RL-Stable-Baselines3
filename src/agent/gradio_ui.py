#!/usr/bin/env python3
from __future__ import annotations
# Copyright (c) 2025 shark8848
# MIT License
#
# Ontology MCP Server - 电商 AI 助手系统
# 本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI
#
# Author: shark8848
# Repository: https://github.com/shark8848/ontology-mcp-server
"""Gradio UI for the LangChain-based agent (uses OpenAI-compatible LLM + MCP HTTP adapter).
Run this script from repository root; it will talk to the MCP server at MCP_BASE_URL (default http://localhost:8000).

支持 ChromaDB 持久化对话记忆功能,可以保持上下文连贯性并支持语义检索。
增强的运行日志显示,包括每个环节的输入输出和模型交互细节。
"""

import os
import uuid
import json
import html
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import yaml
import plotly.graph_objects as go
import plotly.io as pio
import base64
import tempfile
from io import BytesIO
from PIL import Image
from gradio_client import utils as gradio_client_utils


_ORIGINAL_GRADIO_GET_TYPE = gradio_client_utils.get_type


def _safe_gradio_get_type(schema):
    if isinstance(schema, bool):
        return "boolean" if schema else "never"
    return _ORIGINAL_GRADIO_GET_TYPE(schema)


gradio_client_utils.get_type = _safe_gradio_get_type

from agent.react_agent import LangChainAgent
from agent.logger import get_logger
from agent.memory_config import get_memory_config

LOGGER = get_logger(__name__)

# 加载记忆配置
MEMORY_CONFIG = get_memory_config()


def _load_agent_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                return data
    except Exception as exc:  # pragma: no cover - config 解析失败时采用默认
        LOGGER.warning("无法读取 config.yaml: %s", exc)
    return {}


def _coerce_positive_int(value):
    if value is None:
        return None
    try:
        ivalue = int(str(value).strip())
    except (ValueError, TypeError):
        return None
    return ivalue if ivalue > 0 else None


def _resolve_int_setting(env_name: str, config_value, default: int) -> int:
    env_value = os.getenv(env_name)
    parsed_env = _coerce_positive_int(env_value)
    if parsed_env is not None:
        return parsed_env
    parsed_cfg = _coerce_positive_int(config_value)
    if parsed_cfg is not None:
        return parsed_cfg
    return default


_AGENT_CONFIG = _load_agent_config()
_UI_CONFIG = _AGENT_CONFIG.get("ui", {}) if isinstance(_AGENT_CONFIG, dict) else {}

_DEFAULT_LOG_MAX_CHARS = 4000
LOG_MAX_CHARS = _resolve_int_setting(
    "TOOL_LOG_MAX_CHARS",
    _UI_CONFIG.get("tool_log_max_chars"),
    _DEFAULT_LOG_MAX_CHARS,
)

# 思考过程显示配置
SHOW_THINKING_PROCESS = True  # 全局开关

_DEFAULT_STEP_SNIPPET = 500
LOG_STEP_SNIPPET_CHARS = _resolve_int_setting(
    "EXEC_LOG_SNIPPET_CHARS",
    _UI_CONFIG.get("execution_log_snippet_chars"),
    _DEFAULT_STEP_SNIPPET,
)

_DEFAULT_MAX_ITERATIONS = 12
AGENT_MAX_ITERATIONS = _resolve_int_setting(
    "AGENT_MAX_ITERATIONS",
    _AGENT_CONFIG.get("max_iterations"),
    _DEFAULT_MAX_ITERATIONS,
)

# 生成唯一会话ID
SESSION_ID = os.getenv("AGENT_SESSION_ID", f"{MEMORY_CONFIG.session.default_session_prefix}_{uuid.uuid4().hex[:8]}")

# 启用对话记忆的 Agent (使用配置) + Phase 4/5 电商优化
AGENT = LangChainAgent(
    use_memory=None,  # 从配置读取
    session_id=SESSION_ID,
    persist_directory=None,  # 从配置读取
    max_results=None,  # 从配置读取
    use_similarity_search=None,  # 从配置读取
    max_iterations=AGENT_MAX_ITERATIONS,
    enable_conversation_state=True,  # Phase 4: 对话状态跟踪
    enable_system_prompt=True,  # Phase 4: 电商专用提示词
    enable_quality_tracking=True,  # Phase 4: 质量跟踪
    enable_intent_tracking=True,  # Phase 4: 意图识别
    enable_recommendation=True,  # Phase 4: 个性化推荐
)
LOGGER.info("会话ID: %s (后端: %s, 检索模式: %s)", 
           SESSION_ID, MEMORY_CONFIG.backend, MEMORY_CONFIG.strategy.retrieval_mode)
LOGGER.info("Agent max_iterations=%s", AGENT_MAX_ITERATIONS)

# 全局执行日志历史 - 保存所有对话的执行日志
EXECUTION_LOG_HISTORY = []
# 全局对话计数器 - 用于递增编号
CONVERSATION_COUNTER = 0
# 全局 Plan 历史
PLAN_HISTORY = []
# 全局 Tool Call 历史
TOOL_CALL_HISTORY = []


def _render_content_as_html(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            first = lines[0]
            lang = first[3:].strip()
            body = "\n".join(lines[1:-1])
            escaped = html.escape(body)
            lang_attr = f" class=\"language-{lang}\"" if lang else ""
            return f"<pre><code{lang_attr}>{escaped}</code></pre>"
    return f"<pre><code>{html.escape(content)}</code></pre>"


def _collapsible_block(title: str, content: str, *, open_by_default: bool = False) -> str:
    state = " open" if open_by_default else ""
    rendered = _render_content_as_html(content)
    return (
        f"<details class=\"log-details\"{state}>"
        f"<summary>{title}</summary>"
        f"{rendered}"  # html content already escaped
        f"</details>"
    )


def _format_observation_for_ui(observation, *, max_chars: int = LOG_MAX_CHARS):
    """格式化工具/日志结果，尽量保留完整 JSON。"""

    def _truncate(text: str) -> str:
        # 应业务要求，聊天中展示工具结果不得截断
        return text

    if isinstance(observation, (dict, list)):
        pretty = json.dumps(observation, ensure_ascii=False, indent=2)
        return f"```json\n{_truncate(pretty)}\n```", True

    if isinstance(observation, str):
        stripped = observation.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                return _format_observation_for_ui(parsed, max_chars=max_chars)
            except (json.JSONDecodeError, TypeError):
                pass
        return _truncate(observation), False

    return _truncate(repr(observation)), False


def format_tool_log(log_entries) -> str:
    """格式化工具调用日志 - 单次对话"""
    if not log_entries:
        return "(no tool calls yet)"
    lines = []
    for i, e in enumerate(log_entries, 1):
        tool_name = e.get('tool', '')
        
        # 🎯 为本体推理工具添加特殊标识
        if tool_name.startswith('ontology'):
            if 'validate' in tool_name:
                icon = "🛡️"  # SHACL 校验
                tag = "[SHACL校验]"
            elif 'explain' in tool_name:
                icon = "🧠"  # 折扣推理解释
                tag = "[本体推理]"
            elif 'normalize' in tool_name:
                icon = "🔤"  # 商品规范化
                tag = "[本体推理]"
            else:
                icon = "🎯"
                tag = "[本体推理]"
            lines.append(f"**#{i}** {icon} **{tag}** 工具: `{tool_name}`")
        else:
            lines.append(f"**#{i}** 工具: `{tool_name}`")
        
        lines.append(f"  - 输入: `{e.get('input')}`")
        observation = e.get("observation")
        formatted_obs, is_block = _format_observation_for_ui(observation)
        if is_block:
            lines.append("  - 观察:")
            lines.append(_collapsible_block("展开查看观察详情", formatted_obs))
        else:
            lines.append(f"  - 观察: {formatted_obs}")
        lines.append("")
    return "\n".join(lines)


def format_tool_log_history() -> str:
    """格式化累积的工具调用历史 - 最新的在顶部"""
    if not TOOL_CALL_HISTORY:
        return "## 🔧 Tool Calls\n\n> *暂无工具调用记录*"
    
    lines = [f"## 🔧 Tool Calls (共 {len(TOOL_CALL_HISTORY)} 次对话)\n"]
    
    # 统计本体推理调用
    ontology_count = 0
    shacl_count = 0
    for conversation in TOOL_CALL_HISTORY:
        for tool_call in conversation.get("tool_calls", []):
            tool_name = tool_call.get("tool", "")
            if tool_name.startswith("ontology"):
                ontology_count += 1
                if "validate" in tool_name:
                    shacl_count += 1
    
    if ontology_count > 0:
        lines.append(f"🎯 **本体推理调用**: {ontology_count} 次 | 🛡️ **SHACL校验**: {shacl_count} 次\n")
    
    # 遍历历史(已按最新在前排序)
    for conversation in TOOL_CALL_HISTORY:
        conv_num = conversation.get("conversation_num", 0)
        conv_time = conversation.get("start_time", "")
        user_query = conversation.get("user_query", "")
        tool_calls = conversation.get("tool_calls", [])
        
        lines.append(f"### 🔹 对话 #{conv_num} - {conv_time}")
        lines.append(f"**查询**: {user_query}")
        lines.append("")
        
        if not tool_calls:
            lines.append("> *此对话未调用工具*")
        else:
            for i, e in enumerate(tool_calls, 1):
                tool_name = e.get('tool', '')
                
                # 🎯 为本体推理工具添加特殊标识
                if tool_name.startswith('ontology'):
                    if 'validate' in tool_name:
                        icon = "🛡️"
                        tag = "[SHACL校验]"
                    elif 'explain' in tool_name:
                        icon = "🧠"
                        tag = "[本体推理]"
                    elif 'normalize' in tool_name:
                        icon = "🔤"
                        tag = "[本体推理]"
                    else:
                        icon = "🎯"
                        tag = "[本体推理]"
                    lines.append(f"**#{i}** {icon} **{tag}** 工具: `{tool_name}`")
                else:
                    lines.append(f"**#{i}** 工具: `{tool_name}`")
                
                lines.append(f"  - 输入: `{e.get('input')}`")
                observation = e.get("observation")
                formatted_obs, is_block = _format_observation_for_ui(observation)
                if is_block:
                    lines.append("  - 观察:")
                    lines.append(_collapsible_block("展开查看观察详情", formatted_obs))
                else:
                    lines.append(f"  - 观察: {formatted_obs}")
                lines.append("")
        
        lines.append("---\n")  # 对话间分隔线
    
    return "\n".join(lines)


def format_plan_history() -> str:
    """格式化累积的计划历史 - 最新的在顶部"""
    if not PLAN_HISTORY:
        return "## 📋 Plan / Tasks\n\n> *暂无计划记录*"
    
    lines = [f"## 📋 Plan / Tasks (共 {len(PLAN_HISTORY)} 次对话)\n"]
    
    # 遍历历史(已按最新在前排序)
    for conversation in PLAN_HISTORY:
        conv_num = conversation.get("conversation_num", 0)
        conv_time = conversation.get("start_time", "")
        user_query = conversation.get("user_query", "")
        plan = conversation.get("plan", "")
        
        lines.append(f"### 🔹 对话 #{conv_num} - {conv_time}")
        lines.append(f"**查询**: {user_query}")
        lines.append("")
        
        if not plan or plan == "(no plan provided)":
            lines.append("> *此对话无计划*")
        else:
            # 处理plan可能是列表的情况
            if isinstance(plan, list):
                lines.extend(plan)  # 列表中的每个元素都应该是字符串
            else:
                lines.append(str(plan))  # 确保转换为字符串
        
        lines.append("")
        lines.append("---\n")  # 对话间分隔线
    
    return "\n".join(lines)


def format_execution_log(execution_log) -> str:
    """格式化详细执行日志 - 显示所有输入输出的完整结构"""
    if not execution_log:
        return "## 运行日志\n\n(暂无日志)"
    
    lines = [f"## 运行日志 ({len(execution_log)} 条记录)\n"]
    
    for i, entry in enumerate(execution_log, 1):
        step_type = entry.get("step_type", "unknown")
        content = entry.get("content", "")
        metadata = entry.get("metadata", {})
        timestamp = entry.get("timestamp", "")
        
        # 根据步骤类型格式化
        if step_type == "user_input":
            lines.append(f"**[{i}] 📝 用户输入**")
            lines.append(f"```text\n{content}\n```\n")
            
        elif step_type == "memory_retrieval":
            lines.append(f"**[{i}] 🧠 记忆检索**")
            lines.append(f"- 模式: `{metadata.get('mode', 'unknown')}`")
            lines.append(f"- 结果长度: {metadata.get('result_length', 0)} 字符\n")
            
        elif step_type == "memory_context":
            lines.append(f"**[{i}] 📚 历史上下文注入**")
            lines.append(f"```text\n{content}\n```\n")
            
        elif step_type == "enhanced_prompt":
            lines.append(f"**[{i}] 🎯 增强提示词**")
            has_ctx = metadata.get("has_context", False)
            lines.append(f"- 包含历史上下文: {'是' if has_ctx else '否'}")
            if has_ctx:
                lines.append(f"- 上下文长度: {metadata.get('context_length', 0)} 字符")
            lines.append(f"```text\n{content}\n```\n")
            
        elif step_type == "iteration_start":
            iteration = metadata.get("iteration", 0)
            lines.append(f"**[{i}] 🔄 推理轮次 {iteration}**\n")
            
        elif step_type == "llm_input":
            iteration = metadata.get("iteration", 0)
            llm_class = metadata.get("llm_class", "Unknown")
            llm_module = metadata.get("llm_module", "Unknown")
            llm_method = metadata.get("llm_method", "generate")
            
            lines.append(f"**[{i}] 📤 LLM 输入 (第 {iteration} 轮)**")
            lines.append(f"- LLM 类: `{llm_module}.{llm_class}`")
            lines.append(f"- 调用方法: `{llm_method}()`")
            
            if isinstance(content, dict):
                messages = content.get("messages", [])
                tools = content.get("tools", [])
                
                lines.append(f"- 消息数量: {len(messages)}")
                lines.append(f"- 可用工具数: {len(tools)}")
                
                # 显示工具列表 - 修正工具名称提取
                if tools:
                    tool_names = []
                    for t in tools:
                        # OpenAI 函数调用格式: {"type": "function", "function": {"name": "...", ...}}
                        if isinstance(t, dict):
                            if "function" in t and isinstance(t["function"], dict):
                                tool_names.append(t["function"].get("name", "unknown"))
                            else:
                                tool_names.append(t.get("name", "unknown"))
                        else:
                            tool_names.append("unknown")
                    lines.append(f"- 工具列表: `{', '.join(tool_names)}`")
                
                # 显示所有消息
                lines.append(f"\n**消息列表**:")
                for idx, msg in enumerate(messages, 1):
                    role = msg.get("role", "unknown")
                    msg_content = msg.get("content", "")
                    lines.append(f"\n消息 {idx} [{role}]:")
                    if msg_content:
                        lines.append(f"```\n{msg_content}\n```")
                    # 如果有 tool_calls
                    if "tool_calls" in msg:
                        lines.append(f"工具调用: {len(msg['tool_calls'])} 个")
                
                # 显示工具定义
                if tools:
                    lines.append(f"\n**工具定义**:")
                    for tool in tools:
                        lines.append(f"```json\n{json.dumps(tool, ensure_ascii=False, indent=2)}\n```")
            else:
                lines.append(f"```\n{content}\n```")
            lines.append("")
                    
        elif step_type == "llm_output":
            iteration = metadata.get("iteration", 0)
            lines.append(f"**[{i}] 📥 LLM 输出 (第 {iteration} 轮)**")
            
            if isinstance(content, dict):
                response_text = content.get("content", "")
                tool_calls_count = content.get("tool_calls_count", 0)
                tool_calls = content.get("tool_calls", [])
                
                lines.append(f"- 工具调用数: {tool_calls_count}")
                
                if response_text:
                    lines.append(f"\n**模型响应文本**:")
                    lines.append(f"```\n{response_text}\n```")
                
                if tool_calls:
                    lines.append(f"\n**模型请求的工具调用**:")
                    for tc in tool_calls:
                        lines.append(f"```json\n{json.dumps(tc, ensure_ascii=False, indent=2)}\n```")
            else:
                lines.append(f"```\n{content}\n```")
            lines.append("")
                
        elif step_type == "tool_call":
            iteration = metadata.get("iteration", 0)
            tool_class = metadata.get("tool_class", "Unknown")
            tool_module = metadata.get("tool_module", "Unknown")
            
            if isinstance(content, dict):
                tool_name = content.get("name", "unknown")
                args = content.get("arguments", {})
                class_info = content.get("class_info", {})
                
                # 🎯 为本体推理工具添加特殊标识
                if tool_name.startswith('ontology'):
                    if 'validate' in tool_name:
                        icon = "🛡️"
                        tag = "[SHACL校验]"
                    elif 'explain' in tool_name:
                        icon = "🧠"
                        tag = "[本体推理]"
                    elif 'normalize' in tool_name:
                        icon = "🔤"
                        tag = "[本体推理]"
                    else:
                        icon = "🎯"
                        tag = "[本体推理]"
                    lines.append(f"**[{i}] {icon} **{tag}** 执行工具调用 (第 {iteration} 轮)**")
                else:
                    lines.append(f"**[{i}] 🔧 执行工具调用 (第 {iteration} 轮)**")
                
                lines.append(f"- 工具名称: `{tool_name}`")
                lines.append(f"- 工具类: `{class_info.get('module', tool_module)}.{class_info.get('class', tool_class)}`")
                lines.append(f"- 调用方法: `invoke()`")
                lines.append(f"\n**调用参数**:")
                lines.append(f"```json\n{json.dumps(args, ensure_ascii=False, indent=2)}\n```\n")
            else:
                lines.append(f"**[{i}] 🔧 执行工具调用 (第 {iteration} 轮)**")
                lines.append(f"```\n{content}\n```\n")
                
        elif step_type == "tool_result":
            iteration = metadata.get("iteration", 0)
            tool_name = metadata.get("tool_name", "unknown")
            invoked_class = metadata.get("invoked_class", "Unknown")
            invoked_module = metadata.get("invoked_module", "Unknown")
            invoked_method = metadata.get("invoked_method", "invoke")
            
            # 🎯 为本体推理工具结果添加特殊标识
            if tool_name.startswith('ontology'):
                if 'validate' in tool_name:
                    icon = "🛡️"
                    tag = "[SHACL校验]"
                elif 'explain' in tool_name:
                    icon = "🧠"
                    tag = "[本体推理]"
                elif 'normalize' in tool_name:
                    icon = "🔤"
                    tag = "[本体推理]"
                else:
                    icon = "🎯"
                    tag = "[本体推理]"
                lines.append(f"**[{i}] {icon} **{tag}** 工具执行结果 (第 {iteration} 轮)**")
            else:
                lines.append(f"**[{i}] ✅ 工具执行结果 (第 {iteration} 轮)**")
            
            lines.append(f"- 工具名称: `{tool_name}`")
            lines.append(f"- 执行类: `{invoked_module}.{invoked_class}`")
            lines.append(f"- 执行方法: `{invoked_method}()`")
            lines.append(f"\n**返回结果**:")
            formatted_result, is_block = _format_observation_for_ui(content)
            if is_block:
                lines.append(_collapsible_block("展开查看结果", formatted_result))
            else:
                lines.append(f"```text\n{formatted_result}\n```\n")
                
        elif step_type == "final_answer":
            iteration = metadata.get("iteration", 0)
            lines.append(f"**[{i}] 🎉 最终答案 (第 {iteration} 轮)**")
            lines.append(f"```text\n{content}\n```\n")
            
        elif step_type == "memory_save":
            lines.append(f"**[{i}] 💾 保存对话记忆**")
            lines.append(f"- 用户输入: {metadata.get('user_input_length', 0)} 字符")
            lines.append(f"- 助手响应: {metadata.get('response_length', 0)} 字符")
            lines.append(f"- 工具调用数: {metadata.get('tool_calls_count', 0)}\n")
            
        elif step_type == "memory_saved":
            lines.append(f"**[{i}] ✅ 记忆保存成功**")
            lines.append(f"- {content}\n")
            
        elif step_type == "execution_complete":
            lines.append(f"**[{i}] 🏁 执行完成**")
            lines.append(f"- 总迭代次数: {metadata.get('iterations_used', 0)}")
            lines.append(f"- 工具调用次数: {metadata.get('tool_calls', 0)}\n")
            
        elif step_type == "max_iterations":
            lines.append(f"**[{i}] ⚠️ 达到最大迭代限制**")
            lines.append(f"- 限制次数: {metadata.get('max_iterations', 0)}\n")
            
        elif step_type == "ontology_inference":
            lines.append(f"**[{i}] 🧠 本体推理**")
            if isinstance(content, dict):
                inference_type = content.get("inference_type", "unknown")
                lines.append(f"- 推理类型: `{inference_type}`")
                lines.append(f"- 订单ID: {content.get('order_id')}")
                if content.get("order_no"):
                    lines.append(f"- 订单号: {content.get('order_no')}")
                lines.append(f"- 订单状态: {content.get('order_status')}")
                lines.append(f"- 下单后时长: {content.get('hours_since_created')} 小时")
                lines.append(f"- 是否已有物流: {'是' if content.get('has_shipment') else '否'}")
                policy = content.get("policy")
                if policy:
                    lines.append("\n**推理结果**:")
                    policy_text = f"```json\n{json.dumps(policy, ensure_ascii=False, indent=2)}\n```"
                    lines.append(_collapsible_block("展开查看推理结果", policy_text))
            else:
                lines.append(f"```text\n{content}\n```")
            if metadata:
                lines.append(f"- 来源: {metadata.get('source', 'unknown')}")
                if metadata.get("ontology_method"):
                    lines.append(f"- 方法: {metadata['ontology_method']}")
            lines.append("")

        elif step_type == "llm_error":
            llm_class = metadata.get("llm_class", "Unknown")
            llm_module = metadata.get("llm_module", "Unknown")
            lines.append(f"**[{i}] ❌ LLM 调用错误**")
            lines.append(f"- LLM 类: `{llm_module}.{llm_class}`")
            lines.append(f"- 错误类型: `{metadata.get('error_type', 'Unknown')}`")
            lines.append(f"- 错误信息: {metadata.get('error_message', 'No details')}")
            lines.append(f"```\n{content}\n```\n")
        
        else:
            # 未知类型,显示原始内容
            lines.append(f"**[{i}] {step_type}**")
            lines.append(f"```\n{content}\n```\n")
        
        # 时间戳(小字体)
        lines.append(f"<sub>⏱️ {timestamp}</sub>\n")
    
    return "\n".join(lines)


def format_execution_log_history() -> str:
    """格式化累积的执行日志历史 - 最新的在顶部"""
    if not EXECUTION_LOG_HISTORY:
        return "## 📋 执行日志历史\n\n> *暂无执行记录*"
    
    lines = [f"## 📋 执行日志历史 (共 {len(EXECUTION_LOG_HISTORY)} 次对话)\n"]
    
    # 遍历历史(已按最新在前排序,使用存储的对话编号)
    for conversation in EXECUTION_LOG_HISTORY:
        conv_num = conversation.get("conversation_num", 0)  # 使用存储的编号
        conv_logs = conversation.get("logs", [])
        conv_time = conversation.get("start_time", "")
        user_query = conversation.get("user_query", "")
        
        # 对话标题
        lines.append(f"### 🔹 对话 #{conv_num} - {conv_time}")
        lines.append(f"**查询**: {user_query}")
        lines.append("")
        
        # 显示该对话的所有执行步骤
        for i, entry in enumerate(conv_logs, 1):
            step_type = entry.get("step_type", "unknown")
            timestamp = entry.get("timestamp", "")
            content = entry.get("content", "")
            metadata = entry.get("metadata", {})
            
            # 获取图标
            icon_map = {
                "user_input": "📝",
                "memory_retrieval": "🧠",
                "memory_context": "📚",
                "enhanced_prompt": "🎯",
                "iteration_start": "🔄",
                "llm_input": "📤",
                "llm_output": "📥",
                "tool_call": "🔧",
                "tool_result": "✅",
                "final_answer": "🎉",
                "memory_save": "💾",
                "memory_saved": "✅",
                "execution_complete": "🏁",
                "llm_error": "❌",
                "max_iterations": "⚠️",
                "ontology_inference": "🧠",
            }
            icon = icon_map.get(step_type, "📄")
            
            # 步骤标题
            time_short = timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp
            lines.append(f"**步骤 {i}**: {icon} `{step_type}` <sub>{time_short}</sub>")
            
            # 显示关键数据
            if step_type == "user_input":
                lines.append(f"  - 用户输入: {content}")
            
            elif step_type == "memory_retrieval":
                lines.append(f"  - 检索说明: {content}")
                if metadata:
                    mode = metadata.get('mode', '')
                    length = metadata.get('result_length', 0)
                    lines.append(f"  - 模式: `{mode}`, 结果长度: {length}")
            
            elif step_type == "llm_input":
                if isinstance(content, dict):
                    messages = content.get('messages', [])
                    tools = content.get('tools', [])
                    llm_class = metadata.get('llm_class', 'Unknown')
                    llm_module = metadata.get('llm_module', 'Unknown')
                    
                    lines.append(f"  - LLM 类: `{llm_module}.{llm_class}`")
                    lines.append(f"  - 消息数: {len(messages)}, 工具数: {len(tools)}")
                    if tools:
                        # 提取工具名称
                        tool_names = []
                        for t in tools:
                            if isinstance(t, dict):
                                if "function" in t and isinstance(t["function"], dict):
                                    tool_names.append(t["function"].get("name", "unknown"))
                                else:
                                    tool_names.append(t.get("name", "unknown"))
                        lines.append(f"  - 工具列表: {', '.join(tool_names)}")
            
            elif step_type == "llm_output":
                if isinstance(content, dict):
                    text = content.get('content', '')
                    tool_calls = content.get('tool_calls', [])
                    snippet = LOG_STEP_SNIPPET_CHARS
                    lines.append(
                        f"  - 回复: {text[:snippet]}..." if len(text) > snippet else f"  - 回复: {text}"
                    )
                    if tool_calls:
                        call_names = []
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                # tool_calls 格式: [{"id": "...", "name": "...", "arguments": {...}}]
                                name = tc.get('name', 'unknown')
                                call_names.append(name)
                        lines.append(f"  - 调用工具: {', '.join(call_names)}")
                else:
                    snippet = LOG_STEP_SNIPPET_CHARS
                    content_str = str(content)
                    lines.append(
                        f"  - 回复: {content_str[:snippet]}..." if len(content_str) > snippet else f"  - 回复: {content_str}"
                    )
            
            elif step_type == "tool_call":
                if metadata:
                    tool_class = metadata.get('tool_class', '')
                    tool_module = metadata.get('tool_module', '')
                    lines.append(f"  - 工具类: `{tool_module}.{tool_class}`")
                if isinstance(content, dict):
                    tool_name = content.get("name", "unknown")
                    
                    # 🎯 为本体推理工具添加特殊标识
                    if tool_name.startswith('ontology'):
                        if 'validate' in tool_name:
                            icon = "🛡️"
                            tag = "[SHACL校验]"
                        elif 'explain' in tool_name:
                            icon = "🧠"
                            tag = "[本体推理]"
                        elif 'normalize' in tool_name:
                            icon = "🔤"
                            tag = "[本体推理]"
                        else:
                            icon = "🎯"
                            tag = "[本体推理]"
                        lines.append(f"  {icon} **{tag}** 工具名: `{tool_name}`")
                    else:
                        lines.append(f"  - 工具名: `{tool_name}`")
                    
                    snippet = LOG_STEP_SNIPPET_CHARS
                    args_str_full = str(content.get("arguments", {}))
                    args_str = args_str_full[:snippet]
                    lines.append(
                        f"  - 参数: {args_str}..." if len(args_str_full) > snippet else f"  - 参数: {args_str}"
                    )
            
            elif step_type == "tool_result":
                if metadata:
                    tool_name = metadata.get('tool_name', '')
                    invoked_class = metadata.get('invoked_class', '')
                    invoked_module = metadata.get('invoked_module', '')
                    
                    # 🎯 为本体推理工具结果添加特殊标识
                    if tool_name.startswith('ontology'):
                        if 'validate' in tool_name:
                            icon = "🛡️"
                            tag = "[SHACL校验]"
                        elif 'explain' in tool_name:
                            icon = "🧠"
                            tag = "[本体推理]"
                        elif 'normalize' in tool_name:
                            icon = "🔤"
                            tag = "[本体推理]"
                        else:
                            icon = "🎯"
                            tag = "[本体推理]"
                        lines.append(f"  {icon} **{tag}** 工具: `{tool_name}`")
                    else:
                        lines.append(f"  - 工具: `{tool_name}`")
                    
                    lines.append(f"  - 执行类: `{invoked_module}.{invoked_class}`")
                formatted_result, is_block = _format_observation_for_ui(content)
                if is_block:
                    lines.append("  - 结果:")
                    lines.append(_collapsible_block("展开查看结果", formatted_result))
                else:
                    lines.append(f"  - 结果: {formatted_result}")
            
            elif step_type == "final_answer":
                snippet = LOG_STEP_SNIPPET_CHARS
                content_str = str(content)
                lines.append(
                    f"  - 最终回答: {content_str[:snippet]}..." if len(content_str) > snippet else f"  - 最终回答: {content_str}"
                )
            
            elif step_type == "llm_error":
                lines.append(f"  - ❌ 错误: {content}")

            elif step_type == "ontology_inference":
                if isinstance(content, dict):
                    inference_type = content.get('inference_type', 'unknown')
                    lines.append(f"  - 推理类型: `{inference_type}`")
                    lines.append(f"  - 订单ID: {content.get('order_id')}")
                    if content.get('order_no'):
                        lines.append(f"  - 订单号: {content.get('order_no')}")
                    lines.append(f"  - 状态: {content.get('order_status')}")
                    lines.append(f"  - 距下单: {content.get('hours_since_created')} 小时")
                    lines.append(f"  - 有物流: {'是' if content.get('has_shipment') else '否'}")
                    policy = content.get('policy')
                    if policy:
                        snippet = LOG_STEP_SNIPPET_CHARS
                        policy_str = json.dumps(policy, ensure_ascii=False)
                        lines.append(
                            f"  - 推理结果: {policy_str[:snippet]}..." if len(policy_str) > snippet else f"  - 推理结果: {policy_str}"
                        )
                else:
                    lines.append(f"  - 内容: {content}")
                if metadata:
                    lines.append(f"  - 来源: {metadata.get('source', 'unknown')}")
                    if metadata.get('ontology_method'):
                        lines.append(f"  - 方法: {metadata['ontology_method']}")
            
            lines.append("")
        
        lines.append("---\n")  # 对话间分隔线
    
    return "\n".join(lines)


def format_memory_context() -> str:
    """格式化对话记忆上下文(Markdown)"""
    context = AGENT.get_memory_context()
    stats = AGENT.get_memory_stats()
    config = MEMORY_CONFIG
    
    if not context:
        lines = []
        lines.append(f"**后端**: `{stats.get('backend', 'Unknown')}`  ")
        lines.append(f"**检索模式**: `{config.strategy.retrieval_mode}`  ")
        if 'persist_directory' in stats:
            persist_dir = stats['persist_directory']
            # 只显示最后两级目录
            short_dir = '/'.join(persist_dir.split('/')[-2:])
            lines.append(f"**存储**: `.../{short_dir}`  ")
        lines.append(f"**会话**: `{stats.get('session_id', 'N/A')[:12]}...`  ")
        lines.append(f"**记录数**: `{stats.get('total_turns', 0)}`  ")
        max_results = config.strategy.max_recent_turns if config.strategy.retrieval_mode == 'recent' else config.strategy.max_similarity_results
        lines.append(f"**最大结果**: `{max_results}`  ")
        lines.append("")
        lines.append("> *暂无历史记录*")
        return "\n".join(lines)
    
    lines = []
    lines.append(f"**后端**: `{stats.get('backend', 'Unknown')}` | **模式**: `{config.strategy.retrieval_mode}` | **记录数**: `{stats.get('total_turns', 0)}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(context)
    
    return "\n".join(lines)


def format_ecommerce_analysis() -> str:
    """格式化电商分析数据（质量、意图、推荐）"""
    lines = ["## 🛍️ 电商智能分析\n"]
    
    # 1. 对话质量评分
    quality_report = AGENT.get_quality_report()
    if quality_report:
        lines.append("### 📊 对话质量评分\n")
        score = quality_report.get('quality_score', 0)
        
        # 评级
        if score >= 80:
            grade = "🏆 优秀"
        elif score >= 60:
            grade = "✅ 良好"
        elif score >= 40:
            grade = "⚠️ 及格"
        else:
            grade = "❌ 待改进"
        
        lines.append(f"**综合评分**: {score}/100 {grade}\n")
        
        efficiency = quality_report.get('efficiency', {})
        completion = quality_report.get('task_completion', {})
        conv_quality = quality_report.get('conversation_quality', {})
        
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 平均响应时间 | {efficiency.get('avg_response_time', 0):.2f}秒 |")
        lines.append(f"| 平均工具调用 | {efficiency.get('avg_tool_calls', 0):.2f}次 |")
        lines.append(f"| 任务成功率 | {completion.get('success_rate', 0)*100:.1f}% |")
        lines.append(f"| 澄清率 | {conv_quality.get('clarification_rate', 0)*100:.1f}% |")
        lines.append(f"| 主动引导率 | {conv_quality.get('proactive_rate', 0)*100:.1f}% |")
        lines.append("")
    
    # 2. 意图分析
    intent_analysis = AGENT.get_intent_analysis()
    if intent_analysis and intent_analysis.get('total_turns', 0) > 0:
        lines.append("### 🎯 意图分析\n")
        
        lines.append(f"**总轮次**: {intent_analysis.get('total_turns', 0)}\n")
        
        # 意图分布
        intent_dist = intent_analysis.get('intent_distribution', {})
        if intent_dist:
            lines.append("**意图分布**:\n")
            for intent_type, count in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / intent_analysis['total_turns'] * 100) if intent_analysis['total_turns'] > 0 else 0
                bar = "█" * int(percentage / 5)  # 每5%一个方块
                lines.append(f"- `{intent_type}`: {bar} {count}次 ({percentage:.0f}%)")
            lines.append("")
        
        # 复合意图
        composite = intent_analysis.get('composite_intents', [])
        if composite:
            lines.append("**复合意图**:\n")
            for comp in composite:
                name = comp.get('name', 'Unknown')
                desc = comp.get('description', '')
                confidence = comp.get('confidence', 0)
                
                if name == "purchase_intent":
                    icon = "💰"
                elif name == "comparison_intent":
                    icon = "🔄"
                elif name == "after_sales_intent":
                    icon = "📞"
                else:
                    icon = "📍"
                
                lines.append(f"- {icon} **{name}**: {desc} (置信度: {confidence:.2f})")
            lines.append("")
        
        # 当前状态
        current = intent_analysis.get('current_intent', '')
        predicted = intent_analysis.get('predicted_next', [])
        if current:
            lines.append(f"**当前意图**: `{current}`")
        if predicted:
            lines.append(f"**预测下一步**: `{', '.join(predicted)}`")
        lines.append("")
    
    # 3. 对话状态
    if hasattr(AGENT, 'state_manager') and AGENT.state_manager and AGENT.state_manager.state:
        lines.append("### 🛒 对话状态\n")
        state = AGENT.state_manager.state
        
        stage_emoji = {
            "greeting": "👋",
            "browsing": "🔍",
            "selecting": "👀",
            "cart": "🛒",
            "checkout": "💳",
            "tracking": "📦",
            "service": "💬",
            "idle": "💤"
        }
        
        stage = state.stage.value
        lines.append(f"**当前阶段**: {stage_emoji.get(stage, '📍')} {stage}\n")
        
        # 用户上下文
        user_ctx = state.user_context
        lines.append("**用户信息**:")
        lines.append(f"- 用户ID: {user_ctx.user_id or '未登录'}")
        lines.append(f"- VIP状态: {'是 ⭐' if user_ctx.is_vip else '否'}")
        lines.append(f"- 购物车: {user_ctx.cart_item_count} 件")
        
        if user_ctx.last_viewed_products:
            viewed = ', '.join(str(p) for p in user_ctx.last_viewed_products[:3])
            lines.append(f"- 最近浏览: {viewed}")
        
        if user_ctx.recent_order_id:
            lines.append(f"- 最近订单: {user_ctx.recent_order_id}")
        
        lines.append("")
    
    # 如果没有任何数据
    if len(lines) == 1:
        lines.append("> *开始对话后显示分析数据*")
    
    return "\n".join(lines)


def _filter_charts_by_intent(charts: List[Dict[str, Any]], user_input: str, agent_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """根据意图和上下文过滤图表
    
    Args:
        charts: 所有生成的图表
        user_input: 用户输入
        agent_result: Agent执行结果
        
    Returns:
        过滤后的图表列表
    """
    if not charts:
        return []
    
    # 获取意图信息
    intent_summary = agent_result.get("intent_summary", {})
    current_intent = intent_summary.get("current_intent", "")
    
    # 从用户输入中提取用户ID关键词
    user_id_keywords = ["我的", "个人", "用户", "user_id", "用户id"]
    mentions_personal = any(keyword in user_input.lower() for keyword in user_id_keywords)
    
    LOGGER.info("图表过滤: 意图=%s, 提到个人数据=%s, 图表总数=%d", current_intent, mentions_personal, len(charts))
    
    filtered = []
    for chart in charts:
        metadata = chart.get("metadata", {})
        chart_user_id = metadata.get("requested_user_id")
        chart_title = chart.get("title", "未命名图表")
        
        # 规则: 如果用户明确提到个人数据，只保留有用户ID的图表
        if mentions_personal:
            if chart_user_id is not None:
                filtered.append(chart)
                LOGGER.info("✓ 保留个人图表: %s (user_id=%s)", chart_title, chart_user_id)
            else:
                LOGGER.info("✗ 过滤全局图表: %s (用户请求个人数据)", chart_title)
        else:
            # 用户未特指个人，保留所有Agent生成的图表
            filtered.append(chart)
            LOGGER.info("✓ 保留图表: %s (通用查询)", chart_title)
    
    LOGGER.info("图表过滤结果: 保留 %d/%d 个图表", len(filtered), len(charts))
    return filtered


def _normalize_chatbot_messages(history):
    normalized = []
    if not history:
        return normalized
    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role") or ("assistant" if normalized and normalized[-1]["role"] == "user" else "user")
            content = entry.get("content")
            normalized.append({
                "role": role,
                "content": "" if content is None else str(content),
            })
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            user_msg, assistant_msg = entry
            if user_msg is not None:
                normalized.append({"role": "user", "content": str(user_msg)})
            if assistant_msg is not None:
                normalized.append({"role": "assistant", "content": str(assistant_msg)})
        else:
            continue
    return normalized


def handle_user_message(user_message, chat_history=None, show_thinking=True):
    """处理用户消息并更新 UI（支持流式思考过程展示）
    
    Args:
        user_message: 用户输入的消息
        chat_history: 历史对话记录
        show_thinking: 是否显示思考过程
    
    Yields:
        tuple: 包含所有UI组件更新的元组，用于流式更新
    """
    from datetime import datetime
    global CONVERSATION_COUNTER, PLAN_HISTORY, TOOL_CALL_HISTORY
    
    chat_history = _normalize_chatbot_messages(chat_history)
    chat_history.append({"role": "user", "content": user_message})
    assistant_placeholder = {"role": "assistant", "content": ""}
    chat_history.append(assistant_placeholder)

    LOGGER.info("Gradio incoming: %s", user_message[:200])
    chart_figures: List[Any] = []

    try:
        # 初始化结果变量
        thinking_steps = []
        res = None
        final = ""
        plan = ""
        tool_calls = []
        charts = []
        
        # 如果启用思考过程，使用流式方法
        if show_thinking:
            for step in AGENT.run_stream(user_message):
                step_type = step.get("step_type")
                content = step.get("content", "")
                metadata = step.get("metadata", {})
                
                if step_type == "thinking_start":
                    thinking_steps = ["🤔 **正在分析您的需求...**"]
                    assistant_placeholder["content"] = "\n".join(thinking_steps)
                    yield (
                        gr.update(value=chat_history),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                
                elif step_type == "intent_recognized":
                    thinking_steps.append(f"\n🎯 **{content}**")
                    assistant_placeholder["content"] = "\n".join(thinking_steps)
                    yield (
                        gr.update(value=chat_history),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                
                elif step_type == "query_rewritten":
                    keywords = metadata.get("keywords", [])
                    if keywords:
                        thinking_steps.append(f"\n📝 **优化查询**: 关键词=[{', '.join(keywords)}]")
                        assistant_placeholder["content"] = "\n".join(thinking_steps)
                        yield (
                            gr.update(value=chat_history),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )
                
                elif step_type == "tool_calling_start":
                    thinking_steps.append(f"\n🔧 **{content}**")
                    assistant_placeholder["content"] = "\n".join(thinking_steps)
                    yield (
                        gr.update(value=chat_history),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )

                elif step_type == "tool_calling":
                    tool_name = metadata.get("tool", "工具")
                    args_preview = metadata.get("arguments")
                    if isinstance(args_preview, dict):
                        arg_text = json.dumps(args_preview, ensure_ascii=False)
                        if len(arg_text) > 80:
                            arg_text = arg_text[:80] + "..."
                    elif args_preview is None:
                        arg_text = ""
                    else:
                        arg_text = str(args_preview)[:80]
                    detail = f"参数: {arg_text}" if arg_text else ""
                    thinking_steps.append(f"\n   ↳ 调用 `{tool_name}` {detail}")
                    assistant_placeholder["content"] = "\n".join(thinking_steps)
                    yield (
                        gr.update(value=chat_history),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                
                elif step_type == "tool_results":
                    tool_names = metadata.get("tool_names", [])
                    if tool_names:
                        thinking_steps.append(f"\n✅ **已完成工具调用**: {len(tool_names)} 个工具")
                        for i, name in enumerate(tool_names[:3], 1):
                            thinking_steps.append(f"   {i}. `{name}`")
                        if len(tool_names) > 3:
                            thinking_steps.append(f"   ... 还有 {len(tool_names) - 3} 个")
                        assistant_placeholder["content"] = "\n".join(thinking_steps)
                        yield (
                            gr.update(value=chat_history),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )

                elif step_type == "tool_result":
                    tool_name = metadata.get("tool", "工具")
                    preview = metadata.get("observation_preview") or content
                    if isinstance(preview, str) and len(preview) > 120:
                        preview = preview[:120] + "..."
                    thinking_steps.append(f"\n📥 `{tool_name}` 返回: {preview}")
                    assistant_placeholder["content"] = "\n".join(thinking_steps)
                    yield (
                        gr.update(value=chat_history),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                
                elif step_type == "llm_streaming_start":
                    thinking_steps.append(f"\n💬 **{content}**")
                    assistant_placeholder["content"] = "\n".join(thinking_steps)
                    yield (
                        gr.update(value=chat_history),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                
                elif step_type == "llm_streaming":
                    # 逐 token 流式输出（追加到思考步骤之后）
                    accumulated = metadata.get("accumulated", "")
                    # 清除之前的思考步骤，只保留累积的答案
                    assistant_placeholder["content"] = accumulated
                    yield (
                        gr.update(value=chat_history),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )

                elif step_type == "error":
                    thinking_steps.append(f"\n⚠️ **{content}**")
                    assistant_placeholder["content"] = "\n".join(thinking_steps)
                    yield (
                        gr.update(value=chat_history),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                
                elif step_type == "final_answer":
                    # 获取完整结果
                    res = metadata.get("full_result", {})
                    final = content
                    plan = res.get("plan", "(no plan provided)")
                    tool_calls = res.get("tool_log", [])
                    charts = res.get("charts", [])
        else:
            # 不显示思考过程，直接运行
            res = AGENT.run(user_message)
            final = res.get("final_answer") or ""
            plan = res.get("plan") or "(no plan provided)"
            tool_calls = res.get("tool_log", [])
            charts = res.get("charts", [])
        
        LOGGER.info("handle_user_message: 收到 %d 个图表对象", len(charts))
        
        # 根据意图和上下文过滤图表
        filtered_charts = _filter_charts_by_intent(charts, user_message, res)
        if len(filtered_charts) < len(charts):
            LOGGER.info("图表过滤: %d -> %d (移除了不相关的图表)", len(charts), len(filtered_charts))
        charts = filtered_charts
        
        # Phase 4/5: 获取电商增强信息
        intent_summary = res.get("intent_summary", {})
        quality_metrics = res.get("quality_metrics", {})
        conv_state = res.get("conversation_state", {})
        
        # 在回复中添加电商上下文提示（如果有）
        ecommerce_context = []
        
        # 1. 显示当前购物阶段
        if conv_state:
            stage = conv_state.get("stage", "unknown")
            stage_emoji = {
                "greeting": "👋",
                "browsing": "🔍",
                "selecting": "👀",
                "cart": "🛒",
                "checkout": "💳",
                "tracking": "📦",
                "service": "💬",
                "idle": "💤"
            }
            stage_text = {
                "greeting": "问候",
                "browsing": "浏览商品",
                "selecting": "选择商品",
                "cart": "购物车",
                "checkout": "结账",
                "tracking": "订单跟踪",
                "service": "客服咨询",
                "idle": "空闲"
            }
            ecommerce_context.append(f"{stage_emoji.get(stage, '📍')} 当前阶段: {stage_text.get(stage, stage)}")
        
        # 2. 显示意图信息
        if intent_summary and intent_summary.get("current_intent"):
            current_intent = intent_summary["current_intent"]
            ecommerce_context.append(f"🎯 识别意图: {current_intent}")
            
            # 显示预测的下一步
            predicted = intent_summary.get("predicted_next", [])
            if predicted:
                ecommerce_context.append(f"🔮 建议操作: {', '.join(predicted[:2])}")
        
        # 3. 显示复合意图（购买意向等）
        if intent_summary and intent_summary.get("composite_intents"):
            for comp in intent_summary["composite_intents"]:
                comp_name = comp.get("name", "")
                if comp_name == "purchase_intent":
                    ecommerce_context.append("💡 检测到购买意向")
                elif comp_name == "comparison_intent":
                    ecommerce_context.append("🔄 正在比较商品")
                elif comp_name == "after_sales_intent":
                    ecommerce_context.append("📞 关注售后服务")
        
        # 将电商上下文添加到回复中（使用引用块）
        if ecommerce_context:
            context_text = "\n".join([f"> {line}" for line in ecommerce_context])
            final = f"{final}\n\n---\n**智能助手状态**\n{context_text}"
        
        # 更新助手消息内容（清除思考过程，只显示最终答案）
        assistant_placeholder["content"] = final
        
        # 将图表转换为图片并添加到聊天历史中
        if charts:
            chart_markdown_messages = _render_charts_as_images(charts)
            # 将图片 Markdown 合并到回复内容中
            if chart_markdown_messages:
                combined_charts = "".join(chart_markdown_messages)
                final = f"{final}\n{combined_charts}"
                LOGGER.info("已将 %d 个图表（Markdown 格式）嵌入到回复内容中", len(chart_markdown_messages))
                # 更新助手消息内容
                assistant_placeholder["content"] = final
        
        LOGGER.info("图表已转换为 Markdown 图片并嵌入到聊天消息中")
        
        # 递增对话计数器
        CONVERSATION_COUNTER += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 累积执行日志到历史 (最新的插入到列表开头)
        current_execution_log = res.get("execution_log", [])
        if current_execution_log:
            EXECUTION_LOG_HISTORY.insert(0, {
                "conversation_num": CONVERSATION_COUNTER,
                "logs": current_execution_log,
                "start_time": current_time,
                "user_query": user_message
            })
        
        # 累积 Plan 历史
        PLAN_HISTORY.insert(0, {
            "conversation_num": CONVERSATION_COUNTER,
            "plan": plan,
            "start_time": current_time,
            "user_query": user_message
        })
        
        # 累积 Tool Call 历史
        TOOL_CALL_HISTORY.insert(0, {
            "conversation_num": CONVERSATION_COUNTER,
            "tool_calls": tool_calls,
            "start_time": current_time,
            "user_query": user_message
        })
        
        # 格式化历史
        execution_log = format_execution_log_history()
        plan_display = format_plan_history()
        tool_display = format_tool_log_history()
        ecommerce_display = format_ecommerce_analysis()
        
        # 检查是否有错误
        if res.get("error"):
            LOGGER.error(f"Agent 执行出错: {res['error']}")
    except Exception as e:
        error_msg = f"处理请求时发生错误: {type(e).__name__}: {str(e)}"
        LOGGER.error(error_msg, exc_info=True)
        final = f"❌ {error_msg}"
        assistant_placeholder["content"] = final
        plan_display = f"## 📋 Plan / Tasks\n\n**执行失败**: {error_msg}"
        tool_display = f"## 🔧 Tool Calls\n\n**执行失败**: {error_msg}"
        execution_log = f"## 📋 执行日志历史\n\n**执行失败**: {error_msg}"
        ecommerce_display = f"## 🛍️ 电商分析\n\n**执行失败**: {error_msg}"
    memory_md = format_memory_context()
    
    # 最终返回完整结果
    yield (
        gr.update(value=chat_history),
        gr.update(value=plan_display),
        gr.update(value=tool_display),
        gr.update(value=memory_md),
        gr.update(value=ecommerce_display),
        gr.update(value=execution_log),
    )


def _convert_plotly_to_image(fig: go.Figure, width: int = 800, height: int = 500) -> Optional[str]:
    """将 Plotly 图表转换为 base64 编码的图片字符串
    
    Args:
        fig: Plotly Figure 对象
        width: 图片宽度
        height: 图片高度
    
    Returns:
        base64 编码的 Markdown 图片，如果转换失败则返回 None
    """
    try:
        # 转换为 PNG 图片（bytes）
        img_bytes = pio.to_image(fig, format='png', width=width, height=height, engine='kaleido')
        
        # 转换为 base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # 返回 Markdown 图片语法（data URL）
        data_url = f"data:image/png;base64,{img_base64}"
        
        LOGGER.info("图表已转换为 base64 图片，大小: %d bytes", len(img_bytes))
        return data_url
    except Exception as e:
        LOGGER.error("图表转换为图片失败: %s", str(e))
        return None


def _render_charts_as_images(charts: List[Dict[str, Any]]) -> List[str]:
    """将图表数据渲染为图片，返回适用于 Gradio Chatbot 的消息格式
    
    Returns:
        包含图片 HTML 的消息列表
    """
    if not charts:
        return []
    
    chart_messages = []
    
    for idx, chart in enumerate(charts, 1):
        title = chart.get("title", "图表")
        description = chart.get("description", "")
        
        # 构建 Plotly 图表
        payload = _build_plotly_payload(chart)
        if not payload:
            LOGGER.warning("无法生成图表 Plotly payload: %s", title)
            continue
        
        figure = go.Figure()
        figure.update_layout(**payload["layout"])
        
        for trace in payload["data"]:
            trace_type = (trace.get("type") or "").lower()
            if trace_type == "pie":
                figure.add_trace(
                    go.Pie(
                        labels=trace.get("labels", []),
                        values=trace.get("values", []),
                        textinfo=trace.get("textinfo") or "label+percent",
                        hole=trace.get("hole"),
                    )
                )
            elif trace_type == "scatter":
                figure.add_trace(
                    go.Scatter(
                        x=trace.get("x", []),
                        y=trace.get("y", []),
                        mode=trace.get("mode", "lines+markers"),
                        name=trace.get("name"),
                    )
                )
            elif trace_type == "bar":
                figure.add_trace(
                    go.Bar(
                        x=trace.get("x", []),
                        y=trace.get("y", []),
                        name=trace.get("name"),
                    )
                )
        
        # 转换为图片
        img_data_url = _convert_plotly_to_image(figure)
        if img_data_url:
            # 构建包含标题和图片的 Markdown 格式
            chart_md = f"\n\n---\n### 📊 {title}\n"
            if description:
                chart_md += f"*{description}*\n\n"
            # 使用 Markdown 图片语法
            chart_md += f"![{title}]({img_data_url})\n"
            
            chart_messages.append(chart_md)
            LOGGER.info("✓ 图表转换为 Markdown 图片成功: %s", title)
        else:
            LOGGER.warning("✗ 图表转换为图片失败: %s", title)
    
    return chart_messages


def _render_charts_markdown(charts: List[Dict[str, Any]]) -> str:
    """将图表数据渲染为文本表格，方便展示在聊天记录中（备用方案）"""
    if not charts:
        return ""
    
    html_parts = ["\n\n---\n## 📊 数据可视化\n"]
    
    for idx, chart in enumerate(charts, 1):
        chart_type = chart.get("chart_type", "unknown")
        title = chart.get("title", "图表")
        labels = chart.get("labels", [])
        series = chart.get("series", [])
        description = chart.get("description", "")
        
        html_parts.append(f"\n### {idx}. {title}\n")
        if description:
            html_parts.append(f"> {description}\n")
        
        # 渲染为Markdown表格
        if labels and series:
            # 表头
            header = ["项目"] + [s.get("name", f"系列{i}") for i, s in enumerate(series, 1)]
            html_parts.append("\n| " + " | ".join(header) + " |")
            html_parts.append("| " + " | ".join(["---"] * len(header)) + " |")
            
            # 数据行
            for i, label in enumerate(labels):
                row = [str(label)]
                for s in series:
                    data_list = s.get("data", [])
                    value = data_list[i] if i < len(data_list) else "-"
                    row.append(str(value))
                html_parts.append("| " + " | ".join(row) + " |")
            html_parts.append("")
        
        html_parts.append(f"\n*图表类型: {chart_type}*\n")
    
    return "\n".join(html_parts)


def _build_plotly_payload(chart: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    chart_type = (chart.get("chart_type") or "").lower()
    labels = chart.get("labels") or []
    series = chart.get("series") or []
    if not labels or not series:
        return None

    layout = {
        "title": chart.get("title", "图表"),
        "margin": {"l": 40, "r": 10, "t": 60, "b": 40},
        "legend": {"orientation": "h"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
    }
    data = []

    if chart_type == "pie":
        values = series[0].get("data") if series else []
        if not values:
            return None
        data.append(
            {
                "type": "pie",
                "labels": labels,
                "values": values,
                "textinfo": "label+percent",
            }
        )
        layout["showlegend"] = False
    elif chart_type == "trend":
        for item in series:
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": item.get("name"),
                    "x": labels,
                    "y": item.get("data", []),
                }
            )
    else:
        if chart_type == "comparison":
            layout["barmode"] = "group"
        for item in series:
            data.append(
                {
                    "type": "bar",
                    "name": item.get("name"),
                    "x": labels,
                    "y": item.get("data", []),
                }
            )

    return {"data": data, "layout": layout}


def _build_plotly_figures(charts: List[Dict[str, Any]]) -> List[Any]:
    if not charts:
        LOGGER.info("_build_plotly_figures: 无图表")
        return []

    figures: List[Any] = []
    for chart in charts:
        LOGGER.info(
            "_build_plotly_figures: 准备渲染 chart_type=%s title=%s",
            chart.get("chart_type"),
            chart.get("title"),
        )
        payload = _build_plotly_payload(chart)
        if not payload:
            LOGGER.warning(
                "_build_plotly_figures: 无法生成 Plotly payload，chart=%s",
                json.dumps(chart, ensure_ascii=False)[:200],
            )
            continue
        figure = go.Figure()
        figure.update_layout(**payload["layout"])
        for trace in payload["data"]:
            trace_type = (trace.get("type") or "").lower()
            if trace_type == "pie":
                figure.add_trace(
                    go.Pie(
                        labels=trace.get("labels", []),
                        values=trace.get("values", []),
                        textinfo=trace.get("textinfo") or "label+percent",
                        hole=trace.get("hole"),
                    )
                )
            elif trace_type == "scatter":
                figure.add_trace(
                    go.Scatter(
                        x=trace.get("x", []),
                        y=trace.get("y", []),
                        mode=trace.get("mode", "lines+markers"),
                        name=trace.get("name"),
                    )
                )
            elif trace_type == "bar":
                figure.add_trace(
                    go.Bar(
                        x=trace.get("x", []),
                        y=trace.get("y", []),
                        name=trace.get("name"),
                    )
                )
            else:
                figure.add_trace(trace)
        figures.append(figure)
    return figures


def clear_conversation():
    """清空对话历史和记忆"""
    global EXECUTION_LOG_HISTORY, CONVERSATION_COUNTER, PLAN_HISTORY, TOOL_CALL_HISTORY
    
    AGENT.clear_memory()
    EXECUTION_LOG_HISTORY.clear()  # 清空执行日志历史
    PLAN_HISTORY.clear()  # 清空计划历史
    TOOL_CALL_HISTORY.clear()  # 清空工具调用历史
    CONVERSATION_COUNTER = 0  # 重置对话计数器
    LOGGER.info("对话历史、计划、工具调用和执行日志已清空")
    
    empty_memory = format_memory_context()
    empty_plan = "## 📋 Plan / Tasks\n\n> *暂无计划记录*"
    empty_tool = "## 🔧 Tool Calls\n\n> *暂无工具调用记录*"
    empty_ecommerce = "## 🛍️ 电商智能分析\n\n> *开始对话后显示分析数据*"
    empty_log = "## 📋 执行日志历史\n\n> *暂无执行记录*"
    return [], empty_plan, empty_tool, empty_memory, empty_ecommerce, empty_log


with gr.Blocks(
    title="Agent 运行日志 UI",
    css="""
    /* 允许页面高度超过首屏并启用滚动 */
    html, body {
        height: auto;
        min-height: 100vh;
        overflow-y: auto !important;
    }
    .gradio-container {
        min-height: 100vh;
        height: auto;
        display: flex;
        flex-direction: column;
    }
    .main-layout-row {
        gap: 16px;
    }
    .left-panel,
    .right-panel {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .tab-content {
        padding: 10px;
    }
    .quick-phrase-btn {
        font-size: 12px !important;
        padding: 4px 8px !important;
        min-width: 80px !important;
    }
    details.log-details {
        margin: 6px 0 10px 12px;
        padding: 6px 10px;
        border-left: 2px solid #d0d7de;
        background: rgba(0, 0, 0, 0.02);
    }
    details.log-details summary {
        cursor: pointer;
        font-weight: 600;
        color: #1f6feb;
    }
    details.log-details[open] {
        background: rgba(0, 0, 0, 0.04);
    }
    """
) as demo:
    gr.Markdown(f"# Ontology RL Commerce Agent \n**ChromaDB 持久化记忆** (会话: `{SESSION_ID[:12]}...`)")
    
    with gr.Row(equal_height=False, elem_classes="main-layout-row"):
        # 左侧: 聊天区域
        with gr.Column(scale=3, elem_classes="left-panel"):
            chatbot = gr.Chatbot(elem_id="mcp_chat", label="对话历史", height=600, type="messages")
            
            # 🎯 便捷测试短语区域 - 10个快捷按钮
            gr.Markdown("### 🚀 快捷测试短语（点击即可提问）")
            with gr.Row():
                quick_btn1 = gr.Button("📊 查询用户等级", elem_classes="quick-phrase-btn", size="sm")
                quick_btn2 = gr.Button("🎁 查询可用折扣", elem_classes="quick-phrase-btn", size="sm")
                quick_btn3 = gr.Button("🚚 查询物流方案", elem_classes="quick-phrase-btn", size="sm")
                quick_btn4 = gr.Button("↩️ 查询退货政策", elem_classes="quick-phrase-btn", size="sm")
                quick_btn5 = gr.Button("📱 搜索iPhone", elem_classes="quick-phrase-btn", size="sm")
            with gr.Row():
                quick_btn6 = gr.Button("🛒 创建测试订单", elem_classes="quick-phrase-btn", size="sm")
                quick_btn7 = gr.Button("🔍 商品规范化", elem_classes="quick-phrase-btn", size="sm")
                quick_btn8 = gr.Button("🛡️ SHACL校验", elem_classes="quick-phrase-btn", size="sm")
                quick_btn9 = gr.Button("🧠 完整推理流程", elem_classes="quick-phrase-btn", size="sm")
                quick_btn10 = gr.Button("📈 用户消费分析", elem_classes="quick-phrase-btn", size="sm")
            
            # 🧠 思考过程开关
            with gr.Row():
                show_thinking_checkbox = gr.Checkbox(
                    label="💭 显示思考过程",
                    value=True,
                    info="勾选后将实时显示 AI 的分析和工具调用过程"
                )
                iteration_slider = gr.Slider(
                    minimum=4,
                    maximum=20,
                    step=1,
                    value=AGENT_MAX_ITERATIONS,
                    label="最大推理轮次",
                    info="调节单次对话的 LLM 推理迭代上限，以便在界面中快速测试不同效果"
                )
            
            with gr.Row():
                txt = gr.Textbox(show_label=False, placeholder="在这里输入你的请求", lines=2, scale=4)
                clear_btn = gr.Button("清空对话", variant="secondary", scale=1)
            submit = gr.Button("发送", variant="primary")
            
        # 右侧: Tab 页切换 (包含所有辅助信息)
        with gr.Column(scale=2, elem_classes="right-panel"):
            with gr.Tabs(elem_classes="right-panel-scroll"):
                with gr.TabItem("📋 Plan / Tasks"):
                    with gr.Accordion("Plan 详情", open=True):
                        plan_md = gr.Markdown("## 📋 Plan / Tasks\n\n> *暂无计划记录*", elem_id="plan_panel", elem_classes="tab-content")
                
                with gr.TabItem("🔧 Tool Calls"):
                    with gr.Accordion("展开/折叠工具调用", open=False):
                        tool_md = gr.Markdown("## 🔧 Tool Calls\n\n> *暂无工具调用记录*", elem_id="tool_panel", elem_classes="tab-content")
                
                with gr.TabItem("💾 Memory"):
                    with gr.Accordion("展开/折叠记忆上下文", open=False):
                        memory_md = gr.Markdown(
                            value=format_memory_context(), 
                            elem_id="memory_panel",
                            elem_classes="tab-content"
                        )
                
                with gr.TabItem("�️ 电商分析"):
                    with gr.Accordion("展开/折叠电商分析", open=False):
                        ecommerce_md = gr.Markdown(
                            "## 🛍️ 电商智能分析\n\n> *开始对话后显示分析数据*",
                            elem_id="ecommerce_panel",
                            elem_classes="tab-content"
                        )

                with gr.TabItem("📊 Execution Log"):
                    with gr.Accordion("展开/折叠执行日志", open=False):
                        execution_log_md = gr.Markdown(
                            "## 📋 执行日志历史\n\n> *暂无执行记录*", 
                            elem_id="execution_log_panel",
                            elem_classes="tab-content"
                        )

    def submit_and_update(message, history, show_thinking, max_iterations):
        """提交消息并更新所有面板 - 先显示用户消息，再获取回复
        
        Args:
            message: 用户输入的消息
            history: 聊天历史
            show_thinking: 是否显示思考过程
        """
        # 第一步：立即显示用户消息（Assistant回复为"思考中..."）并禁用所有按钮
        base_history = _normalize_chatbot_messages(history)
        try:
            iteration_limit = int(max_iterations)
        except (TypeError, ValueError):
            iteration_limit = AGENT_MAX_ITERATIONS
        iteration_limit = max(1, iteration_limit)
        if AGENT.max_iterations != iteration_limit:
            LOGGER.info("临时调整 max_iterations=%s (原=%s)", iteration_limit, AGENT.max_iterations)
        AGENT.max_iterations = iteration_limit
        pending_history = base_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⏳ 正在思考..."},
        ]
        
        # 立即返回更新（禁用按钮，防止重复提交）
        yield (
            gr.update(value=pending_history),  # chatbot
            gr.update(),  # plan_md (保持不变)
            gr.update(),  # tool_md (保持不变)
            gr.update(),  # memory_md (保持不变)
            gr.update(),  # ecommerce_md (保持不变)
            gr.update(),  # execution_log_md (保持不变)
            gr.update(value="", interactive=False),  # 清空输入框并禁用
            gr.update(interactive=False),  # 禁用发送按钮
            gr.update(interactive=False),  # 禁用快捷按钮1
            gr.update(interactive=False),  # 禁用快捷按钮2
            gr.update(interactive=False),  # 禁用快捷按钮3
            gr.update(interactive=False),  # 禁用快捷按钮4
            gr.update(interactive=False),  # 禁用快捷按钮5
            gr.update(interactive=False),  # 禁用快捷按钮6
            gr.update(interactive=False),  # 禁用快捷按钮7
            gr.update(interactive=False),  # 禁用快捷按钮8
            gr.update(interactive=False),  # 禁用快捷按钮9
            gr.update(interactive=False),  # 禁用快捷按钮10
        )
        
        # 第二步：调用后端获取真实回复（流式生成）
        for result in handle_user_message(message, base_history, show_thinking):
            yield (
                result[0],  # chatbot (包含思考过程或最终回复)
                result[1],  # plan_md
                result[2],  # tool_md
                result[3],  # memory_md
                result[4],  # ecommerce_md
                result[5],  # execution_log_md
                gr.update(value="", interactive=True),  # 启用输入框
                gr.update(interactive=True),  # 启用发送按钮
                gr.update(interactive=True),  # 启用快捷按钮1
                gr.update(interactive=True),  # 启用快捷按钮2
                gr.update(interactive=True),  # 启用快捷按钮3
                gr.update(interactive=True),  # 启用快捷按钮4
                gr.update(interactive=True),  # 启用快捷按钮5
                gr.update(interactive=True),  # 启用快捷按钮6
                gr.update(interactive=True),  # 启用快捷按钮7
                gr.update(interactive=True),  # 启用快捷按钮8
                gr.update(interactive=True),  # 启用快捷按钮9
                gr.update(interactive=True),  # 启用快捷按钮10
            )
    
    # 🎯 快捷短语函数 - 预设测试查询（使用生成器）
    def quick_phrase_1(history, show_thinking, max_iterations):
        """查询用户等级推理"""
        message = "查询用户ID为1的用户等级，并解释推理过程"
        yield from submit_and_update(message, history, show_thinking, max_iterations)
    
    def quick_phrase_2(history, show_thinking, max_iterations):
        """查询折扣推理"""
        message = "用户ID 1购买金额15000元，查询可用的折扣优惠，并解释推理依据"
        yield from submit_and_update(message, history, show_thinking, max_iterations)
    
    def quick_phrase_3(history, show_thinking, max_iterations):
        """查询物流方案"""
        message = "查询用户ID 1的物流配送方案，包括运费和预计送达时间"
        yield from submit_and_update(message, history, show_thinking, max_iterations)
    
    def quick_phrase_4(history, show_thinking, max_iterations):
        """查询退货政策"""
        message = "用户ID 1购买了AirPods Pro 2（配件类商品），已拆封但包装完好，能否退货？"
        yield from submit_and_update(message, history, show_thinking, max_iterations)
    
    def quick_phrase_5(history, show_thinking, max_iterations):
        """搜索商品"""
        message = "搜索iPhone相关的商品，显示名称、价格和库存"
        yield from submit_and_update(message, history, show_thinking, max_iterations)
    
    def quick_phrase_6(history, show_thinking, max_iterations):
        """创建测试订单"""
        message = "用户ID 1购买2台iPhone 15 Pro（商品ID 2），配送地址：成都武侯区，电话：15308215756"
        yield from submit_and_update(message, history, show_thinking, max_iterations)
    
    def quick_phrase_7(history, show_thinking, max_iterations):
        """商品规范化测试"""
        message = "规范化查询：苹果15手机"
        yield from submit_and_update(message, history, show_thinking, max_iterations)
    
    def quick_phrase_8(history, show_thinking, max_iterations):
        """SHACL校验测试"""
        message = "验证订单数据：用户ID 1，商品ID 2，数量3，地址成都，电话15308215756，是否符合SHACL规则"
        yield from submit_and_update(message, history, show_thinking, max_iterations)
    
    def quick_phrase_9(history, show_thinking, max_iterations):
        """完整推理流程"""
        message = "完整演示本体推理流程：用户ID 1，我有20000元，我爱苹果和耳机, 请尽可能把这钱花完，帮我下个订单,并输出最终订单清单"
        yield from submit_and_update(message, history, show_thinking, max_iterations)
    
    def quick_phrase_10(history, show_thinking, max_iterations):
        """用户消费分析"""
        message = "分析用户ID 1的消费情况，包括累计消费、等级变化和推荐策略"
        yield from submit_and_update(message, history, show_thinking, max_iterations)

    # 绑定事件 - 输出包含所有需要更新的组件
    outputs = [
        chatbot, plan_md, tool_md, memory_md, ecommerce_md, execution_log_md, 
        txt, submit,  # 输入框和发送按钮
        quick_btn1, quick_btn2, quick_btn3, quick_btn4, quick_btn5,  # 快捷按钮
        quick_btn6, quick_btn7, quick_btn8, quick_btn9, quick_btn10
    ]
    
    submit.click(submit_and_update, [txt, chatbot, show_thinking_checkbox, iteration_slider], outputs)
    txt.submit(submit_and_update, [txt, chatbot, show_thinking_checkbox, iteration_slider], outputs)
    clear_btn.click(clear_conversation, None, [chatbot, plan_md, tool_md, memory_md, ecommerce_md, execution_log_md])
    
    # 🎯 绑定快捷按钮事件
    quick_btn1.click(quick_phrase_1, [chatbot, show_thinking_checkbox, iteration_slider], outputs)
    quick_btn2.click(quick_phrase_2, [chatbot, show_thinking_checkbox, iteration_slider], outputs)
    quick_btn3.click(quick_phrase_3, [chatbot, show_thinking_checkbox, iteration_slider], outputs)
    quick_btn4.click(quick_phrase_4, [chatbot, show_thinking_checkbox, iteration_slider], outputs)
    quick_btn5.click(quick_phrase_5, [chatbot, show_thinking_checkbox, iteration_slider], outputs)
    quick_btn6.click(quick_phrase_6, [chatbot, show_thinking_checkbox, iteration_slider], outputs)
    quick_btn7.click(quick_phrase_7, [chatbot, show_thinking_checkbox, iteration_slider], outputs)
    quick_btn8.click(quick_phrase_8, [chatbot, show_thinking_checkbox, iteration_slider], outputs)
    quick_btn9.click(quick_phrase_9, [chatbot, show_thinking_checkbox, iteration_slider], outputs)
    quick_btn10.click(quick_phrase_10, [chatbot, show_thinking_checkbox, iteration_slider], outputs)


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT") or os.getenv("GRADIO_PORT", "7860"))
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    share = _env_flag("GRADIO_SHARE")
    LOGGER.info("Launching Gradio UI on %s:%s (share=%s)", server_name, port, share)
    
    # 启用队列以支持生成器函数（渐进式UI更新）
    demo.queue()
    
    launch_kwargs = {
        "server_name": server_name,
        "server_port": port,
        "share": share,
        "show_api": False,
    }
    
    try:
        demo.launch(**launch_kwargs)
    except (ValueError, OSError) as exc:
        message = str(exc)
        if not share and "shareable link" in message.lower():
            LOGGER.warning("Localhost 访问受限，自动回退为 share=True")
            launch_kwargs["share"] = True
            demo.launch(**launch_kwargs)
        else:
            raise
