from __future__ import annotations
# Copyright (c) 2025 shark8848
# MIT License
#
# Ontology MCP Server - 电商 AI 助手系统
# 本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI
#
# Author: shark8848
# Repository: https://github.com/shark8848/ontology-mcp-server
"""OpenAI/DeepSeek 聊天模型工厂。"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - import guarded for informative error
    raise ImportError(
        "openai package is required for chat model support. "
        "Please install it with `pip install openai`."
    ) from exc

from .logger import get_logger

logger = get_logger(__name__)

DEFAULT_API_URL = "https://api.deepseek.com/v1"
_VERSION_SUFFIX_RE = re.compile(r"/v\d+(?:\.\d+)?$", re.IGNORECASE)


def _strip_endpoint_suffix(url: str) -> str:
    """Remove route fragments like /chat/completions or /responses."""

    for suffix in ("/chat/completions", "/chat/completions/", "/responses", "/responses/"):
        if url.endswith(suffix):
            return url[: -len(suffix)].rstrip("/")
    return url


def _ensure_version_segment(url: str) -> str:
    """Append /v1 if user configured a host root without version."""

    if _VERSION_SUFFIX_RE.search(url):
        return url
    return url.rstrip("/") + "/v1"
DEFAULT_MODEL = "deepseek-chat"

DEFAULT_OLLAMA_API_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_API_KEY = "ollama"


def _load_yaml_config() -> dict:
    try:
        import yaml
    except Exception:
        return {}

    candidates = [
        Path(__file__).resolve().parent / "config.yaml",
        Path(__file__).resolve().parents[1] / "agent" / "config.yaml",
    ]

    for cfg_path in candidates:
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            continue
        if isinstance(data, dict):
            return data

    return {}


def _first_non_empty(*values: Any) -> Optional[Any]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _coerce_float(value: Optional[Any]) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Optional[Any]) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


class DeepseekChatModel:
    """为 OpenAI 兼容接口提供简易的聊天封装。"""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ) -> None:
        client_kwargs: Dict[str, Any] = {}
        if request_timeout is not None:
            client_kwargs["timeout"] = request_timeout

        self.client = OpenAI(base_url=api_url, api_key=api_key, **client_kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """同步生成（非流式）"""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(
                f"LLM API 调用失败: {type(e).__name__}: {str(e)}\n"
                f"API URL: {self.client.base_url}\n"
                f"Model: {self.model}\n"
                f"Messages count: {len(messages)}\n"
                f"Tools count: {len(tools) if tools else 0}",
                exc_info=True
            )
            raise
        
        if not response.choices:
            return {"content": "", "tool_calls": [], "raw_response": response}

        choice = response.choices[0]
        message = choice.message

        content_text = ""
        if isinstance(message.content, list):
            for part in message.content:
                if isinstance(part, dict) and "text" in part:
                    content_text += str(part.get("text", ""))
        elif message.content:
            content_text = str(message.content)

        tool_calls: List[Dict[str, Any]] = []
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:  # type: ignore[attr-defined]
                arguments: Dict[str, Any]
                raw_arguments = getattr(call.function, "arguments", "")
                try:
                    arguments = json.loads(raw_arguments) if raw_arguments else {}
                except json.JSONDecodeError:
                    arguments = {"_raw": raw_arguments}
                tool_calls.append(
                    {
                        "id": call.id,
                        "name": call.function.name,
                        "arguments": arguments,
                    }
                )

        return {
            "content": content_text,
            "tool_calls": tool_calls,
            "raw_response": response,
        }
    
    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """流式生成，逐 token yield 内容
        
        Yields:
            Dict[str, Any]: 包含以下字段
            - delta_content: 本次新增的文本片段
            - accumulated_content: 累积的完整文本
            - tool_calls: 工具调用列表（仅在完成时返回）
            - finish_reason: 完成原因（仅在完成时返回）
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,  # 启用流式
        }
        if tools:
            kwargs["tools"] = tools
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        try:
            stream = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(
                f"LLM 流式 API 调用失败: {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            raise
        
        accumulated_content = ""
        tool_calls_data = []
        
        for chunk in stream:
            if not chunk.choices:
                continue
            
            choice = chunk.choices[0]
            delta = choice.delta
            
            # 处理文本内容增量
            if hasattr(delta, 'content') and delta.content:
                delta_text = delta.content
                accumulated_content += delta_text
                yield {
                    "delta_content": delta_text,
                    "accumulated_content": accumulated_content,
                    "tool_calls": [],
                    "finish_reason": None,
                }
            
            # 处理工具调用（流式模式下通常在最后返回）
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.index >= len(tool_calls_data):
                        tool_calls_data.append({
                            "id": tool_call.id or "",
                            "name": tool_call.function.name if hasattr(tool_call.function, 'name') else "",
                            "arguments": ""
                        })
                    
                    if hasattr(tool_call.function, 'arguments'):
                        tool_calls_data[tool_call.index]["arguments"] += tool_call.function.arguments or ""
            
            # 检查是否完成
            if choice.finish_reason:
                # 解析工具调用参数
                parsed_tool_calls = []
                for tc in tool_calls_data:
                    try:
                        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {"_raw": tc["arguments"]}
                    
                    parsed_tool_calls.append({
                        "id": tc["id"],
                        "name": tc["name"],
                        "arguments": args,
                    })
                
                yield {
                    "delta_content": "",
                    "accumulated_content": accumulated_content,
                    "tool_calls": parsed_tool_calls,
                    "finish_reason": choice.finish_reason,
                }


def build_chat_model(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    request_timeout: Optional[float] = None,
) -> DeepseekChatModel:
    """创建 OpenAI 兼容的聊天模型实例。"""

    cfg = _load_yaml_config()

    provider = (
        _first_non_empty(
            os.getenv("LLM_PROVIDER"),
            cfg.get("LLM_PROVIDER"),
            os.getenv("OPENAI_PROVIDER"),
            cfg.get("OPENAI_PROVIDER"),
            os.getenv("DEEPSEEK_PROVIDER"),
            cfg.get("DEEPSEEK_PROVIDER"),
            "deepseek",
        )
        or "deepseek"
    ).strip().lower()

    # 兼容多种写法，例如 local_ollama / ollama-qwen
    is_ollama = provider in {"ollama", "local", "ollama-qwen", "ollama_qwen", "ollama_local"}

    def resolve_value(
        explicit: Optional[str],
        provider_keys: List[str],
        shared_keys: List[str],
        default: Optional[str],
    ) -> Optional[str]:
        sources: List[Optional[str]] = [explicit]
        for key in provider_keys:
            sources.extend([os.getenv(key), cfg.get(key)])
        for key in shared_keys:
            sources.extend([os.getenv(key), cfg.get(key)])
        sources.append(default)
        return _first_non_empty(*sources)

    provider_url_keys = ["OLLAMA_API_URL"] if is_ollama else ["DEEPSEEK_API_URL", "OPENAI_API_URL"]
    provider_key_keys = ["OLLAMA_API_KEY"] if is_ollama else ["DEEPSEEK_API_KEY", "OPENAI_API_KEY"]
    provider_model_keys = ["OLLAMA_MODEL"] if is_ollama else ["DEEPSEEK_MODEL", "OPENAI_MODEL"]

    shared_keys: List[str] = []
    shared_key_keys: List[str] = []
    shared_model_keys: List[str] = []

    resolved_api_url = resolve_value(
        api_url,
        provider_url_keys,
        shared_keys,
        DEFAULT_OLLAMA_API_URL if is_ollama else DEFAULT_API_URL,
    )
    resolved_api_url = resolved_api_url or (
        DEFAULT_OLLAMA_API_URL if is_ollama else DEFAULT_API_URL
    )
    resolved_api_url = resolved_api_url.strip()
    if not is_ollama and resolved_api_url:
        sanitized_url = _strip_endpoint_suffix(resolved_api_url.rstrip("/"))
        if not sanitized_url:
            sanitized_url = DEFAULT_API_URL
        if sanitized_url != resolved_api_url:
            logger.warning(
                "Detected full endpoint in OPENAI/DEEPSEEK API URL; trimmed to base '%s'",
                sanitized_url,
            )
        resolved_api_url = _ensure_version_segment(sanitized_url)
    else:
        resolved_api_url = resolved_api_url.rstrip("/") or (
            DEFAULT_OLLAMA_API_URL if is_ollama else DEFAULT_API_URL
        )

    resolved_api_key = resolve_value(
        api_key,
        provider_key_keys,
        shared_key_keys,
        DEFAULT_OLLAMA_API_KEY if is_ollama else None,
    )

    if not resolved_api_key:
        raise RuntimeError(
            "未找到 OpenAI/DeepSeek API Key，请设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY"
        )

    resolved_model = resolve_value(
        model,
        provider_model_keys,
        shared_model_keys,
        DEFAULT_OLLAMA_MODEL if is_ollama else DEFAULT_MODEL,
    )
    resolved_model = resolved_model or (
        DEFAULT_OLLAMA_MODEL if is_ollama else DEFAULT_MODEL
    )

    resolved_temperature = (
        temperature
        if temperature is not None
        else _coerce_float(
            _first_non_empty(
                os.getenv("OPENAI_TEMPERATURE"),
                cfg.get("OPENAI_TEMPERATURE"),
                os.getenv("DEEPSEEK_TEMPERATURE"),
                cfg.get("DEEPSEEK_TEMPERATURE"),
            )
        )
    )

    resolved_max_tokens = (
        max_tokens
        if max_tokens is not None
        else _coerce_int(
            _first_non_empty(
                os.getenv("OPENAI_MAX_TOKENS"),
                cfg.get("OPENAI_MAX_TOKENS"),
                os.getenv("DEEPSEEK_MAX_TOKENS"),
                cfg.get("DEEPSEEK_MAX_TOKENS"),
            )
        )
    )

    logger.info(
        "Initializing %s chat model=%s base=%s",
        "Ollama" if is_ollama else "DeepSeek",
        resolved_model,
        resolved_api_url,
    )

    return DeepseekChatModel(
        api_url=resolved_api_url,
        api_key=resolved_api_key,
        model=resolved_model,
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        request_timeout=request_timeout,
    )


def get_default_chat_model() -> DeepseekChatModel:
    """便捷函数：按默认配置创建一个聊天模型。"""

    return build_chat_model()


# 为向后兼容保留旧名称
OpenAICompatibleLLM = build_chat_model
DeepseekLLM = build_chat_model

