from __future__ import annotations
# Copyright (c) 2025 shark8848
# MIT License
#
# Ontology MCP Server - 电商 AI 助手系统
# 本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI
#
# Author: shark8848
# Repository: https://github.com/shark8848/ontology-mcp-server
"""基于 OpenAI 函数调用的轻量智能体封装，支持对话记忆。"""

import json
import threading
import time
import yaml
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from queue import SimpleQueue
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from datetime import datetime

from .llm_deepseek import get_default_chat_model
from .logger import get_logger
from .mcp_adapter import MCPAdapter, ToolDefinition
from .memory_config import (
    get_memory_config, 
    use_chromadb, 
    use_similarity_search as config_use_similarity_search
)
from .prompts import PromptManager, get_default_prompt_manager
from .conversation_state import ConversationStateManager, ConversationStage
from .quality_metrics import QualityMetricsTracker, TaskOutcome
from .intent_tracker import IntentTracker
from .recommendation_engine import RecommendationEngine

# 优先使用 ChromaDB 记忆,回退到基础记忆
try:
    from .chroma_memory import ChromaConversationMemory as ConversationMemory
    CHROMA_AVAILABLE = True
except ImportError:
    from .memory import ConversationMemory
    CHROMA_AVAILABLE = False

logger = get_logger(__name__)


def _load_agent_config() -> Dict[str, Any]:
    """加载 agent 配置文件"""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.warning("无法读取 config.yaml: %s", exc)
    return {}


class LangChainAgent:
    """Minimal agent loop using OpenAI function-calling with MCP tools.
    
    支持对话记忆功能:
    - 自动记录对话历史
    - 生成对话摘要
    - 在每轮推理时注入历史上下文
    - 跟踪购物会话状态（Phase 4 新增）
    - 电商场景专用提示词（Phase 4 新增）
    """

    def __init__(
        self,
        llm=None,
        mcp: Optional[MCPAdapter] = None,
        *,
        max_iterations: int = 10,
        use_memory: bool = None,
        session_id: str = None,
        persist_directory: str = None,
        max_results: int = None,
        use_similarity_search: bool = None,
        max_history: Optional[int] = None,
        max_summary_length: Optional[int] = None,
        enable_conversation_state: bool = True,
        enable_system_prompt: bool = True,
        enable_quality_tracking: bool = True,
        enable_intent_tracking: bool = True,
        enable_recommendation: bool = False,
    ) -> None:
        """初始化 Agent
        
        Args:
            llm: LLM 模型实例
            mcp: MCP 适配器实例
            max_iterations: 最大推理迭代次数
            use_memory: 是否启用对话记忆（None=从配置读取）
            session_id: 会话ID (用于 ChromaDB 隔离不同对话)
            persist_directory: ChromaDB 持久化目录（None=从配置读取）
            max_results: 检索时返回的最大结果数（None=从配置读取）
            use_similarity_search: 是否使用语义相似度检索上下文（None=从配置读取）
            max_history: 基础记忆保留的最大历史轮次（None=从配置读取）
            max_summary_length: 基础记忆注入上下文的摘要数量（None=从配置读取）
            enable_conversation_state: 是否启用对话状态跟踪（Phase 4）
            enable_system_prompt: 是否使用电商专用系统提示（Phase 4）
            enable_quality_tracking: 是否启用对话质量跟踪（Phase 4 优化）
            enable_intent_tracking: 是否启用意图识别跟踪（Phase 4 优化）
            enable_recommendation: 是否启用个性化推荐（Phase 4 优化）
        """
        self.mcp = mcp or MCPAdapter()
        self.tools: List[ToolDefinition] = self.mcp.create_tools()
        self.llm = llm or get_default_chat_model()
        self.max_iterations = max_iterations

        self.tool_map: Dict[str, ToolDefinition] = {tool.name: tool for tool in self.tools}
        self.tool_specs = [tool.to_openai_tool() for tool in self.tools]
        
        # 会话ID（用于多个组件）
        self.session_id = session_id or f"session_{id(self)}"
        
        # 加载配置 (用于意图识别等组件)
        self.config = _load_agent_config()
        
        # Phase 4: Prompt 管理器
        self.enable_system_prompt = enable_system_prompt
        self.prompt_manager: Optional[PromptManager] = None
        if enable_system_prompt:
            self.prompt_manager = get_default_prompt_manager()
            logger.info("已启用电商专用系统提示词")
        
        # Phase 4: 对话状态管理器
        self.enable_conversation_state = enable_conversation_state
        self.state_manager: Optional[ConversationStateManager] = None
        if enable_conversation_state:
            self.state_manager = ConversationStateManager()
            self.state_manager.initialize_session(self.session_id)
            logger.info("已启用对话状态跟踪")
        
        # Phase 4 优化: 质量跟踪器
        self.enable_quality_tracking = enable_quality_tracking
        self.quality_tracker: Optional[QualityMetricsTracker] = None
        if enable_quality_tracking:
            self.quality_tracker = QualityMetricsTracker(session_id=self.session_id)
            logger.info("已启用对话质量跟踪")
        
        # Phase 4 优化: 意图跟踪器
        self.enable_intent_tracking = enable_intent_tracking
        self.intent_tracker: Optional[IntentTracker] = None
        if enable_intent_tracking:
            # 初始化混合意图识别器
            from .intent_tracker import HybridIntentRecognizer
            intent_config = self.config.get("intent_recognition", {})
            recognizer = HybridIntentRecognizer(llm=llm, config=intent_config)
            self.intent_tracker = IntentTracker(session_id=self.session_id, recognizer=recognizer)
            logger.info("已启用意图识别跟踪 (混合策略)")
        
        # Phase 4 优化: 推荐引擎
        self.enable_recommendation = enable_recommendation
        self.recommendation_engine: Optional[RecommendationEngine] = None
        if enable_recommendation:
            self.recommendation_engine = RecommendationEngine()
            logger.info("已启用个性化推荐引擎")
        
        # Phase 5: 查询改写器
        from .query_rewriter import QueryRewriter
        query_rewriter_config = self.config.get("query_rewriter", {})
        self.query_rewriter: Optional[QueryRewriter] = None
        if query_rewriter_config.get("enabled", True):
            self.query_rewriter = QueryRewriter(llm=self.llm, config=query_rewriter_config)
            logger.info("已启用查询改写器")

        ui_config = self.config.get("ui", {}) if isinstance(self.config, dict) else {}
        self.simulated_stream_delay = float(ui_config.get("simulated_stream_delay", 0.08))

        # 强制本体/SHACL 校验相关状态
        self._pending_validation: Optional[Dict[str, Any]] = None
        self._validation_issued_turn: Optional[int] = None
        self._last_validation_iteration: Optional[int] = None

        self._negative_history_keywords = [
            "无法生成图表",
            "图表功能暂时不可用",
            "数据可视化工具暂时无法提供",
            "不能生成柱状图",
            "工具已被禁用",
        ]
        
        # Phase 6: 人工确认机制
        self.CRITICAL_TOOLS = {
            "commerce_create_order": {
                "name": "创建订单",
                "risk_level": "high",
                "requires_confirmation": True
            },
            "commerce_process_payment": {
                "name": "处理支付",
                "risk_level": "critical",
                "requires_confirmation": True
            },
            "commerce_cancel_order": {
                "name": "取消订单",
                "risk_level": "high",
                "requires_confirmation": True
            },
            "commerce_update_order_status": {
                "name": "更新订单状态",
                "risk_level": "medium",
                "requires_confirmation": True
            },
            "ontology_validate_order": {
                "name": "订单SHACL校验",
                "risk_level": "critical",
                "requires_confirmation": False,
            }
        }
        self.pending_confirmations: List[Dict[str, Any]] = []
        self.confirmation_mode: bool = False
        
        # Phase 7: 产品验证层 - 追踪最近搜索到的product_id
        self.recent_search_product_ids: Set[int] = set()  # 最近搜索到的产品ID集合
        self.last_search_results: Dict[str, Any] = {}  # 最近一次搜索结果缓存
        
        # 加载记忆配置
        memory_config = get_memory_config()
        
        # 参数优先级高于配置
        if use_memory is None:
            use_memory = memory_config.enabled
        if use_similarity_search is None:
            use_similarity_search = config_use_similarity_search()
        configured_max_history = (
            max_history if max_history is not None else memory_config.strategy.max_recent_turns
        )
        configured_summary_length = (
            max_summary_length
            if max_summary_length is not None
            else memory_config.summary.max_summary_length
        )
        if max_results is None and max_history is not None:
            max_results = max_history
        
        # 初始化记忆组件
        self.use_memory = use_memory
        self.use_similarity_search = use_similarity_search
        self.memory: Optional[ConversationMemory] = None
        
        if use_memory:
            backend_type = memory_config.backend
            
            # 尝试使用配置的后端
            if backend_type == "chromadb" and CHROMA_AVAILABLE:
                try:
                    self.memory = ConversationMemory(
                        session_id=session_id,
                        persist_directory=persist_directory,
                        max_results=max_results,
                        llm_model=self.llm if memory_config.strategy.enable_llm_summary else None,
                        config=memory_config,
                    )
                    logger.info(
                        "Initialized agent with ChromaDB memory: session=%s, mode=%s",
                        self.memory.session_id, memory_config.strategy.retrieval_mode
                    )
                except Exception as e:
                    logger.error("Failed to initialize ChromaDB memory: %s", e)
                    logger.warning("Falling back to basic memory")
                    from .memory import ConversationMemory as BasicMemory
                    self.memory = BasicMemory(
                        max_history=configured_max_history,
                        max_summary_length=configured_summary_length
                    )
            elif backend_type == "basic" or not CHROMA_AVAILABLE:
                if not CHROMA_AVAILABLE and backend_type == "chromadb":
                    logger.warning("ChromaDB not available, falling back to basic memory")
                from .memory import ConversationMemory as BasicMemory
                self.memory = BasicMemory(
                    max_history=configured_max_history,
                    max_summary_length=configured_summary_length
                )
        
        logger.info("Initialized OpenAI agent with %d tools", len(self.tools))

    # ------------------------------------------------------------------
    # Mandatory ontology / SHACL orchestration helpers
    # ------------------------------------------------------------------
    def _should_require_validation(
        self,
        user_input: str,
        tool_log: List[Dict[str, Any]],
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """根据当前上下文判断是否必须执行 ontology_validate_order。"""

        if self._pending_validation:
            return True, self._pending_validation

        # 如果最近一次订单创建后已经执行过校验，则无需重复
        last_create_index: Optional[int] = None
        for idx, entry in enumerate(tool_log):
            if entry.get("tool") == "commerce_create_order":
                last_create_index = idx

        if last_create_index is not None:
            for entry in tool_log[last_create_index + 1 :]:
                if entry.get("tool") == "ontology_validate_order":
                    return False, None

        stage = None
        if self.state_manager and self.state_manager.state:
            stage = self.state_manager.state.stage

        lowered = user_input.lower()
        requires_validation = False
        payload: Dict[str, Any] = {}

        keywords = ["验证订单", "shacl", "校验", "validate", "数据校验"]
        if any(keyword in lowered for keyword in keywords):
            requires_validation = True

        if not requires_validation and stage == ConversationStage.CHECKOUT:
            requires_validation = True

        if not requires_validation and tool_log:
            for entry in reversed(tool_log):
                tool_name = entry.get("tool", "")
                if tool_name == "commerce_create_order":
                    requires_validation = True
                    payload = entry.get("input", {}) or {}
                    break

        if not requires_validation:
            return False, None

        if "data" not in payload:
            payload["data"] = ""
        if "format" not in payload:
            payload["format"] = "turtle"

        return True, payload

    def _enqueue_validation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """创建一个伪造的工具调用记录，提示 LLM 必须执行 SHACL 校验。"""

        validation_record = {
            "tool": "ontology_validate_order",
            "input": payload,
            "observation": json.dumps(
                {
                    "_tool_info": {
                        "class": "MandatoryValidation",
                        "module": __name__,
                        "method": "auto_enqueue",
                    },
                    "result": {
                        "status": "pending",
                        "message": "系统要求立即调用 ontology_validate_order 完成 SHACL 校验。",
                    },
                },
                ensure_ascii=False,
            ),
            "iteration": -1,
        }
        self._pending_validation = payload
        return validation_record

    def _inject_validation_reminder(
        self,
        messages: List[Dict[str, Any]],
        payload: Dict[str, Any],
        iteration: int,
    ) -> None:
        """在对话中注入系统提醒，要求 LLM 调用校验工具。"""

        if self._validation_issued_turn == iteration:
            return

        reminder = (
            "【系统硬性要求】订单生成后禁止继续回复。"
            "必须立即调用 `ontology_validate_order` 完成 SHACL 校验，"
            "否则系统将终止本轮输出。"
            "如需示例参数，可参考: "
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        messages.append({"role": "system", "content": reminder})
        self._validation_issued_turn = iteration

    def _get_user_context(self) -> Optional[Any]:
        """Safely fetch the latest user context from memory."""
        if not self.memory or not hasattr(self.memory, "user_context_manager"):
            return None
        try:
            return self.memory.user_context_manager.get_context()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("获取用户上下文失败: %s", exc)
            return None

    def _ingest_user_context_from_tool(self, tool_name: str, tool_args: Any, payload: Any) -> None:
        """Push tool interaction data into the user context manager in real time."""
        if not self.memory or not hasattr(self.memory, "user_context_manager"):
            return
        manager = self.memory.user_context_manager
        try:
            manager.ingest_tool_call(tool_name, tool_args, payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("即时更新用户上下文失败: %s", exc)

    def _ingest_user_context_from_text(self, text: Optional[str]) -> None:
        """Capture identifiers directly from user utterances."""
        if not text or not self.memory or not hasattr(self.memory, "user_context_manager"):
            return
        try:
            self.memory.user_context_manager.ingest_free_text(text)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("基于文本更新用户上下文失败: %s", exc)

    def _call_tool(self, name: str, args: Dict[str, Any]) -> str:
        """调用工具并返回结果"""
        tool = self.tool_map.get(name)
        if tool is None:
            logger.warning("LLM attempted to call unknown tool: %s", name)
            return f"未找到工具 {name}"
        tool_class = tool.__class__.__name__
        tool_module = tool.__class__.__module__

        try:
            parsed_args: Any = args
            if isinstance(parsed_args, str):
                try:
                    parsed_args = json.loads(parsed_args)
                except json.JSONDecodeError:
                    parsed_args = {"_raw": parsed_args}

            if not isinstance(parsed_args, dict):
                parsed_args = {"_raw": parsed_args}

            logger.debug(f"调用工具 {name} (类: {tool_module}.{tool_class})")
            parsed = tool.parse_arguments(parsed_args)
            result = tool.invoke(parsed)

            safe_result = self._make_json_safe(result)
            return json.dumps({
                "_tool_info": {
                    "class": tool_class,
                    "module": tool_module,
                    "method": "invoke",
                },
                "result": safe_result,
            }, ensure_ascii=False)

        except Exception as exc:  # pragma: no cover - defensive runtime logging
            logger.exception("Tool %s invocation failed", name)
            return json.dumps({
                "_tool_info": {
                    "class": tool_class,
                    "module": tool_module,
                    "method": "invoke",
                },
                "error": f"调用失败: {type(exc).__name__}: {str(exc)}",
            }, ensure_ascii=False)

    def _prepare_order_arguments(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """Align commerce_create_order payload with tracked user context."""
        sanitized = deepcopy(args) if args else {}
        ctx = self._get_user_context()

        if not ctx or ctx.user_id is None:
            return sanitized, "missing_user_id"

        try:
            sanitized["user_id"] = int(ctx.user_id)
        except (TypeError, ValueError):
            return sanitized, "invalid_user_id"

        items = sanitized.get("items")
        if not items or not isinstance(items, list):
            return sanitized, "missing_items"

        target_product_id: Optional[int] = None
        if getattr(ctx, "recent_product_id", None) is not None:
            target_product_id = int(ctx.recent_product_id)
        elif len(self.recent_search_product_ids) == 1:
            target_product_id = next(iter(self.recent_search_product_ids))

        if target_product_id is None:
            return sanitized, "missing_product"

        for item in items:
            if not isinstance(item, dict):
                continue
            item["product_id"] = target_product_id

        return sanitized, None

    @staticmethod
    def _parse_tool_observation(observation: Any) -> Optional[Dict[str, Any]]:
        """Extract structured payload from stored tool observation."""
        data: Any = observation
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (TypeError, json.JSONDecodeError):
                return None

        if not isinstance(data, dict):
            return None

        payload: Any = data.get("result", data)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                return data

        return payload if isinstance(payload, (dict, list)) else data

    def _build_checkout_summary(self, tool_log: List[Dict[str, Any]]) -> Optional[str]:
        """Generate a concise order/payment summary for the user."""
        order_payload = None
        payment_payload = None

        for entry in reversed(tool_log):
            tool_name = entry.get("tool")
            if tool_name == "commerce_process_payment" and payment_payload is None:
                payment_payload = self._parse_tool_observation(entry.get("observation"))
            if tool_name == "commerce_create_order" and order_payload is None:
                order_payload = self._parse_tool_observation(entry.get("observation"))
            if order_payload and payment_payload:
                break

        if not order_payload and not payment_payload:
            return None

        order_info = None
        if isinstance(order_payload, dict):
            order_info = order_payload.get("order") or order_payload

        payment_info = None
        if isinstance(payment_payload, dict):
            payment_info = payment_payload.get("payment") or payment_payload

        lines: List[str] = ["✅ 已完成订单流程总结:"]

        if isinstance(order_info, dict):
            order_no = order_info.get("order_no") or order_info.get("order_id")
            total_amount = order_info.get("final_amount") or order_info.get("total_amount")
            status = order_info.get("order_status") or order_info.get("status")
            if order_no:
                lines.append(f"• 订单号: {order_no}")
            if total_amount is not None:
                lines.append(f"• 实付金额: ¥{float(total_amount):.2f}")
            if status:
                lines.append(f"• 当前状态: {status}")

        if isinstance(payment_info, dict):
            pay_status = payment_info.get("payment_status") or payment_info.get("status")
            pay_method = payment_info.get("payment_method") or payment_info.get("method")
            transaction_id = payment_info.get("transaction_id")
            amount = payment_info.get("payment_amount") or payment_info.get("amount")
            if pay_method:
                lines.append(f"• 支付方式: {pay_method}")
            if amount is not None:
                lines.append(f"• 支付金额: ¥{float(amount):.2f}")
            if pay_status:
                lines.append(f"• 支付状态: {pay_status}")
            if transaction_id:
                lines.append(f"• 交易号: {transaction_id}")
        else:
            lines.append("• 订单已生成，等待支付确认。")

        lines.append("如需调整订单或取消支付，请直接告知我。")
        return "\n".join(lines)

    @staticmethod
    def _stringify_observation(value: Any) -> str:
        """Convert arbitrary observation payload into a printable string."""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            try:
                return str(value)
            except Exception:  # pragma: no cover - extreme fallback
                return ""

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        """Convert tool outputs (including dataclasses) into JSON-safe data."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if is_dataclass(value):
            try:
                return LangChainAgent._make_json_safe(asdict(value))
            except Exception:  # pragma: no cover - defensive guard
                return str(value)

        if isinstance(value, dict):
            return {
                str(key): LangChainAgent._make_json_safe(val)
                for key, val in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [LangChainAgent._make_json_safe(item) for item in value]

        if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
            try:
                return LangChainAgent._make_json_safe(value.to_dict())
            except Exception:  # pragma: no cover - defensive guard
                return str(value)

        return str(value)

    def _summarize_confirmation_result(self, tool_name: str, raw_result: Any) -> Optional[str]:
        """Generate a friendly summary for confirmed high-risk tool outputs."""
        payload = raw_result
        if isinstance(raw_result, str):
            try:
                payload = json.loads(raw_result)
            except (json.JSONDecodeError, TypeError, ValueError):
                payload = raw_result

        if isinstance(payload, dict) and "result" in payload and len(payload) == 2 and "_tool_info" in payload:
            payload = payload.get("result")

        if tool_name == "commerce_create_order" and isinstance(payload, dict):
            order = payload.get("order") or payload
            if isinstance(order, dict):
                lines = ["订单已创建成功，关键信息如下："]
                order_no = order.get("order_no") or order.get("order_id")
                total_amount = order.get("total_amount") or order.get("final_amount")
                status = order.get("order_status") or order.get("status")
                address = order.get("shipping_address")
                phone = order.get("contact_phone")
                if order_no:
                    lines.append(f"• 订单号: {order_no}")
                if total_amount is not None:
                    lines.append(f"• 金额: ¥{float(total_amount):.2f}")
                if status:
                    lines.append(f"• 状态: {status}")
                if address:
                    lines.append(f"• 地址: {address}")
                if phone:
                    lines.append(f"• 电话: {phone}")
                items = order.get("items")
                if isinstance(items, list) and items:
                    first_item = items[0]
                    if isinstance(first_item, dict):
                        name = first_item.get("product_name") or first_item.get("product_id")
                        qty = first_item.get("quantity")
                        lines.append(f"• 商品: {name} × {qty}")
                return "\n".join(lines)

        if tool_name == "commerce_process_payment" and isinstance(payload, dict):
            payment = payload.get("payment") or payload
            if isinstance(payment, dict):
                lines = ["支付已完成："]
                transaction_id = payment.get("transaction_id")
                amount = payment.get("payment_amount") or payment.get("amount")
                method = payment.get("payment_method") or payment.get("method")
                status = payment.get("payment_status") or payment.get("status")
                if amount is not None:
                    lines.append(f"• 金额: ¥{float(amount):.2f}")
                if method:
                    lines.append(f"• 支付方式: {method}")
                if transaction_id:
                    lines.append(f"• 交易号: {transaction_id}")
                if status:
                    lines.append(f"• 状态: {status}")
                return "\n".join(lines)

        if isinstance(payload, (dict, list)):
            try:
                return json.dumps(payload, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                pass

        return str(raw_result) if raw_result is not None else None

        tool_class = tool.__class__.__name__
        tool_module = tool.__class__.__module__

        try:
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}

            logger.debug(f"调用工具 {name} (类: {tool_module}.{tool_class})")
            parsed = tool.parse_arguments(args)
            result = tool.invoke(parsed)

            return json.dumps({
                "_tool_info": {
                    "class": tool_class,
                    "module": tool_module,
                    "method": "invoke",
                },
                "result": result if isinstance(result, (dict, list, str, int, float, bool)) else str(result),
            }, ensure_ascii=False)

        except Exception as exc:  # pragma: no cover - defensive runtime logging
            logger.exception("Tool %s invocation failed", name)
            return json.dumps({
                "_tool_info": {
                    "class": tool_class,
                    "module": tool_module,
                    "method": "invoke",
                },
                "error": f"调用失败: {type(exc).__name__}: {str(exc)}",
            }, ensure_ascii=False)

    def _query_product_details(self, product_id: int) -> Optional[Dict[str, Any]]:
        """查询商品详细信息用于确认提示
        
        Args:
            product_id: 商品ID
            
        Returns:
            Dict: 商品信息 {product_id, product_name, brand, price, category}
            None: 查询失败
        """
        try:
            # 使用commerce_get_product_detail工具查询
            tool = self.tool_map.get("commerce_get_product_detail")
            if not tool:
                logger.warning("commerce_get_product_detail工具不可用")
                return None
            
            result = tool.invoke({"product_id": product_id})
            if isinstance(result, str):
                result = json.loads(result)
            
            # 提取关键字段
            if isinstance(result, dict) and "product_id" in result:
                return {
                    "product_id": result.get("product_id"),
                    "product_name": result.get("product_name", "未知"),
                    "brand": result.get("brand", "未知"),
                    "price": result.get("price", 0.0),
                    "category": result.get("category", "未知")
                }
            
            return None
        except Exception as e:
            logger.error("查询商品详情失败 (product_id=%s): %s", product_id, e)
            return None

    def _generate_confirmation_request(self, tool_name: str, tool_info: dict, args: dict) -> str:
        """生成人工确认提示"""
        templates = {
            "commerce_create_order": """⚠️ **需要您的确认**

我即将为您创建订单:
- 用户ID: {user_id}
- 商品信息:
{item_details}
- 收货地址: {shipping_address}
- 联系电话: {contact_phone}
{warning_message}
**请回复**:
- "确认" 或 "是" → 创建订单
- "取消" 或 "否" → 取消操作""",
            
            "commerce_process_payment": """💳 **支付确认**

订单号: {order_id}
支付金额: ¥{amount}
支付方式: {payment_method}

⚠️ 此操作将扣款，请确认:
- "确认支付" → 执行支付
- "取消" → 不支付""",
            
            "commerce_cancel_order": """🚫 **取消订单确认**

订单号: {order_id}

确认取消此订单吗?
- "确认取消" → 取消订单
- "保留" → 保留订单""",
            
            "commerce_update_order_status": """📝 **更新订单状态确认**

订单号: {order_id}
新状态: {new_status}

确认更新订单状态吗?
- "确认" → 更新状态
- "取消" → 取消操作"""
        }
        
        template = templates.get(tool_name, "确认执行操作: {tool_name}?")
        
        # 根据工具类型填充参数
        try:
            if tool_name == "commerce_create_order":
                items = args.get("items", [])
                
                # 查询商品详情并生成详细信息
                item_details_list = []
                total_amount = 0.0
                
                for item in items:
                    product_id = item.get('product_id')
                    quantity = item.get('quantity', 0)
                    
                    # 查询商品详情
                    product_info = self._query_product_details(product_id) if product_id else None
                    
                    if product_info:
                        product_name = product_info['product_name']
                        brand = product_info['brand']
                        price = product_info['price']
                        subtotal = price * quantity
                        total_amount += subtotal
                        
                        item_details_list.append(
                            f"  * {product_name} ({brand}) - 单价¥{price:.2f} × {quantity}件 = ¥{subtotal:.2f}\n"
                            f"    [商品ID={product_id}]"
                        )
                    else:
                        # 查询失败,使用基本信息
                        item_details_list.append(
                            f"  * 商品ID={product_id or 'N/A'}, 数量={quantity}件 (⚠️ 无法获取详细信息)"
                        )
                
                item_details = "\n".join(item_details_list)
                
                # 生成总价提示
                if total_amount > 0:
                    item_details += f"\n\n💰 **预估总价**: ¥{total_amount:.2f} (不含优惠)"
                
                # 生成警告信息（检测对话历史中的品牌需求）
                warning_message = ""
                if self.memory and hasattr(self.memory, 'get_recent_turns'):
                    try:
                        recent_turns = self.memory.get_recent_turns(max_turns=5)
                        user_messages = []
                        for turn in recent_turns:
                            if hasattr(turn, 'user_input'):
                                user_messages.append(turn.user_input.lower())
                        
                        # 检测用户是否提到特定品牌
                        requested_brands = []
                        brand_keywords = {
                            "小米": "Xiaomi",
                            "xiaomi": "Xiaomi",
                            "苹果": "Apple",
                            "apple": "Apple",
                            "华为": "Huawei",
                            "huawei": "Huawei"
                        }
                        
                        for msg in user_messages:
                            for keyword, brand in brand_keywords.items():
                                if keyword in msg and brand not in requested_brands:
                                    requested_brands.append(brand)
                        
                        # 检查订单中的品牌是否匹配
                        if requested_brands and items:
                            order_brands = []
                            for item in items:
                                product_id = item.get('product_id')
                                product_info = self._query_product_details(product_id) if product_id else None
                                if product_info:
                                    order_brands.append(product_info['brand'])
                            
                            # 如果用户要求的品牌不在订单中
                            mismatched = [b for b in requested_brands if b not in order_brands]
                            if mismatched and order_brands:
                                warning_message = f"\n\n🚨 **注意**: 您之前提到了 [{', '.join(mismatched)}] 品牌，但订单中的商品是 [{', '.join(set(order_brands))}] 品牌！\n如果这不是您想要的，请回复'取消'。\n"
                    except Exception as e:
                        logger.debug("生成警告信息时出错: %s", e)
                
                return template.format(
                    user_id=args.get("user_id", "未提供"),
                    item_details=item_details or "  * 无商品信息",
                    shipping_address=args.get("shipping_address", "未提供"),
                    contact_phone=args.get("contact_phone", "未提供"),
                    warning_message=warning_message
                )
            
            elif tool_name == "commerce_process_payment":
                return template.format(
                    order_id=args.get("order_id", "未知"),
                    amount=args.get("amount", "0.00"),
                    payment_method=args.get("payment_method", "未指定")
                )
            
            elif tool_name == "commerce_cancel_order":
                return template.format(
                    order_id=args.get("order_id", "未知")
                )
            
            elif tool_name == "commerce_update_order_status":
                return template.format(
                    order_id=args.get("order_id", "未知"),
                    new_status=args.get("status", "未知")
                )
            
            return template.format(tool_name=tool_name)
        except Exception as e:
            logger.error("生成确认请求失败: %s", e)
            return f"⚠️ 需要您确认操作: {tool_info['name']}\n\n请回复 '确认' 或 '取消'"
    
    def _llm_classify_confirmation(self, user_input: str, operation_name: str) -> str:
        """使用LLM分类用户的确认意图
        
        Args:
            user_input: 用户输入
            operation_name: 操作名称（如"创建订单"）
            
        Returns:
            str: "confirmed" | "cancelled" | "questioning" | "unclear"
        """
        try:
            prompt = f"""请分析用户对于"{operation_name}"操作的响应意图。

用户输入: "{user_input}"

请判断用户的真实意图是以下哪一种，只返回对应的英文单词：
1. confirmed - 用户明确同意/确认执行该操作
   示例: "确认"、"好的"、"可以"、"同意"、"是的"、"没问题"、"ok"
   
2. cancelled - 用户明确拒绝/取消该操作
   示例: "取消"、"不要"、"不买了"、"算了"、"不行"、"no"
   
3. questioning - 用户对操作有疑问/质疑
   示例: "为什么是这个"、"怎么搞错了"、"不对吧"、"是不是弄错了"
   
4. unclear - 无法明确判断用户意图
   示例: "嗯"、"这个..."、其他模糊表述

注意：
- "是不是搞错了" 应该判断为 questioning，不是 confirmed
- "为何变成XX了" 应该判断为 questioning，不是 confirmed  
- 只有明确的肯定词才能判断为 confirmed
- 任何疑问句都应该判断为 questioning

只返回一个单词: confirmed/cancelled/questioning/unclear"""

            # 创建临时LLM实例用于分类(使用低max_tokens)
            from .llm_deepseek import build_chat_model
            classifier_llm = build_chat_model(
                temperature=0.1,
                max_tokens=10
            )
            
            response = classifier_llm.generate(
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response.get("content", "").strip().lower()
            
            # 验证返回值
            valid_results = ["confirmed", "cancelled", "questioning", "unclear"]
            if result in valid_results:
                logger.info("LLM分类确认意图: '%s' → %s", user_input[:50], result)
                return result
            else:
                logger.warning("LLM返回无效结果: %s, 回退到unclear", result)
                return "unclear"
                
        except Exception as e:
            logger.error("LLM分类确认意图失败: %s, 回退到关键词匹配", e)
            # 回退到简单关键词匹配
            user_lower = user_input.lower()
            if any(kw in user_lower for kw in ["确认", "好", "可以", "同意", "是的", "ok", "yes"]):
                return "confirmed"
            elif any(kw in user_lower for kw in ["取消", "不要", "不买", "算了", "no"]):
                return "cancelled"
            elif any(kw in user_lower for kw in ["为什么", "为何", "怎么", "不对", "错了", "搞错"]):
                return "questioning"
            else:
                return "unclear"
    
    def _handle_confirmation_response(self, user_input: str) -> Optional[Dict[str, Any]]:
        """处理用户确认响应（使用LLM智能判断）
        
        Returns:
            Dict: {"action": "confirmed"|"cancelled"|"pending", "result": str, "executed": bool}
            None: 不在确认模式或无法识别
        """
        if not self.confirmation_mode or not self.pending_confirmations:
            return None
        
        # 获取待确认操作
        pending = self.pending_confirmations[-1]
        tool_name = pending["tool_name"]
        tool_info = self.CRITICAL_TOOLS.get(tool_name, {})
        operation_name = tool_info.get("name", "操作")
        
        # 使用LLM分类用户意图
        intent = self._llm_classify_confirmation(user_input, operation_name)
        
        if intent == "confirmed":
            # 用户确认 → 执行操作
            tool_name = pending["tool_name"]
            args = pending["args"]
            
            logger.info("✅ 用户确认操作: %s", tool_name)
            
            # 实际执行工具调用(绕过确认检查)
            tool = self.tool_map.get(tool_name)
            if tool:
                try:
                    parsed = tool.parse_arguments(args)
                    result = tool.invoke(parsed)
                    
                    # 清理确认状态
                    self.pending_confirmations.pop()
                    if not self.pending_confirmations:
                        self.confirmation_mode = False
                    
                    logger.info("操作已执行: %s", tool_name)
                    return {
                        "action": "confirmed",
                        "result": result,
                        "executed": True,
                        "tool_name": tool_name,
                        "args": deepcopy(args)
                    }
                except Exception as e:
                    logger.error("执行确认操作失败: %s", e)
                    self.pending_confirmations.pop()
                    if not self.pending_confirmations:
                        self.confirmation_mode = False
                    return {
                        "action": "error",
                        "result": f"执行失败: {str(e)}",
                        "executed": False
                    }
            else:
                logger.error("工具未找到: %s", tool_name)
                self.pending_confirmations.pop()
                if not self.pending_confirmations:
                    self.confirmation_mode = False
                return {
                    "action": "error",
                    "result": f"工具 {tool_name} 未找到",
                    "executed": False
                }
        
        elif intent == "cancelled":
            # 用户取消
            logger.info("❌ 用户取消操作: %s", tool_name)
            
            self.pending_confirmations.pop()
            if not self.pending_confirmations:
                self.confirmation_mode = False
            
            return {
                "action": "cancelled",
                "result": f"操作已取消: {operation_name}。还有什么我可以帮您的吗?",
                "executed": False,
                "tool_name": tool_name
            }
        
        elif intent == "questioning":
            # 用户有疑问 - 当作取消处理
            logger.info("❓ 用户质疑操作(自动取消): %s, 输入: %s", tool_name, user_input[:50])
            
            self.pending_confirmations.pop()
            if not self.pending_confirmations:
                self.confirmation_mode = False
            
            return {
                "action": "cancelled",
                "result": f"检测到您对订单有疑问，操作已自动取消。如需重新下单，请明确告知您需要的商品信息。",
                "executed": False,
                "tool_name": tool_name
            }
        
        else:  # unclear
            # 无法识别意图
            logger.warning("⏳ 无法明确判断确认意图: %s", user_input[:50])
            return {
                "action": "pending",
                "result": "抱歉，我没有理解您的意思。请明确回复：\n• '确认' - 同意执行该操作\n• '取消' - 取消该操作",
                "executed": False
            }

    def _summarize_tool_observation(self, tool_entry: Dict[str, Any]) -> Optional[str]:
        """基于最近一次工具输出构造兜底总结。"""
        observation = tool_entry.get("observation")
        if not observation:
            return None

        parsed: Any = None
        if isinstance(observation, str):
            try:
                parsed = json.loads(observation)
            except json.JSONDecodeError:
                parsed = None
        elif isinstance(observation, (dict, list)):
            parsed = observation

        if isinstance(parsed, dict):
            payload = parsed.get("result", parsed)
        else:
            payload = parsed

        if isinstance(payload, dict):
            items = payload.get("items") or payload.get("products") or payload.get("results")
            if isinstance(items, list) and items:
                lines = ["以下是我刚获取的商品信息摘要："]
                for item in items[:5]:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or item.get("title") or "商品"
                    price = item.get("price") or item.get("unit_price")
                    stock = item.get("stock")
                    if stock is None:
                        stock = item.get("inventory")
                    details = [f"名称：{name}"]
                    if price is not None:
                        details.append(f"价格：¥{price}")
                    if stock is not None:
                        details.append(f"库存：{stock}")
                    lines.append("- " + "，".join(details))
                total = payload.get("count") or payload.get("total")
                if total:
                    lines.append(f"共返回 {total} 条结果，可继续筛选以获取更精确列表。")
                return "\n".join(lines)

        if isinstance(payload, list) and payload:
            preview = payload[:3]
            return f"工具返回了 {len(payload)} 条记录示例：{preview}"

        if isinstance(observation, str):
            trimmed = observation.strip()
            return trimmed[:600] + ("..." if len(trimmed) > 600 else "")

        return None

    def _emit_stream_event(
        self,
        handler: Optional[Callable[[Dict[str, Any]], None]],
        step_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Safely emit a streaming event to the registered handler."""
        if not handler:
            return
        try:
            handler({
                "step_type": step_type,
                "content": content,
                "metadata": metadata or {},
            })
        except Exception as exc:  # pragma: no cover - tracing handler failures
            logger.debug("Stream handler raised %s", exc)

    def _chunk_text_for_stream(self, text: str, chunk_size: int = 80) -> List[str]:
        """Split plain text into small chunks for simulated streaming."""
        if not text:
            return []
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _stream_final_answer(self, result: Dict[str, Any]):
        """Yield events for the final assistant answer (real or simulated streaming)."""
        final_answer = result.get("final_answer", "") or ""
        raw_messages = result.get("raw_messages", []) or []
        streaming_supported = hasattr(self.llm, "generate_stream")
        used_llm_stream = False

        def _emit_final(text: str, streaming: bool, extra: Optional[Dict[str, Any]] = None):
            meta = {"full_result": result, "streaming": streaming}
            if extra:
                meta.update(extra)
            yield {
                "step_type": "final_answer",
                "content": text,
                "metadata": meta,
            }

        if (
            streaming_supported
            and raw_messages
            and raw_messages[-1].get("role") == "tool"
        ):
            yield {
                "step_type": "llm_streaming_start",
                "content": "正在生成答案...",
                "metadata": {},
            }
            try:
                accumulated_answer = ""
                for chunk in self.llm.generate_stream(raw_messages, tools=None):
                    delta = chunk.get("delta_content", "")
                    if delta:
                        accumulated_answer = chunk.get("accumulated_content", accumulated_answer + delta)
                        yield {
                            "step_type": "llm_streaming",
                            "content": delta,
                            "metadata": {"accumulated": accumulated_answer},
                        }
                    if chunk.get("finish_reason"):
                        used_llm_stream = True
                        final_answer = accumulated_answer or final_answer
                        result["final_answer"] = final_answer
                        yield from _emit_final(final_answer, True)
                        return
            except Exception as exc:  # pragma: no cover - streaming fallback
                logger.error("LLM 流式生成失败: %s", exc, exc_info=True)
                yield {
                    "step_type": "error",
                    "content": f"流式生成失败: {exc}",
                    "metadata": {"phase": "llm_streaming"},
                }

        if not used_llm_stream:
            if final_answer:
                yield {
                    "step_type": "llm_streaming_start",
                    "content": "正在组织答案...",
                    "metadata": {},
                }
                accumulated = ""
                for chunk in self._chunk_text_for_stream(final_answer):
                    accumulated += chunk
                    yield {
                        "step_type": "llm_streaming",
                        "content": chunk,
                        "metadata": {"accumulated": accumulated},
                    }
                    delay = getattr(self, "simulated_stream_delay", 0.0)
                    if delay > 0:
                        time.sleep(delay)
            yield from _emit_final(final_answer, False)

    def run(
        self,
        user_input: str,
        *,
        stream_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """执行 Agent 推理循环
        
        Args:
            user_input: 用户输入
            stream_handler: 可选的流式事件回调
            
        Returns:
            Dict: 包含 final_answer, plan, history, tool_log, execution_log 等信息
        """

        logger.info("LangChain agent received input: %s", user_input)
        self._emit_stream_event(
            stream_handler,
            "thinking_start",
            "正在分析您的需求...",
            {"user_input": user_input[:120]},
        )
        self._ingest_user_context_from_text(user_input)
        
        # Phase 6: 优先处理人工确认响应
        if self.confirmation_mode:
            confirmation_result = self._handle_confirmation_response(user_input)
            if confirmation_result:
                action = confirmation_result["action"]
                result_msg = confirmation_result["result"]
                
                if action == "confirmed":
                    # 用户确认,操作已执行
                    tool_name = confirmation_result.get("tool_name", "未知操作")
                    logger.info("✅ 用户确认并执行: %s", tool_name)
                    summary_text = self._summarize_confirmation_result(tool_name, result_msg)
                    follow_up_hint = ""
                    if tool_name == "commerce_create_order":
                        follow_up_hint = (
                            "\n\n🔜 订单已生成，如需我继续安排支付，请直接回复\"确认支付\"" 
                            "或告知想使用的支付方式，我会立即为您处理扣款。"
                        )
                    elif tool_name == "commerce_process_payment":
                        follow_up_hint = (
                            "\n\n如需我继续跟进发货、开票或订单状态，请告诉我下一步需求。"
                        )
                    if summary_text:
                        display_result = summary_text + follow_up_hint
                    elif isinstance(result_msg, str):
                        display_result = result_msg + follow_up_hint
                    else:
                        display_result = json.dumps(result_msg, ensure_ascii=False, indent=2) + follow_up_hint

                    final_message = f"✓ 操作已完成: {tool_name}\n\n{display_result}\n\n还有什么我可以帮您的吗?"

                    tool_args = confirmation_result.get("args", {}) or {}
                    tool_obj = self.tool_map.get(tool_name)
                    tool_class = tool_obj.__class__.__name__ if tool_obj else "UnknownTool"
                    tool_module = tool_obj.__class__.__module__ if tool_obj else __name__
                    safe_result = self._make_json_safe(result_msg)
                    observation_payload = json.dumps(
                        {
                            "_tool_info": {
                                "class": tool_class,
                                "module": tool_module,
                                "method": "invoke",
                            },
                            "result": safe_result,
                            "executed_via_confirmation": True,
                        },
                        ensure_ascii=False,
                    )
                    confirmation_tool_entry = {
                        "tool": tool_name,
                        "input": deepcopy(tool_args),
                        "observation": observation_payload,
                        "iteration": 0,
                        "confirmed": True,
                    }

                    try:
                        self._ingest_user_context_from_tool(tool_name, tool_args, safe_result)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.debug("确认结果写入用户上下文失败: %s", exc)

                    if self.state_manager:
                        try:
                            self.state_manager.update_from_tool_results([confirmation_tool_entry])
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.debug("基于确认结果更新对话状态失败: %s", exc)

                    if self.use_memory and self.memory:
                        try:
                            self.memory.add_turn(
                                user_input=user_input,
                                agent_response=final_message,
                                tool_calls=[confirmation_tool_entry],
                            )
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.debug("确认结果写入记忆失败: %s", exc)

                    # 将执行结果包装为正常返回
                    return {
                        "final_answer": final_message,
                        "plan": [],
                        "history": [],
                        "tool_log": [confirmation_tool_entry],
                        "execution_log": [{
                            "step_type": "confirmation",
                            "content": f"用户确认执行 {tool_name}",
                            "timestamp": datetime.now().isoformat()
                        }]
                    }
                
                elif action == "cancelled":
                    # 用户取消
                    logger.info("❌ 用户取消操作")
                    return {
                        "final_answer": f"{result_msg}\n\n还有什么我可以帮您的吗?",
                        "plan": [],
                        "history": [],
                        "tool_log": [],
                        "execution_log": [{
                            "step_type": "confirmation_cancelled",
                            "content": result_msg,
                            "timestamp": datetime.now().isoformat()
                        }]
                    }
                
                elif action == "pending":
                    # 无法识别意图,继续等待
                    logger.warning("⏳ 等待明确的确认响应")
                    return {
                        "final_answer": result_msg,
                        "plan": [],
                        "history": [],
                        "tool_log": [],
                        "execution_log": [{
                            "step_type": "confirmation_pending",
                            "content": "等待用户确认",
                            "timestamp": datetime.now().isoformat()
                        }]
                    }
                
                elif action == "error":
                    # 执行错误
                    logger.error("❌ 确认操作执行失败")
                    return {
                        "final_answer": f"抱歉,{result_msg}\n\n请稍后重试。",
                        "plan": [],
                        "history": [],
                        "tool_log": [],
                        "execution_log": [{
                            "step_type": "confirmation_error",
                            "content": result_msg,
                            "timestamp": datetime.now().isoformat()
                        }]
                    }
        
        # Phase 4 优化: 开始质量跟踪
        if self.quality_tracker:
            self.quality_tracker.start_turn()
        
        # Phase 4 优化: 跟踪用户意图
        turn_id = len(self.quality_tracker.session_metrics.turns) + 1 if self.quality_tracker else 0
        current_intent = None
        if self.intent_tracker:
            current_intent = self.intent_tracker.track_intent(user_input, turn_id)
            logger.info(f"识别意图: {current_intent.category.value} (置信度: {current_intent.confidence:.2f})")
            if current_intent:
                self._emit_stream_event(
                    stream_handler,
                    "intent_recognized",
                    f"识别意图: {current_intent.category.value}",
                    {
                        "intent": current_intent.category.value,
                        "confidence": getattr(current_intent, "confidence", 0.0),
                    },
                )
        
        # Phase 5: 查询改写 (针对推荐意图)
        enhanced_input = user_input
        rewritten_query = None
        from .intent_tracker import IntentCategory
        if (self.query_rewriter and current_intent and 
            current_intent.category in [IntentCategory.RECOMMENDATION, IntentCategory.SEARCH]):
            try:
                rewritten_query = self.query_rewriter.rewrite(user_input, current_intent)
                enhanced_input = self.query_rewriter.format_enhanced_prompt(user_input, rewritten_query)
                logger.info(f"查询已改写: 类别={rewritten_query.category}, 关键词={rewritten_query.keywords[:3]}")
                self._emit_stream_event(
                    stream_handler,
                    "query_rewritten",
                    f"查询改写完成: 类别={rewritten_query.category}",
                    {
                        "keywords": rewritten_query.keywords[:3],
                        "strategy": rewritten_query.search_strategy,
                    },
                )
            except Exception as e:
                logger.error(f"查询改写失败: {e}, 使用原始查询")
                enhanced_input = user_input
        
        # 初始化执行日志
        execution_log = []
        
        def add_log(step_type: str, content: Any, metadata: dict = None) -> dict:
            """添加执行日志条目"""
            log_entry = {
                "step_type": step_type,
                "content": content,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "metadata": metadata if metadata else {}
            }
            execution_log.append(log_entry)
            return log_entry

        # 记录用户输入
        add_log("user_input", user_input, {})
        
        # 记录查询改写结果
        if rewritten_query:
            add_log("query_rewrite", {
                "original": user_input,
                "understood_intent": rewritten_query.understood_intent,
                "category": rewritten_query.category,
                "keywords": rewritten_query.keywords,
                "expanded_keywords": rewritten_query.expanded_keywords[:10],
                "strategy": rewritten_query.search_strategy,
                "confidence": rewritten_query.confidence
            }, {"reasoning": rewritten_query.reasoning})

        # 注入对话历史上下文
        context_prefix = ""
        if self.use_memory and self.memory and hasattr(self.memory, 'get_recent_turns'):
            # ChromaDB 记忆
            if self.use_similarity_search and user_input:
                # 使用语义相似度检索
                context_prefix = self.memory.get_context_for_prompt(
                    use_similarity=True,
                    query=user_input,
                    max_turns=5,
                )
                add_log("memory_retrieval", "使用语义相似度检索", {
                    "mode": "similarity",
                    "result_length": len(context_prefix),
                    "query": user_input[:100]
                })
            else:
                # 使用最近对话
                context_prefix = self.memory.get_context_for_prompt(
                    use_similarity=False,
                    max_turns=5,
                )
                add_log("memory_retrieval", "使用最近对话检索", {
                    "mode": "recent",
                    "result_length": len(context_prefix)
                })
            
            if context_prefix:
                logger.info("注入对话历史上下文: %d 字符", len(context_prefix))
                logger.info("对话历史内容: %s", context_prefix[:500])  # 记录前500字符到日志
                add_log("memory_context", context_prefix, {
                    "length": len(context_prefix)
                })
        elif self.use_memory and self.memory and hasattr(self.memory, 'get_context_for_prompt'):
            # 基础记忆
            context_prefix = self.memory.get_context_for_prompt()
            if context_prefix:
                logger.info("注入对话历史上下文: %d 字符", len(context_prefix))
                logger.info("对话历史内容: %s", context_prefix[:500])  # 记录前500字符到日志
                add_log("memory_context", context_prefix, {
                    "length": len(context_prefix)
                })
        if context_prefix:
            context_prefix = self._filter_negative_history(context_prefix)
        
        # 构造系统提示(如果有历史上下文)
        messages: List[Dict[str, Any]] = []
        
        # Phase 4: 添加系统提示
        if self.enable_system_prompt and self.prompt_manager:
            system_prompt = self.prompt_manager.get_system_prompt()
            messages.append({"role": "system", "content": system_prompt})
            add_log("system_prompt", "已添加电商系统提示", {
                "prompt_length": len(system_prompt)
            })
        
        # 构建用户消息（可能包含历史上下文 + 查询改写）
        if context_prefix:
            if self.prompt_manager:
                # 如果有查询改写,使用改写后的输入
                base_message = enhanced_input if rewritten_query else user_input
                final_input = self.prompt_manager.build_user_message(base_message, context_prefix)
            else:
                base_message = enhanced_input if rewritten_query else user_input
                final_input = f"{context_prefix}\n\n# 当前用户问题\n{base_message}"
            messages.append({"role": "user", "content": final_input})
            add_log("enhanced_prompt", final_input, {
                "has_context": True,
                "has_rewrite": rewritten_query is not None,
                "context_length": len(context_prefix)
            })
        else:
            # 使用查询改写后的输入(如果有)
            final_input = enhanced_input if rewritten_query else user_input
            messages.append({"role": "user", "content": final_input})
            add_log("enhanced_prompt", final_input, {
                "has_context": False,
                "has_rewrite": rewritten_query is not None
            })
        
        tool_log: List[Dict[str, Any]] = []
        plan_lines: List[str] = []
        history: List[str] = []
        tool_call_history: List[str] = []  # 记录工具调用历史，用于检测重复
        tool_phase_announced = False

        final_answer = ""
        last_ai: Optional[Dict[str, Any]] = None
        forced_summary: Optional[str] = None

        for iteration in range(1, self.max_iterations + 1):
            forced_summary = None
            add_log("iteration_start", f"开始第 {iteration} 轮推理", {"iteration": iteration})

            requires_validation, payload_hint = self._should_require_validation(user_input, tool_log)
            if requires_validation:
                self._pending_validation = payload_hint or self._pending_validation or {"data": "", "format": "turtle"}
                self._inject_validation_reminder(messages, self._pending_validation, iteration)
                add_log(
                    "validation_required",
                    "系统检测到必须执行 ontology_validate_order",
                    {
                        "iteration": iteration,
                        "payload_hint": self._pending_validation,
                    },
                )
            
            # 记录 LLM 输入 - 完整消息和工具定义,以及类信息
            llm_class = self.llm.__class__.__name__
            llm_module = self.llm.__class__.__module__
            add_log("llm_input", {
                "messages": messages,
                "tools": self.tool_specs
            }, {
                "iteration": iteration,
                "messages_count": len(messages),
                "tools_count": len(self.tool_specs),
                "llm_class": llm_class,
                "llm_module": llm_module,
                "llm_method": "generate"
            })
            
            try:
                result = self.llm.generate(messages, tools=self.tool_specs)
            except Exception as e:
                error_msg = f"LLM 调用失败: {type(e).__name__}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self._emit_stream_event(
                    stream_handler,
                    "error",
                    error_msg,
                    {
                        "phase": "llm_generate",
                        "iteration": iteration,
                    },
                )
                add_log("llm_error", error_msg, {
                    "iteration": iteration,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "llm_class": llm_class,
                    "llm_module": llm_module
                })
                
                # 返回错误信息
                final_answer = f"抱歉，处理您的请求时遇到错误：{error_msg}"
                add_log("execution_complete", "执行因错误终止", {
                    "iterations_used": iteration,
                    "tool_calls": len(tool_log),
                    "error": True
                })
                
                return {
                    "final_answer": final_answer,
                    "plan": "\n".join(plan_lines),
                    "tool_log": tool_log,
                    "execution_log": execution_log,
                    "error": error_msg
                }
            
            # 记录 LLM 输出 - 完整响应内容和工具调用
            assistant_content = result.get("content", "")
            tool_calls = result.get("tool_calls", [])
            add_log("llm_output", {
                "content": assistant_content,
                "tool_calls_count": len(tool_calls),
                "tool_calls": tool_calls  # 完整的工具调用信息
            }, {
                "iteration": iteration
            })
            
            history.append(assistant_content)

            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
            }
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": (
                                call["arguments"]
                                if isinstance(call.get("arguments"), str)
                                else json.dumps(call.get("arguments", {}), ensure_ascii=False)
                            ),
                        },
                    }
                    for call in tool_calls
                ]
            messages.append(assistant_message)
            last_ai = assistant_message

            if not tool_calls:
                requires_validation, payload_hint = self._should_require_validation(user_input, tool_log)
                if requires_validation:
                    self._pending_validation = payload_hint or self._pending_validation or {"data": "", "format": "turtle"}
                    self._inject_validation_reminder(messages, self._pending_validation, iteration)
                    add_log(
                        "validation_guard",
                        "阻止结束对话，等待 ontology_validate_order",
                        {
                            "iteration": iteration,
                        },
                    )
                    continue

                final_answer = assistant_content
                add_log("final_answer", final_answer, {"iteration": iteration})
                break
                
            for call in tool_calls:
                tool_name = call.get("name", "")
                raw_args = call.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed_args = {"_raw": raw_args}
                else:
                    parsed_args = raw_args

                if tool_name in self.CRITICAL_TOOLS:
                    tool_meta = self.CRITICAL_TOOLS[tool_name]
                    intent_is_unknown = (
                        current_intent is None
                        or getattr(current_intent, "category", None) == IntentCategory.UNKNOWN
                    )

                    if tool_name == "commerce_create_order":
                        sanitized_args, payload_error = self._prepare_order_arguments(parsed_args)
                        if payload_error:
                            error_messages = {
                                "missing_user_id": "下单前需要绑定您的会员身份，请先告知注册手机号或用户ID。",
                                "invalid_user_id": "当前用户编号无效，请确认后重新提供。",
                                "missing_items": "订单缺少商品明细，请告诉我需要购买的商品和数量。",
                                "missing_product": "尚未识别到具体商品，请先说明想购买的型号或货号。",
                            }
                            guard_message = error_messages.get(payload_error, "创建订单所需信息不完整，请补充后再试。")
                            add_log(
                                "order_payload_guard",
                                guard_message,
                                {"iteration": iteration, "reason": payload_error}
                            )
                            self._emit_stream_event(
                                stream_handler,
                                "confirmation_required",
                                guard_message,
                                {"tool": tool_name, "reason": payload_error},
                            )
                            plan_snapshot = plan_lines + ["等待补充下单所需信息"]
                            return {
                                "final_answer": guard_message,
                                "plan": "\n".join(plan_snapshot),
                                "history": history,
                                "tool_log": tool_log,
                                "execution_log": execution_log,
                            }

                        parsed_args = sanitized_args
                        raw_args = json.dumps(parsed_args, ensure_ascii=False)

                    confirmation_prompt = self._generate_confirmation_request(tool_name, tool_meta, parsed_args)
                    if intent_is_unknown:
                        confirmation_prompt = (
                            "⚠️ 当前这句话未识别为下单/支付指令，请先明确确认。\n\n"
                            + confirmation_prompt
                        )

                    self.pending_confirmations.append({
                        "tool_name": tool_name,
                        "args": deepcopy(parsed_args),
                        "requested_at": datetime.now().isoformat(),
                    })
                    self.confirmation_mode = True
                    plan_lines.append(f"等待用户确认: {tool_name}")

                    add_log(
                        "confirmation_prompt",
                        {
                            "tool": tool_name,
                            "intent_unknown": intent_is_unknown,
                            "message": confirmation_prompt[:200],
                        },
                        {"iteration": iteration},
                    )
                    self._emit_stream_event(
                        stream_handler,
                        "confirmation_required",
                        "需要您的确认后才能继续",
                        {
                            "tool": tool_name,
                            "intent_unknown": intent_is_unknown,
                        },
                    )
                    return {
                        "final_answer": confirmation_prompt,
                        "plan": "\n".join(plan_lines),
                        "history": history,
                        "tool_log": tool_log,
                        "execution_log": execution_log,
                        "requires_confirmation": True,
                        "pending_operation": {
                            "tool_name": tool_name,
                            "args": parsed_args,
                        },
                    }

                if not tool_phase_announced:
                    self._emit_stream_event(
                        stream_handler,
                        "tool_calling_start",
                        "正在调用工具获取信息...",
                        {"iteration": iteration},
                    )
                    tool_phase_announced = True

                self._emit_stream_event(
                    stream_handler,
                    "tool_calling",
                    f"调用 {tool_name}",
                    {
                        "tool": tool_name,
                        "arguments": parsed_args,
                        "iteration": iteration,
                    },
                )
                
                # 检测重复工具调用
                tool_signature = f"{tool_name}({json.dumps(parsed_args, sort_keys=True, ensure_ascii=False)})"
                tool_call_history.append(tool_signature)
                
                # 根据工具类型设置不同的重复阈值
                # 查询类工具更严格(2次)，其他工具标准(3次)
                query_tools = [
                    "commerce_get_user_orders", "commerce_get_order_detail",
                    "commerce_get_user_info", "commerce_get_cart",
                    "commerce_search_products", "commerce_get_product_detail"
                ]
                repeat_threshold = 2 if tool_name in query_tools else 3
                
                # 如果同一工具被连续调用达到阈值，强制终止并返回结果
                if len(tool_call_history) >= repeat_threshold:
                    recent_calls = tool_call_history[-repeat_threshold:]
                    if len(set(recent_calls)) == 1:  # 最近N次调用完全相同
                        logger.warning(
                            "检测到工具 %s 被连续调用%d次(阈值=%d)，强制终止迭代",
                            tool_name, repeat_threshold, repeat_threshold
                        )
                        add_log(
                            "repeated_tool_call_guard",
                            f"工具 {tool_name} 重复调用，强制终止",
                            {"iteration": iteration, "tool_name": tool_name, "call_count": repeat_threshold, "threshold": repeat_threshold}
                        )
                        
                        # 使用最后一次工具调用的结果作为最终答案
                        if tool_log:
                            last_result = tool_log[-1].get("observation", "")
                            final_answer = f"根据查询结果：{last_result[:500]}..."
                        else:
                            final_answer = "已完成查询，请查看上方工具调用结果。"

                        self._emit_stream_event(
                            stream_handler,
                            "error",
                            f"工具 {tool_name} 重复调用，已停止后续推理",
                            {
                                "tool": tool_name,
                                "iteration": iteration,
                                "repeat_threshold": repeat_threshold,
                            },
                        )
                        add_log("final_answer", final_answer, {"iteration": iteration, "reason": "repeated_calls"})
                        break  # 跳出 for call in tool_calls 循环
                
                # 检测单个工具在整个对话中被调用超过5次
                tool_name_count = sum(1 for sig in tool_call_history if sig.startswith(f"{tool_name}("))
                if tool_name_count > 5:
                    logger.warning(
                        "工具 %s 在本轮对话中已被调用%d次，可能存在循环",
                        tool_name, tool_name_count
                    )
                    add_log(
                        "excessive_tool_calls",
                        f"工具 {tool_name} 调用次数过多",
                        {"iteration": iteration, "tool_name": tool_name, "total_calls": tool_name_count}
                    )
                
                # 获取工具类信息
                tool_obj = self.tool_map.get(tool_name)
                tool_class_info = {}
                if tool_obj:
                    tool_class_info = {
                        "class": tool_obj.__class__.__name__,
                        "module": tool_obj.__class__.__module__
                    }
                
                # 记录工具调用 - 完整参数和类信息
                logger.info("工具调用[%d]: %s, 参数: %s", iteration, tool_name, json.dumps(parsed_args, ensure_ascii=False)[:200])
                add_log("tool_call", {
                    "name": tool_name,
                    "arguments": parsed_args,
                    "tool_call_id": call.get("id"),
                    "class_info": tool_class_info
                }, {
                    "iteration": iteration,
                    "tool_id": call.get("id"),
                    "tool_class": tool_class_info.get("class", "Unknown"),
                    "tool_module": tool_class_info.get("module", "Unknown")
                })
                
                observation = self._call_tool(tool_name, raw_args)
                logger.info("工具返回[%d]: %s, 结果: %s", iteration, tool_name, str(observation)[:300])
                
                # Phase 6: 检查是否返回确认请求
                try:
                    check_confirmation = json.loads(observation)
                    if isinstance(check_confirmation, dict) and check_confirmation.get("requires_confirmation"):
                        # 拦截到关键操作,需要用户确认
                        confirmation_message = check_confirmation.get("message", "需要您的确认")
                        logger.warning("⚠️ 关键操作需要确认,终止推理循环")
                        
                        add_log("confirmation_required", {
                            "tool_name": tool_name,
                            "message": confirmation_message,
                            "risk_level": check_confirmation.get("risk_level", "unknown")
                        }, {"iteration": iteration})
                        
                        # 立即返回确认提示给用户
                        return {
                            "final_answer": confirmation_message,
                            "plan": plan_lines,
                            "history": messages,
                            "tool_log": tool_log,
                            "execution_log": execution_log,
                            "requires_confirmation": True,
                            "pending_operation": {
                                "tool_name": tool_name,
                                "args": parsed_args
                            }
                        }
                except (json.JSONDecodeError, TypeError):
                    pass  # 不是JSON或不需要确认,继续正常流程
                
                # Phase 4 优化: 记录工具调用（用于质量跟踪）
                if self.quality_tracker:
                    self.quality_tracker.record_tool_call(tool_name)
                
                # 尝试解析工具返回的元信息
                tool_result_info = {}
                sanitized_observation = None
                payload: Any = None
                try:
                    result_data = json.loads(observation)
                    if isinstance(result_data, dict):
                        payload = result_data.get("result", result_data)
                    else:
                        payload = result_data

                    if isinstance(payload, str):
                        try:
                            parsed_payload = json.loads(payload)
                            payload = parsed_payload
                            if isinstance(result_data, dict):
                                result_data["result"] = parsed_payload
                        except (json.JSONDecodeError, TypeError):
                            pass

                    custom_entries = []
                    if isinstance(payload, dict) and "_execution_log" in payload:
                        embedded_logs = payload.pop("_execution_log")
                        if isinstance(embedded_logs, dict):
                            embedded_logs = [embedded_logs]
                        if isinstance(embedded_logs, list):
                            for entry in embedded_logs:
                                if not isinstance(entry, dict):
                                    continue
                                custom_entries.append(entry)

                    for entry in custom_entries:
                        step_type = entry.get("step_type", "custom_event")
                        entry_content = entry.get("content", {})
                        entry_metadata = entry.get("metadata", {})
                        add_log(step_type, entry_content, entry_metadata)

                    if isinstance(result_data, dict) and "_tool_info" in result_data:
                        tool_result_info = result_data["_tool_info"]

                    observation_clean = self._stringify_observation(payload)

                    if isinstance(result_data, (dict, list)):
                        sanitized_observation = json.dumps(result_data, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError, ValueError):
                    observation_clean = self._stringify_observation(observation)

                if sanitized_observation is not None:
                    observation = sanitized_observation
                observation_clean = self._stringify_observation(observation_clean)
                
                # 记录工具结果 - 完整输出和执行信息
                add_log("tool_result", observation_clean, {
                    "iteration": iteration,
                    "tool_name": tool_name,
                    "tool_call_id": call.get("id"),
                    "invoked_class": tool_result_info.get("class", tool_class_info.get("class", "Unknown")),
                    "invoked_module": tool_result_info.get("module", tool_class_info.get("module", "Unknown")),
                    "invoked_method": tool_result_info.get("method", "invoke")
                })

                tool_log.append(
                    {
                        "tool": tool_name,
                        "input": parsed_args,
                        "observation": observation,
                        "iteration": iteration,
                    }
                )
                context_payload = payload if payload is not None else observation
                self._ingest_user_context_from_tool(tool_name, parsed_args, context_payload)
                summarized = self._summarize_tool_observation(tool_log[-1])
                preview_text = observation_clean
                if summarized:
                    preview_text = f"{summarized}\n\n---\n{observation_clean}"
                self._emit_stream_event(
                    stream_handler,
                    "tool_result",
                    preview_text or f"{tool_name} 返回结果",
                    {
                        "tool": tool_name,
                        "iteration": iteration,
                        "observation": observation_clean,
                        "observation_summary": summarized,
                    },
                )
                plan_lines.append(
                    f"Step {len(tool_log)} → {tool_name}({json.dumps(parsed_args, ensure_ascii=False)})"
                )
                if tool_name == "ontology_validate_order":
                    self._pending_validation = None
                    self._last_validation_iteration = iteration
                    self._validation_issued_turn = None
                    add_log(
                        "validation_completed",
                        "已执行 ontology_validate_order",
                        {"iteration": iteration},
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": tool_name,
                        "content": observation,
                    }
                )

                if tool_name == "commerce_process_payment":
                    summary_message = self._build_checkout_summary(tool_log)
                    if summary_message:
                        forced_summary = summary_message
                        plan_lines.append("完成支付并已向您汇报结果。")
                        break
            
            # 如果因重复调用而break，需要跳出主循环
            # 检查最近2次或3次调用（根据工具类型）
            if len(tool_call_history) >= 2:
                # 检查最近2次（查询类工具）
                if len(tool_call_history[-2:]) == 2 and len(set(tool_call_history[-2:])) == 1:
                    break  # 跳出 for iteration 主循环
            if len(tool_call_history) >= 3:
                # 检查最近3次（其他工具）
                if len(set(tool_call_history[-3:])) == 1:
                    break  # 跳出 for iteration 主循环

            if forced_summary:
                final_answer = forced_summary
                add_log("auto_checkout_summary", final_answer, {"iteration": iteration})
                break
        else:  # pragma: no cover - safeguards when exceeding iterations
            logger.warning("LangChain agent reached max iterations without final answer")
            add_log("max_iterations", "达到最大迭代次数", {"max_iterations": self.max_iterations})
            fallback_summary = None
            if tool_log:
                fallback_summary = self._summarize_tool_observation(tool_log[-1])

            if fallback_summary:
                final_answer = (
                    "我已拿到最新的工具数据，但系统在生成完整回答前触发了迭代上限。\n"
                    f"{fallback_summary}\n"
                    "如需更多筛选或继续查询，请告诉我具体条件。"
                )
            elif last_ai is not None and last_ai.get("content"):
                final_answer = last_ai.get("content", "")
            else:
                final_answer = "我已获取相关工具结果，但生成回复时命中了迭代上限，请尝试调整条件或稍后再试。"
            plan_lines.append("达到最大迭代次数，可能需要人工介入。")

        plan = "\n".join(plan_lines)
        
        # Phase 4: 更新对话状态
        if self.enable_conversation_state and self.state_manager:
            # 从工具调用结果更新状态
            self.state_manager.update_from_tool_results(tool_log)
            
            # 推断并更新对话阶段
            inferred_stage = self.state_manager.infer_stage_from_intent(user_input, tool_log)
            if self.state_manager.state:
                self.state_manager.state.update_stage(
                    inferred_stage,
                    reason=f"基于用户输入和{len(tool_log)}个工具调用"
                )
                self.state_manager.state.add_intent(user_input[:100])
            
            # 记录状态摘要
            state_summary = self.state_manager.get_context_summary()
            add_log("conversation_state", state_summary, {
                "stage": inferred_stage.value if inferred_stage else "unknown",
            })
        
        # 保存本轮对话到记忆
        if self.use_memory and self.memory:
            add_log("memory_save", "保存对话到记忆", {
                "user_input_length": len(user_input),
                "response_length": len(final_answer),
                "tool_calls_count": len(tool_log)
            })

            try:
                self.memory.add_turn(
                    user_input=user_input,
                    agent_response=final_answer,
                    tool_calls=tool_log,
                )
                if hasattr(self.memory, '_cache'):
                    # ChromaDB 记忆
                    logger.info("本轮对话已保存到 ChromaDB (总计 %d 轮)", len(self.memory._cache))
                    add_log("memory_saved", f"ChromaDB: 总计 {len(self.memory._cache)} 轮", {})
                elif hasattr(self.memory, 'history'):
                    # 基础记忆
                    logger.info("本轮对话已保存到记忆 (总计 %d 轮)", len(self.memory.history))
                    add_log("memory_saved", f"基础记忆: 总计 {len(self.memory.history)} 轮", {})
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("写入对话记忆失败，继续主流程: %s", exc)
                add_log("memory_save_failed", str(exc), {})
        
        # Phase 4 优化: 结束质量跟踪
        if self.quality_tracker:
            # 判断任务是否完成
            task_completed = bool(final_answer and len(tool_log) > 0)
            outcome = TaskOutcome.SUCCESS if task_completed else TaskOutcome.PARTIAL
            
            # 判断是否需要澄清（Agent 是否主动询问信息）
            needs_clarification = any(
                keyword in final_answer 
                for keyword in ["可以告诉我", "需要您", "请提供", "能否提供"]
            )
            
            # 判断是否主动引导
            proactive_guidance = any(
                keyword in final_answer
                for keyword in ["建议", "推荐", "您可以", "试试", "看看"]
            )
            
            self.quality_tracker.end_turn(
                turn_id=turn_id,
                user_input=user_input,
                agent_response=final_answer,
                task_completed=task_completed,
                outcome=outcome,
                needs_clarification=needs_clarification,
                proactive_guidance=proactive_guidance,
            )
            
            # 记录质量指标到执行日志
            quality_summary = self.quality_tracker.get_summary()
            add_log("quality_metrics", quality_summary, {})

        add_log("execution_complete", "执行完成", {
            "iterations_used": iteration + 1,
            "tool_calls": len(tool_log)
        })

        # 提取图表数据
        charts: List[Dict[str, Any]] = []
        chart_tool_calls = [entry for entry in tool_log if entry.get("tool") == "analytics_get_chart_data"]
        
        # 获取当前意图用于过滤图表
        current_intent = None
        intent_entities = {}
        if self.intent_tracker:
            intent_obj = self.intent_tracker.get_current_intent()
            if intent_obj:
                current_intent = intent_obj.category.value
                intent_entities = intent_obj.extracted_entities
        
        # 提取用户上下文（用户ID等）
        user_context_info = {}
        if hasattr(self, 'memory') and self.memory and hasattr(self.memory, 'user_context_manager'):
            ctx = self.memory.user_context_manager.get_context()
            user_context_info = {
                'user_id': ctx.user_id,
                'has_user_context': ctx.user_id is not None
            }
        
        for entry in chart_tool_calls:
            try:
                obs = entry.get("observation", "{}")
                parsed = json.loads(obs) if isinstance(obs, str) else obs
                chart_payload = parsed
                if isinstance(parsed, dict) and "result" in parsed:
                    result_section = parsed.get("result")
                    if isinstance(result_section, str):
                        try:
                            chart_payload = json.loads(result_section)
                        except json.JSONDecodeError:
                            chart_payload = result_section
                    else:
                        chart_payload = result_section
                if (
                    isinstance(chart_payload, dict)
                    and chart_payload.get("chart_type")
                    and "error" not in chart_payload
                ):
                    # 添加意图和上下文元数据到图表
                    if "metadata" not in chart_payload:
                        chart_payload["metadata"] = {}
                    chart_payload["metadata"]["intent"] = current_intent
                    chart_payload["metadata"]["intent_entities"] = intent_entities
                    chart_payload["metadata"]["user_context"] = user_context_info
                    
                    # 从工具输入中提取user_id参数
                    tool_input = entry.get("input", {})
                    if isinstance(tool_input, dict):
                        chart_payload["metadata"]["requested_user_id"] = tool_input.get("user_id")
                    
                    charts.append(chart_payload)
            except (json.JSONDecodeError, TypeError):
                logger.warning("图表工具返回内容解析失败", exc_info=True)

        if chart_tool_calls:
            if charts:
                chart_titles = [chart.get("title", chart.get("chart_type")) for chart in charts]
                logger.info(
                    "已从 %d 次图表工具调用中解析出 %d 个图表: %s",
                    len(chart_tool_calls),
                    len(charts),
                    chart_titles,
                )
            else:
                last_observation = chart_tool_calls[-1].get("observation", "")
                preview = str(last_observation)[:200]
                logger.warning(
                    "共有 %d 次图表工具调用，但未解析出可用图表。最后一次返回: %s",
                    len(chart_tool_calls),
                    preview,
                )

        chart_request_keywords = ["图表", "柱状图", "趋势图", "饼图", "对比图", "可视化"]
        chart_requested = any(keyword in user_input for keyword in chart_request_keywords)
        if chart_requested and not charts:
            logger.warning(
                "检测到图表需求但没有可用图表输出: tool_calls=%d response_preview=%s",
                len(chart_tool_calls),
                final_answer[:160] if final_answer else "",
            )

        # 构建返回结果
        result = {
            "final_answer": final_answer,
            "plan": plan,
            "history": history,
            "tool_log": tool_log,
            "raw_messages": messages,
            "execution_log": execution_log,  # 新增详细执行日志
            "charts": charts,  # 新增图表数据
        }
        
        # Phase 4 优化: 添加额外的分析信息
        if self.intent_tracker:
            result["intent_summary"] = self.intent_tracker.get_summary()
        
        if self.quality_tracker:
            result["quality_metrics"] = self.quality_tracker.get_summary()
        
        if self.state_manager and self.state_manager.state:
            result["conversation_state"] = {
                "stage": self.state_manager.state.stage.value,
                "history": [self.state_manager.state.stage.value],  # 简化版本
            }

        return result
    
    def get_memory_context(self) -> str:
        """获取当前对话记忆上下文
        
        Returns:
            str: 格式化的对话历史摘要
        """
        if not self.use_memory or not self.memory:
            return ""

        if hasattr(self.memory, 'get_context_for_prompt'):
            return self.memory.get_context_for_prompt()
        return ""

    def _filter_negative_history(self, context: str) -> str:
        """过滤掉含有图表不可用描述的历史，避免误导 LLM。"""
        if not context:
            return context
        lines = context.splitlines()
        filtered_lines = []
        removed = 0
        for line in lines:
            if any(keyword in line for keyword in self._negative_history_keywords):
                removed += 1
                continue
            filtered_lines.append(line)
        if removed:
            logger.info("过滤记忆负面记录: 移除 %d 行", removed)
        return "\n".join(filtered_lines)
    
    def get_full_history(self) -> List[Dict[str, Any]]:
        """获取完整对话历史
        
        Returns:
            List[Dict]: 历史记录列表
        """
        if not self.use_memory or not self.memory:
            return []
        
        if hasattr(self.memory, 'get_full_history'):
            return self.memory.get_full_history()
        return []
    
    def search_similar_conversations(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """搜索语义相似的历史对话
        
        Args:
            query: 查询文本
            n_results: 返回结果数
            
        Returns:
            List[Dict]: 相似对话列表
        """
        if not self.use_memory or not self.memory:
            return []
        
        if hasattr(self.memory, 'search_similar'):
            turns = self.memory.search_similar(query, n_results)
            return [turn.to_dict() for turn in turns]
        return []
    
    def clear_memory(self):
        """清空对话记忆"""
        if self.use_memory and self.memory:
            if hasattr(self.memory, 'clear_session'):
                # ChromaDB 记忆
                self.memory.clear_session()
                logger.info("ChromaDB 对话记忆已清空")
            elif hasattr(self.memory, 'clear'):
                # 基础记忆
                self.memory.clear()
                logger.info("对话记忆已清空")
    
    def save_memory(self, filepath: str):
        """保存对话记忆到文件 (仅基础记忆支持)
        
        Args:
            filepath: 保存路径
        """
        if self.use_memory and self.memory and hasattr(self.memory, 'save_to_file'):
            self.memory.save_to_file(filepath)
    
    def load_memory(self, filepath: str):
        """从文件加载对话记忆 (仅基础记忆支持)
        
        Args:
            filepath: 文件路径
        """
        if self.use_memory and self.memory and hasattr(self.memory, 'load_from_file'):
            self.memory.load_from_file(filepath)
    
    # Phase 4 优化: 质量和推荐相关方法
    
    def get_quality_report(self) -> Dict[str, Any]:
        """获取对话质量报告
        
        Returns:
            Dict: 质量指标摘要
        """
        if self.quality_tracker:
            return self.quality_tracker.get_summary()
        return {}
    
    def get_intent_analysis(self) -> Dict[str, Any]:
        """获取意图分析
        
        Returns:
            Dict: 意图跟踪摘要
        """
        if self.intent_tracker:
            return self.intent_tracker.get_summary()
        return {}
    
    def get_recommendations(
        self,
        user_id: str,
        top_n: int = 5,
        strategy: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """获取个性化推荐
        
        Args:
            user_id: 用户ID
            top_n: 返回推荐数量
            strategy: 推荐策略 ("content", "collaborative", "hybrid", "popular")
            
        Returns:
            List[Dict]: 推荐结果列表
        """
        if not self.recommendation_engine:
            return []
        
        recommendations = self.recommendation_engine.recommend(user_id, top_n, strategy)
        return [
            {
                "product_id": r.product_id,
                "product_name": r.product_name,
                "score": round(r.score, 2),
                "reason": r.reason,
                "strategy": r.strategy,
            }
            for r in recommendations
        ]
    
    def export_analytics(self) -> Dict[str, Any]:
        """导出完整的分析数据
        
        Returns:
            Dict: 包含质量、意图、推荐等所有分析数据
        """
        analytics = {
            "session_id": self.session_id,
            "quality_metrics": self.get_quality_report(),
            "intent_analysis": self.get_intent_analysis(),
        }
        
        if self.state_manager and self.state_manager.state:
            analytics["conversation_state"] = {
                "current_stage": self.state_manager.state.stage.value,
                "stage_history": [self.state_manager.state.stage.value],  # 简化：只显示当前阶段
                "user_context": self.state_manager.state.user_context.to_dict(),
                "intent_history": self.state_manager.state.intent_history,
            }
        
        if self.quality_tracker:
            analytics["quality_export"] = self.quality_tracker.export_to_json()
        
        return analytics
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息
        
        Returns:
            Dict: 统计信息
        """
        if not self.use_memory or not self.memory:
            return {"enabled": False}
        
        if hasattr(self.memory, 'get_session_stats'):
            # ChromaDB 记忆
            return {
                "enabled": True,
                "backend": "ChromaDB",
                **self.memory.get_session_stats()
            }
        elif hasattr(self.memory, 'history'):
            # 基础记忆
            return {
                "enabled": True,
                "backend": "InMemory",
                "total_turns": len(self.memory.history),
                "max_history": getattr(self.memory, 'max_history', 0),
            }
        
        return {"enabled": True, "backend": "Unknown"}
    
    def get_conversation_state(self) -> Optional[Dict[str, Any]]:
        """获取当前对话状态 (Phase 4)
        
        Returns:
            Optional[Dict]: 对话状态字典，无状态时返回 None
        """
        if not self.enable_conversation_state or not self.state_manager:
            return None
        
        state = self.state_manager.get_state()
        return state.to_dict() if state else None
    
    def get_current_stage(self) -> Optional[str]:
        """获取当前对话阶段 (Phase 4)
        
        Returns:
            Optional[str]: 对话阶段名称
        """
        if not self.enable_conversation_state or not self.state_manager:
            return None
        
        state = self.state_manager.get_state()
        return state.stage.value if state else None
    
    def run_stream(self, user_input: str):
        """执行带实时事件推送的推理循环。"""

        sentinel_done = "__final_result__"
        sentinel_error = "__fatal_error__"
        event_queue: SimpleQueue = SimpleQueue()

        def handler(event: Dict[str, Any]):
            event_queue.put(event)

        def worker():
            try:
                result = self.run(user_input, stream_handler=handler)
                event_queue.put({"step_type": sentinel_done, "result": result})
            except Exception as exc:  # pragma: no cover - catastrophic failure fallback
                message = f"执行出错: {type(exc).__name__}: {exc}"
                logger.error("Agent run_stream worker failed: %s", message, exc_info=True)
                event_queue.put({
                    "step_type": sentinel_error,
                    "content": message,
                    "metadata": {"exception": type(exc).__name__},
                })

        threading.Thread(target=worker, daemon=True).start()

        while True:
            event = event_queue.get()
            step_type = event.get("step_type")

            if step_type == sentinel_done:
                result = event.get("result", {}) or {}
                for chunk in self._stream_final_answer(result):
                    yield chunk
                break

            if step_type == sentinel_error:
                error_content = event.get("content", "执行出错")
                yield {
                    "step_type": "error",
                    "content": error_content,
                    "metadata": event.get("metadata", {}),
                }
                yield {
                    "step_type": "final_answer",
                    "content": error_content,
                    "metadata": {
                        "full_result": {},
                        "streaming": False,
                        "error": error_content,
                    },
                }
                break

            yield event


# 保持旧名称兼容
ReactAgent = LangChainAgent
