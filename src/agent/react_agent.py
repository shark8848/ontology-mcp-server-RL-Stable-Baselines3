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
from typing import Any, Dict, List, Optional, Tuple

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
        max_iterations: int = 6,
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
            self.intent_tracker = IntentTracker(session_id=self.session_id)
            logger.info("已启用意图识别跟踪")
        
        # Phase 4 优化: 推荐引擎
        self.enable_recommendation = enable_recommendation
        self.recommendation_engine: Optional[RecommendationEngine] = None
        if enable_recommendation:
            self.recommendation_engine = RecommendationEngine()
            logger.info("已启用个性化推荐引擎")

        # 强制本体/SHACL 校验相关状态
        self._pending_validation: Optional[Dict[str, Any]] = None
        self._validation_issued_turn: Optional[int] = None
        self._last_validation_iteration: Optional[int] = None
        
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
            "【系统强制校验】检测到用户已进入结算/下单阶段。"
            "请立即调用工具 `ontology_validate_order`，确保订单数据通过 SHACL 校验。"
            "如需示例参数，可参考: "
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        messages.append({"role": "system", "content": reminder})
        self._validation_issued_turn = iteration

    def _call_tool(self, name: str, args: Dict[str, Any]) -> str:
        """调用工具并返回结果"""
        tool = self.tool_map.get(name)
        if tool is None:
            logger.warning("LLM attempted to call unknown tool: %s", name)
            return f"未找到工具 {name}"
        
        # 记录工具对象信息
        tool_class = tool.__class__.__name__
        tool_module = tool.__class__.__module__
        
        try:
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            
            # 记录解析前的参数
            logger.debug(f"调用工具 {name} (类: {tool_module}.{tool_class})")
            
            parsed = tool.parse_arguments(args)
            result = tool.invoke(parsed)
            
            # 返回结果和元信息
            return json.dumps({
                "_tool_info": {
                    "class": tool_class,
                    "module": tool_module,
                    "method": "invoke"
                },
                "result": result if isinstance(result, (dict, list, str, int, float, bool)) else str(result)
            }, ensure_ascii=False)
            
        except Exception as exc:  # pragma: no cover - defensive runtime logging
            logger.exception("Tool %s invocation failed", name)
            return json.dumps({
                "_tool_info": {
                    "class": tool_class,
                    "module": tool_module,
                    "method": "invoke"
                },
                "error": f"调用失败: {type(exc).__name__}: {str(exc)}"
            }, ensure_ascii=False)

    def run(self, user_input: str) -> Dict[str, Any]:
        """执行 Agent 推理循环
        
        Args:
            user_input: 用户输入
            
        Returns:
            Dict: 包含 final_answer, plan, history, tool_log, execution_log 等信息
        """

        logger.info("LangChain agent received input: %s", user_input)
        
        # Phase 4 优化: 开始质量跟踪
        if self.quality_tracker:
            self.quality_tracker.start_turn()
        
        # Phase 4 优化: 跟踪用户意图
        turn_id = len(self.quality_tracker.session_metrics.turns) + 1 if self.quality_tracker else 0
        current_intent = None
        if self.intent_tracker:
            current_intent = self.intent_tracker.track_intent(user_input, turn_id)
            logger.info(f"识别意图: {current_intent.category.value} (置信度: {current_intent.confidence:.2f})")
        
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
                add_log("memory_context", context_prefix, {
                    "length": len(context_prefix)
                })
        elif self.use_memory and self.memory and hasattr(self.memory, 'get_context_for_prompt'):
            # 基础记忆
            context_prefix = self.memory.get_context_for_prompt()
            if context_prefix:
                logger.info("注入对话历史上下文: %d 字符", len(context_prefix))
                add_log("memory_context", context_prefix, {
                    "length": len(context_prefix)
                })
        
        # 构造系统提示(如果有历史上下文)
        messages: List[Dict[str, Any]] = []
        
        # Phase 4: 添加系统提示
        if self.enable_system_prompt and self.prompt_manager:
            system_prompt = self.prompt_manager.get_system_prompt()
            messages.append({"role": "system", "content": system_prompt})
            add_log("system_prompt", "已添加电商系统提示", {
                "prompt_length": len(system_prompt)
            })
        
        # 构建用户消息（可能包含历史上下文）
        if context_prefix:
            if self.prompt_manager:
                enhanced_input = self.prompt_manager.build_user_message(user_input, context_prefix)
            else:
                enhanced_input = f"{context_prefix}\n\n# 当前用户问题\n{user_input}"
            messages.append({"role": "user", "content": enhanced_input})
            add_log("enhanced_prompt", enhanced_input, {
                "has_context": True,
                "context_length": len(context_prefix)
            })
        else:
            messages.append({"role": "user", "content": user_input})
            add_log("enhanced_prompt", user_input, {"has_context": False})
        
        tool_log: List[Dict[str, Any]] = []
        plan_lines: List[str] = []
        history: List[str] = []

        final_answer = ""
        last_ai: Optional[Dict[str, Any]] = None

        for iteration in range(1, self.max_iterations + 1):
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
                
                # 获取工具类信息
                tool_obj = self.tool_map.get(tool_name)
                tool_class_info = {}
                if tool_obj:
                    tool_class_info = {
                        "class": tool_obj.__class__.__name__,
                        "module": tool_obj.__class__.__module__
                    }
                
                # 记录工具调用 - 完整参数和类信息
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
                
                # Phase 4 优化: 记录工具调用（用于质量跟踪）
                if self.quality_tracker:
                    self.quality_tracker.record_tool_call(tool_name)
                
                # 尝试解析工具返回的元信息
                tool_result_info = {}
                sanitized_observation = None
                try:
                    result_data = json.loads(observation)
                    payload: Any
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

                    if isinstance(payload, (dict, list)):
                        observation_clean = json.dumps(payload, ensure_ascii=False)
                    else:
                        observation_clean = str(payload)

                    if isinstance(result_data, (dict, list)):
                        sanitized_observation = json.dumps(result_data, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError, ValueError):
                    observation_clean = observation

                if sanitized_observation is not None:
                    observation = sanitized_observation
                
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
        else:  # pragma: no cover - safeguards when exceeding iterations
            logger.warning("LangChain agent reached max iterations without final answer")
            add_log("max_iterations", "达到最大迭代次数", {"max_iterations": self.max_iterations})
            if last_ai is not None:
                final_answer = last_ai.get("content", "")
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

        # 构建返回结果
        result = {
            "final_answer": final_answer,
            "plan": plan,
            "history": history,
            "tool_log": tool_log,
            "raw_messages": messages,
            "execution_log": execution_log,  # 新增详细执行日志
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


# 保持旧名称兼容
ReactAgent = LangChainAgent
