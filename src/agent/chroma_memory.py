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
"""基于 ChromaDB 的对话记忆管理，支持向量检索和持久化存储。
功能:
1. 使用 ChromaDB 持久化存储对话历史
2. 支持语义相似度检索
3. 自动生成对话摘要
4. 支持多会话管理
"""

import os
import json
import re
import shutil
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None

from agent.logger import get_logger
from agent.memory_config import get_memory_config
from agent.user_context_extractor import UserContextManager

LOGGER = get_logger(__name__)


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    turn_id: str  # 唯一标识符
    session_id: str  # 会话ID
    user_input: str
    agent_response: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class ChromaConversationMemory:
    """基于 ChromaDB 的对话记忆管理器
    
    特性:
    - 持久化存储到本地磁盘
    - 支持向量相似度检索
    - 自动生成摘要和元数据
    - 多会话隔离
    """
    
    def __init__(
        self,
        session_id: str = None,
        persist_directory: str = None,
        collection_name: str = None,
        max_results: int = None,
        llm_model=None,
        config=None,
    ):
        """初始化 Chroma 记忆管理器
        
        Args:
            session_id: 会话ID，用于隔离不同对话
            persist_directory: 持久化目录，默认从配置读取
            collection_name: Chroma collection 名称，默认从配置读取
            max_results: 检索时返回的最大结果数，默认从配置读取
            llm_model: 用于生成摘要的 LLM 实例(可选)
            config: 记忆配置对象，默认使用全局配置
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
        
        # 加载配置
        if config is None:
            config = get_memory_config()
        self.config = config
        
        # 应用配置（参数优先级高于配置文件）
        self.session_id = session_id or f"{config.session.default_session_prefix}_{id(self)}"
        self.collection_name = collection_name or config.chromadb.collection_name
        
        # 根据检索模式选择 max_results
        if max_results is None:
            if config.strategy.retrieval_mode == "recent":
                max_results = config.strategy.max_recent_turns
            else:
                max_results = config.strategy.max_similarity_results
        self.max_results = max_results
        
        # LLM 摘要配置
        self.llm_model = llm_model
        self.enable_llm_summary = config.strategy.enable_llm_summary and llm_model is not None
        
        # 持久化目录
        if persist_directory is None:
            persist_directory = config.chromadb.persist_directory
            # 支持相对路径
            if not os.path.isabs(persist_directory):
                persist_directory = os.path.join(
                    os.path.dirname(__file__), 
                    "..", "..", 
                    persist_directory
                )
        
        self.persist_directory = os.path.abspath(persist_directory)
        
        # 确保目录存在
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # 获取或创建 collection（兼容新版本的 metadata 限制）
        self.collection = self._create_collection_with_retry()
        
        # 内存缓存(用于快速访问当前会话)
        self._cache: List[ConversationTurn] = []
        if config.performance.enable_cache:
            self._load_session_cache()
        
        # 用户上下文管理器
        self.user_context_manager = UserContextManager(self.session_id)
        
        LOGGER.info(
            "ChromaDB 记忆初始化: session=%s, persist_dir=%s, collection=%s, mode=%s",
            self.session_id, self.persist_directory, self.collection_name,
            config.strategy.retrieval_mode
        )

    def _create_collection_with_retry(self):
        """创建 collection，处理 legacy metadata 兼容逻辑"""
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Conversation memory storage"}
            )
        except Exception as exc:
            if self._is_reserved_metadata_error(exc):
                LOGGER.warning(
                    "检测到旧版 Chroma 元数据格式(_type)与当前版本不兼容，将自动备份并重建索引"
                )
                self._backup_and_reset_store()
                return self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Conversation memory storage"}
                )
            LOGGER.error("Failed to create Chroma collection: %s", exc)
            raise

    @staticmethod
    def _is_reserved_metadata_error(error: Exception) -> bool:
        message = str(error) if error else ""
        return "_type" in message

    def _backup_and_reset_store(self) -> None:
        """备份旧版数据并重置客户端，避免 _type 元数据导致的初始化失败"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(self.persist_directory.rstrip(os.sep)) or "chroma_memory"
        parent_dir = os.path.dirname(self.persist_directory.rstrip(os.sep)) or os.getcwd()
        backup_dir = os.path.join(parent_dir, f"{base_name}_legacy_{timestamp}")
        try:
            if os.path.exists(self.persist_directory):
                shutil.copytree(self.persist_directory, backup_dir, dirs_exist_ok=False)
                LOGGER.warning("已备份旧 Chroma 数据到: %s", backup_dir)
        except Exception as backup_error:
            LOGGER.warning("备份旧 Chroma 数据失败: %s", backup_error)
        try:
            self.client.reset()
            LOGGER.warning("旧 Chroma 索引已清空，已准备重新创建 collection")
        except Exception as reset_error:
            LOGGER.error("重置 Chroma 客户端失败: %s", reset_error)
            raise
    
    def _load_session_cache(self):
        """从 ChromaDB 加载当前会话的历史到缓存"""
        try:
            # 查询当前会话的所有记录
            results = self.collection.get(
                where={"session_id": self.session_id},
                include=["metadatas", "documents"]
            )
            
            if not results["ids"]:
                LOGGER.info("当前会话无历史记录")
                return
            
            # 重建缓存
            self._cache.clear()
            for i, turn_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                doc = results["documents"][i]
                
                turn = ConversationTurn(
                    turn_id=turn_id,
                    session_id=metadata["session_id"],
                    user_input=metadata["user_input"],
                    agent_response=metadata["agent_response"],
                    tool_calls=json.loads(metadata.get("tool_calls", "[]")),
                    timestamp=metadata["timestamp"],
                    summary=doc,  # document 存储摘要
                    metadata=metadata,
                )
                self._cache.append(turn)
            
            # 按时间排序
            self._cache.sort(key=lambda x: x.timestamp)
            
            LOGGER.info("从 ChromaDB 加载 %d 条历史记录到缓存", len(self._cache))
            
        except Exception as e:
            LOGGER.warning("加载会话缓存失败: %s", e)
    
    def _generate_summary(self, turn: ConversationTurn) -> str:
        """生成对话摘要
        
        Args:
            turn: 对话轮次
            
        Returns:
            str: 摘要文本
        """
        # 基础规则摘要
        user_summary = turn.user_input[:100]
        
        tool_summary = ""
        if turn.tool_calls:
            tool_names = [tc.get("tool", "unknown") for tc in turn.tool_calls]
            tool_summary = f", 调用工具: {', '.join(tool_names)}"
        
        response_summary = turn.agent_response[:50]
        if len(turn.agent_response) > 50:
            response_summary += "..."
        
        summary = f"用户: {user_summary}{tool_summary} → {response_summary}"
        
        # 如果有 LLM，尝试生成更好的摘要
        if self.llm_model:
            try:
                summary_prompt = f"""请为以下对话生成简洁摘要(不超过50字):

用户: {turn.user_input}
Agent: {turn.agent_response}

摘要:"""
                messages = [{"role": "user", "content": summary_prompt}]
                response = self.llm_model.generate(messages, tools=[])
                llm_summary = response.get("content", "").strip()
                
                if llm_summary:
                    summary = llm_summary
                    LOGGER.debug("使用 LLM 生成摘要")
            except Exception as e:
                LOGGER.warning("LLM 摘要生成失败: %s, 使用基础摘要", e)
        
        return summary
    
    def add_turn(
        self, 
        user_input: str, 
        agent_response: str, 
        tool_calls: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None,
    ) -> ConversationTurn:
        """添加一轮对话到记忆
        
        Args:
            user_input: 用户输入
            agent_response: Agent 响应
            tool_calls: 工具调用列表
            metadata: 额外元数据
            
        Returns:
            ConversationTurn: 新增的对话记录
        """
        # 生成唯一ID
        turn_id = f"{self.session_id}_{datetime.now().timestamp()}"
        
        # 创建对话记录
        turn = ConversationTurn(
            turn_id=turn_id,
            session_id=self.session_id,
            user_input=user_input,
            agent_response=agent_response,
            tool_calls=tool_calls or [],
            metadata=metadata or {},
        )
        
        # 🎯 更新用户上下文（动态提取关键信息）
        self.user_context_manager.update_from_conversation(
            user_input, agent_response, tool_calls
        )

        # 对于 create_order 工具，强制从 observation 或 input 中提取并显式设置最近订单号，避免误覆盖或遗漏
        try:
            if tool_calls:
                for tc in tool_calls:
                    tname = (tc.get('tool') or '').lower()
                    if 'create_order' in tname:
                        obs = tc.get('observation') or ''
                        inp = tc.get('input') or {}
                        # 优先从 observation 中提取 ORD 格式订单号
                        m = re.search(r"\b(ORD\d{15,})\b", str(obs))
                        if m:
                            self.user_context_manager.set_recent_order(m.group(1))
                            continue

                        # 如果 observation 中没有，从 input（可能是字符串或dict）中尝试提取
                        if isinstance(inp, str):
                            m2 = re.search(r"\b(ORD\d{15,})\b", inp)
                            if m2:
                                self.user_context_manager.set_recent_order(m2.group(1))
                                continue
                        elif isinstance(inp, dict):
                            # 有些工具返回 order_id 字段
                            oid = inp.get('order_id') or inp.get('order_no') or inp.get('order')
                            if oid and isinstance(oid, str):
                                m3 = re.search(r"\b(ORD\d{15,})\b", oid)
                                if m3:
                                    self.user_context_manager.set_recent_order(m3.group(1))
                                    continue
        except Exception:
            LOGGER.exception("在 create_order 后强制设置最近订单号时出错")
        
        # 生成摘要
        turn.summary = self._generate_summary(turn)
        
        # 准备元数据
        chroma_metadata = {
            "session_id": self.session_id,
            "user_input": user_input[:500],  # 限制长度避免超限
            "agent_response": agent_response[:500],
            "tool_calls": json.dumps(tool_calls or []),
            "timestamp": turn.timestamp,
        }
        # 合并额外元数据
        if metadata:
            chroma_metadata.update(metadata)
        chroma_metadata = self._sanitize_metadata(chroma_metadata)
        
        # 存储到 ChromaDB
        try:
            self.collection.add(
                ids=[turn_id],
                documents=[turn.summary],  # 摘要作为可检索文档
                metadatas=[chroma_metadata],
            )
            LOGGER.info("新增对话记录: turn_id=%s", turn_id)
        except Exception as e:
            LOGGER.error("存储对话记录失败: %s", e)
            raise
        
        # 更新缓存
        self._cache.append(turn)
        
        return turn

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """过滤掉以下划线开头的键，并确保值可序列化"""
        if not metadata:
            return {}
        sanitized: Dict[str, Any] = {}
        for key, value in metadata.items():
            sanitized_key = str(key)
            if sanitized_key.startswith("_"):
                sanitized_key = sanitized_key.lstrip("_") or "meta"
                LOGGER.debug("检测到以下划线开头的元数据键，已自动重命名: %s", key)
            sanitized[sanitized_key] = self._coerce_metadata_value(value)
        return sanitized

    @staticmethod
    def _coerce_metadata_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    
    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """获取最近N轮对话
        
        Args:
            n: 返回的对话轮数
            
        Returns:
            List[ConversationTurn]: 对话列表
        """
        return self._cache[-n:] if self._cache else []
    
    def search_similar(
        self, 
        query: str, 
        n_results: int = None
    ) -> List[ConversationTurn]:
        """基于语义相似度搜索历史对话
        
        Args:
            query: 查询文本
            n_results: 返回结果数量，默认使用 max_results
            
        Returns:
            List[ConversationTurn]: 相似对话列表
        """
        if n_results is None:
            n_results = self.max_results
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, len(self._cache)),
                where={"session_id": self.session_id},
                include=["metadatas", "documents", "distances"]
            )
            
            if not results["ids"] or not results["ids"][0]:
                return []
            
            # 重建对话记录
            similar_turns = []
            for i, turn_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                doc = results["documents"][0][i]
                distance = results["distances"][0][i]
                
                turn = ConversationTurn(
                    turn_id=turn_id,
                    session_id=metadata["session_id"],
                    user_input=metadata["user_input"],
                    agent_response=metadata["agent_response"],
                    tool_calls=json.loads(metadata.get("tool_calls", "[]")),
                    timestamp=metadata["timestamp"],
                    summary=doc,
                    metadata={**metadata, "similarity_distance": distance},
                )
                similar_turns.append(turn)
            
            LOGGER.debug("相似度检索返回 %d 条结果", len(similar_turns))
            return similar_turns
            
        except Exception as e:
            LOGGER.error("相似度检索失败: %s", e)
            return []
    
    def get_context_for_prompt(
        self, 
        use_similarity: bool = False,
        query: str = None,
        max_turns: int = 5,
    ) -> str:
        """获取用于注入 prompt 的上下文
        
        Args:
            use_similarity: 是否使用相似度检索
            query: 查询文本(仅在 use_similarity=True 时使用)
            max_turns: 最大返回轮数
            
        Returns:
            str: 格式化的上下文字符串
        """
        context_parts = []
        
        # 🎯 第一部分：用户上下文信息（关键！）
        user_context = self.user_context_manager.get_prompt_injection()
        if user_context:
            context_parts.append(user_context)
        
        # 第二部分：对话历史
        if use_similarity and query:
            turns = self.search_similar(query, max_turns)
        else:
            turns = self.get_recent_turns(max_turns)
        
        if turns:
            context_lines = ["**对话历史摘要**:"]
            for i, turn in enumerate(turns, 1):
                # 显示完整的用户问题和摘要
                user_q = turn.user_input[:80]
                if len(turn.user_input) > 80:
                    user_q += "..."
                context_lines.append(f"{i}. 用户: {user_q}")
                context_lines.append(f"   回复: {turn.summary}")
            
            context_parts.append("\n".join(context_lines))
            LOGGER.debug("生成上下文: %d 轮摘要", len(turns))
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def get_full_history(self) -> List[Dict[str, Any]]:
        """获取当前会话的完整历史
        
        Returns:
            List[Dict]: 历史记录列表
        """
        return [turn.to_dict() for turn in self._cache]
    
    def clear_session(self):
        """清空当前会话的记忆"""
        try:
            # 获取当前会话的所有记录ID
            results = self.collection.get(
                where={"session_id": self.session_id}
            )
            
            if results["ids"]:
                # 删除记录
                self.collection.delete(ids=results["ids"])
                LOGGER.info("清空会话记忆: 删除 %d 条记录", len(results["ids"]))
            
            # 清空缓存
            self._cache.clear()
            
            # 🎯 清空用户上下文
            self.user_context_manager.clear()
            
        except Exception as e:
            LOGGER.error("清空会话失败: %s", e)
    
    def delete_collection(self):
        """删除整个 collection（慎用）"""
        try:
            self.client.delete_collection(name=self.collection_name)
            LOGGER.warning("已删除 collection: %s", self.collection_name)
        except Exception as e:
            LOGGER.error("删除 collection 失败: %s", e)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计信息"""
        user_ctx = self.user_context_manager.get_context()
        return {
            "session_id": self.session_id,
            "total_turns": len(self._cache),
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name,
            "user_context": {
                "user_id": user_ctx.user_id,
                "has_phone": bool(user_ctx.phone),
                "order_count": len(user_ctx.order_ids),
                "product_count": len(user_ctx.viewed_product_ids),
            }
        }
    
    @classmethod
    def list_sessions(cls, persist_directory: str = None) -> List[str]:
        """列出所有会话ID
        
        Args:
            persist_directory: 持久化目录
            
        Returns:
            List[str]: 会话ID列表
        """
        if not CHROMA_AVAILABLE:
            return []
        
        if persist_directory is None:
            persist_directory = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "data", "chroma_memory"
            )
        
        try:
            client = chromadb.PersistentClient(path=persist_directory)
            collections = client.list_collections()
            
            sessions = set()
            for col in collections:
                results = col.get(include=["metadatas"])
                for metadata in results.get("metadatas", []):
                    if "session_id" in metadata:
                        sessions.add(metadata["session_id"])
            
            return sorted(sessions)
            
        except Exception as e:
            LOGGER.error("列出会话失败: %s", e)
            return []


# 向后兼容的别名
ConversationMemory = ChromaConversationMemory
