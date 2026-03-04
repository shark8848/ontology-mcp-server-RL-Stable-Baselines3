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
"""对话记忆配置加载模块"""

import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import yaml

from agent.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ChromaDBConfig:
    """ChromaDB 配置"""
    persist_directory: str = "data/chroma_memory"
    collection_name: str = "conversation_memory"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_provider: str = "default"  # default | ollama | openai_compatible
    embedding_api_url: Optional[str] = None
    embedding_api_key: Optional[str] = None


@dataclass
class RetrievalStrategyConfig:
    """记忆检索策略配置"""
    retrieval_mode: str = "recent"  # "recent" 或 "similarity"
    max_recent_turns: int = 10
    max_similarity_results: int = 5
    similarity_threshold: float = 0.5
    enable_llm_summary: bool = False


@dataclass
class SummaryConfig:
    """摘要生成策略配置"""
    trigger: str = "threshold"  # "always", "threshold", "manual"
    turns_threshold: int = 5
    text_length_threshold: int = 500
    max_summary_length: int = 200


@dataclass
class SessionConfig:
    """会话管理配置"""
    default_session_prefix: str = "session"
    timeout: int = 0  # 0 表示永不超时
    auto_cleanup: bool = False


@dataclass
class PerformanceConfig:
    """性能优化配置"""
    enable_cache: bool = True
    cache_size: int = 100
    batch_size: int = 10


@dataclass
class MemoryConfig:
    """对话记忆完整配置"""
    enabled: bool = True
    backend: str = "chromadb"  # "chromadb" 或 "basic"
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)
    strategy: RetrievalStrategyConfig = field(default_factory=RetrievalStrategyConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MemoryConfig":
        """从字典创建配置"""
        if not config_dict:
            return cls()
        
        # 递归创建子配置
        chromadb_config = ChromaDBConfig(**config_dict.get("chromadb", {}))
        
        # strategy 和 summary 分开处理
        strategy_dict = config_dict.get("strategy", {})
        # 移除嵌套的 summary 字段(如果存在)
        strategy_dict_clean = {k: v for k, v in strategy_dict.items() if k != "summary"}
        strategy_config = RetrievalStrategyConfig(**strategy_dict_clean)
        
        summary_config = SummaryConfig(**config_dict.get("summary", {}))
        session_config = SessionConfig(**config_dict.get("session", {}))
        performance_config = PerformanceConfig(**config_dict.get("performance", {}))
        
        return cls(
            enabled=config_dict.get("enabled", True),
            backend=config_dict.get("backend", "chromadb"),
            chromadb=chromadb_config,
            strategy=strategy_config,
            summary=summary_config,
            session=session_config,
            performance=performance_config,
        )


class MemoryConfigLoader:
    """记忆配置加载器"""
    
    _instance: Optional["MemoryConfigLoader"] = None
    _config: Optional[MemoryConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = self._load_config()
    
    def _load_config(self) -> MemoryConfig:
        """加载配置（优先级: 环境变量 > YAML 文件 > 默认值）"""
        config_dict = {}
        
        # 1. 尝试从 YAML 文件加载
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "config.yaml"
        )
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    yaml_config = yaml.safe_load(f) or {}
                    config_dict = yaml_config.get("memory", {})
                    LOGGER.info("从 YAML 加载记忆配置: %s", config_path)
            except Exception as e:
                LOGGER.warning("无法加载 YAML 配置: %s", e)
        
        # 2. 环境变量覆盖 (支持常用配置项)
        env_overrides = {
            "enabled": os.getenv("MEMORY_ENABLED"),
            "backend": os.getenv("MEMORY_BACKEND"),
        }
        
        # 过滤 None 值
        env_overrides = {k: v for k, v in env_overrides.items() if v is not None}
        
        # 类型转换
        if "enabled" in env_overrides:
            env_overrides["enabled"] = env_overrides["enabled"].lower() in ("true", "1", "yes")
        
        config_dict.update(env_overrides)
        
        # 3. ChromaDB 路径环境变量
        if os.getenv("CHROMA_PERSIST_DIR"):
            if "chromadb" not in config_dict:
                config_dict["chromadb"] = {}
            config_dict["chromadb"]["persist_directory"] = os.getenv("CHROMA_PERSIST_DIR")
        
        # 4. 检索策略环境变量
        if os.getenv("MEMORY_RETRIEVAL_MODE"):
            if "strategy" not in config_dict:
                config_dict["strategy"] = {}
            config_dict["strategy"]["retrieval_mode"] = os.getenv("MEMORY_RETRIEVAL_MODE")
        
        if os.getenv("MEMORY_MAX_TURNS"):
            if "strategy" not in config_dict:
                config_dict["strategy"] = {}
            try:
                config_dict["strategy"]["max_recent_turns"] = int(os.getenv("MEMORY_MAX_TURNS"))
            except ValueError:
                pass
        
        LOGGER.info("记忆配置加载完成: backend=%s, mode=%s", 
                   config_dict.get("backend", "chromadb"),
                   config_dict.get("strategy", {}).get("retrieval_mode", "recent"))
        
        return MemoryConfig.from_dict(config_dict)
    
    @property
    def config(self) -> MemoryConfig:
        """获取配置"""
        return self._config
    
    def reload(self) -> MemoryConfig:
        """重新加载配置"""
        self._config = self._load_config()
        return self._config


# 全局配置实例
def get_memory_config() -> MemoryConfig:
    """获取记忆配置（单例模式）"""
    loader = MemoryConfigLoader()
    return loader.config


# 便捷函数
def is_memory_enabled() -> bool:
    """检查记忆功能是否启用"""
    return get_memory_config().enabled


def get_backend_type() -> str:
    """获取记忆后端类型"""
    return get_memory_config().backend


def use_chromadb() -> bool:
    """是否使用 ChromaDB"""
    config = get_memory_config()
    return config.enabled and config.backend == "chromadb"


def use_similarity_search() -> bool:
    """是否使用语义相似度检索"""
    config = get_memory_config()
    return config.strategy.retrieval_mode == "similarity"


def get_persist_directory() -> str:
    """获取持久化目录"""
    return get_memory_config().chromadb.persist_directory


def get_max_results() -> int:
    """获取最大检索结果数"""
    config = get_memory_config()
    if config.strategy.retrieval_mode == "recent":
        return config.strategy.max_recent_turns
    else:
        return config.strategy.max_similarity_results
