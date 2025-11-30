from __future__ import annotations
# Copyright (c) 2025 shark8848
# MIT License
#
# Ontology MCP Server - 电商 AI 助手系统
# 本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI
#
# Author: shark8848
# Repository: https://github.com/shark8848/ontology-mcp-server
"""配置管理：集中管理环境变量与资源路径。"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import yaml

from .logger import get_logger

LOGGER = get_logger(__name__)


class Settings:
    """读取环境变量，提供默认路径。"""

    def __init__(self) -> None:
        base = Path(
            os.getenv("ONTOLOGY_DATA_DIR", Path(__file__).resolve().parent.parent / "data")
        )
        self.data_dir = base
        self.ttl_path = Path(os.getenv("ONTOLOGY_TTL", base / "ontology_commerce.ttl"))
        self.shapes_path = Path(os.getenv("ONTOLOGY_SHAPES", base / "ontology_shapes.ttl"))
        self.synonyms_json = Path(os.getenv("ONTOLOGY_SYNONYMS_JSON", base / "product_synonyms.json"))
        self.synonyms_ttl = Path(os.getenv("ONTOLOGY_SYNONYMS_TTL", base / "product_synonyms.ttl"))
        self.capabilities_jsonld = Path(os.getenv("ONTOLOGY_CAPABILITIES_JSONLD", base / "capabilities.jsonld"))
        audit_override = os.getenv("ONTOLOGY_MCP_AUDIT", "")
        self.audit_file = Path(audit_override) if audit_override else None

        self._config_data, self.config_path = self._load_yaml_config()
        self.use_owlready2 = self._resolve_bool_setting(
            env_var="ONTOLOGY_USE_OWLREADY2",
            config_keys=("ontology", "use_owlready2"),
            default=True,
        )

    # ------------------------------------------------------------------
    # 配置解析辅助方法
    # ------------------------------------------------------------------
    def _load_yaml_config(self) -> Tuple[Dict[str, Any], Optional[Path]]:
        """加载 YAML 配置（config.yaml），若不存在则返回空字典。"""

        candidate_paths = []
        override_path = os.getenv("ONTOLOGY_CONFIG_YAML")
        if override_path:
            candidate_paths.append(Path(override_path))

        project_root = Path(__file__).resolve().parents[2]
        candidate_paths.append(project_root / "src" / "agent" / "config.yaml")
        candidate_paths.append(project_root / "config" / "config.yaml")

        for path in candidate_paths:
            if not path or not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
                LOGGER.info("已加载配置文件: %s", path)
                return data, path
            except Exception as exc:
                LOGGER.warning("读取配置文件失败 (%s): %s", path, exc)
        return {}, None

    def _resolve_bool_setting(
        self,
        *,
        env_var: str,
        config_keys: Sequence[str],
        default: bool,
    ) -> bool:
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value.lower() in {"1", "true", "yes"}

        config_value = self._get_config_value(config_keys)
        if config_value is None:
            return default
        if isinstance(config_value, bool):
            return config_value
        if isinstance(config_value, str):
            return config_value.lower() in {"1", "true", "yes"}
        return bool(config_value)

    def _get_config_value(self, keys: Sequence[str]) -> Any:
        data: Any = self._config_data
        for key in keys:
            if not isinstance(data, dict) or key not in data:
                return None
            data = data[key]
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
