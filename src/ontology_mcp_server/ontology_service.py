from __future__ import annotations
# Copyright (c) 2025 shark8848
# MIT License
#
# Ontology MCP Server - 电商 AI 助手系统
# 本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI
#
# Author: shark8848
# Repository: https://github.com/shark8848/ontology-mcp-server
"""本体推理与同义词归一服务。"""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Any, Tuple

from rdflib import Graph, Namespace

from .config import get_settings
from .logger import get_logger

ONTO = Namespace("http://example.com/commerce#")


class OntologyService:
    """聚合折扣解释与同义词归一逻辑。"""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)
        self.logger.info("初始化 OntologyService")
        self.settings = get_settings()
        self._ontology_graph = Graph()
        self._load_ontology()
        self._synonyms = self._load_synonyms()

    def _load_ontology(self) -> None:
        ttl = self.settings.ttl_path
        if ttl.exists():
            try:
                self._ontology_graph.parse(ttl, format="turtle")
                self.logger.info("已加载本体: %s", ttl)
            except Exception as exc:
                # 示例 TTL 可能包含不兼容语法（如 SWRL）；记录并忽略
                self.logger.warning("加载本体时出错（已忽略）: %s", exc)
        else:
            self.logger.info("本体文件不存在，跳过加载: %s", ttl)

    def _load_synonyms(self) -> Dict[str, Any]:
        syn_path = self.settings.synonyms_json
        if syn_path.exists():
            try:
                with syn_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.logger.info("已加载同义词字典: %s (entries=%d)", syn_path, len(data))
                return data
            except Exception as exc:
                self.logger.exception("加载同义词失败: %s", exc)
        else:
            self.logger.info("同义词文件不存在: %s", syn_path)
        return {}

    def explain_discount(self, is_vip: bool, amount: float) -> Tuple[bool, float, str]:
        """返回折扣命中情况、折扣率及规则来源。"""
        self.logger.debug(
            "折扣推理入口[推理方式=owlready2优先]: is_vip=%s amount=%s use_owlready2=%s",
            is_vip,
            amount,
            self.settings.use_owlready2,
        )
        if self.settings.use_owlready2:
            self.logger.debug("折扣推理[推理方式=owlready2] 尝试执行")
            inferred = self._infer_with_owlready2(is_vip, amount)
            if inferred is not None:
                self.logger.info("折扣推理[推理方式=owlready2] 命中: %s", inferred)
                return inferred
        if is_vip and amount > 1000:
            self.logger.info(
                "折扣推理[推理方式=静态规则] 匹配 VIP 且总额>1000，命中折扣 0.1"
            )
            return True, 0.1, "swrl: if VIP and total>1000 then discount=0.1"
        self.logger.info("折扣推理[推理方式=静态规则] 未命中任何折扣规则")
        return False, 0.0, "no rule matched"

    def _infer_with_owlready2(self, is_vip: bool, amount: float) -> Tuple[bool, float, str] | None:
        try:
            from owlready2 import World, sync_reasoner_pellet, sync_reasoner
        except Exception:
            self.logger.warning("折扣推理[推理方式=owlready2] 无法导入依赖，跳过")
            return None
        ttl_path = self.settings.ttl_path
        if not ttl_path.exists():
            return None
        temp_path: Path | None = None
        try:
            world = World()
            source_path = ttl_path
            if ttl_path.suffix.lower() in {".ttl", ".n3", ".nt"}:
                try:
                    graph = Graph()
                    graph.parse(ttl_path, format="turtle")
                    with NamedTemporaryFile(suffix=".owl", delete=False) as tmp:
                        graph.serialize(tmp.name, format="xml")
                        temp_path = Path(tmp.name)
                        source_path = temp_path
                    self.logger.debug(
                        "折扣推理[owlready2] 已将 %s 转换为 RDF/XML 临时文件: %s",
                        ttl_path,
                        source_path,
                    )
                except Exception as exc:
                    self.logger.warning(
                        "折扣推理[owlready2] 转换 TTL -> RDF/XML 失败，直接加载原文件: %s",
                        exc,
                    )
                    temp_path = None

            onto = world.get_ontology(f"file://{source_path}").load()
            Customer = getattr(onto, "Customer", None)
            VIPCustomer = getattr(onto, "VIPCustomer", None)
            Order = getattr(onto, "Order", None)
            hasCustomer = getattr(onto, "hasCustomer", None)
            totalAmount = getattr(onto, "totalAmount", None)
            discountRate = getattr(onto, "discountRate", None)
            if not all([Customer, VIPCustomer, Order, hasCustomer, totalAmount, discountRate]):
                self.logger.warning("折扣推理[推理方式=owlready2] 缺少必需类/属性，跳过")
                return None
            with onto:
                customer = (VIPCustomer if is_vip else Customer)("_c_tmp")
                order = Order("_o_tmp")
                order.hasCustomer = [customer]
                order.totalAmount = [float(amount)]
            try:
                sync_reasoner_pellet(world=world, infer_property_values=True, infer_data_property_values=True)
            except Exception:
                self.logger.debug("折扣推理[推理方式=owlready2] pellet 不可用，回退 sync_reasoner")
                sync_reasoner(world=world, infer_property_values=True, infer_data_property_values=True)
            values = list(getattr(order, "discountRate", []) or [])
            if values:
                try:
                    rate = float(values[0])
                except Exception:
                    rate = 0.1 if is_vip and amount > 1000 else 0.0
                self.logger.info("折扣推理[推理方式=owlready2] 推断折扣率: %s", rate)
                return (rate > 0.0), rate, "owlready2: inferred by ontology rule"
        except Exception:
            self.logger.exception("折扣推理[推理方式=owlready2] 执行出错，放弃：")
            return None
        finally:
            if temp_path:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
        return None

    def normalize_product(self, text: str) -> Dict[str, Any]:
        lower = text.lower()
        # JSON 词典优先：{"canon_name": {"uri": ..., "synonyms": [...]}}
        for canon, info in self._synonyms.items():
            syns = info.get("synonyms") or []
            for s in syns:
                if s.lower() in lower:
                    self.logger.info("文本 '%s' 匹配到同义词 '%s' -> %s", text, s, canon)
                    return {
                        "canonical_name": canon,
                        "uri": info.get("uri"),
                        "matched_synonym": s,
                    }
            if canon.lower() in lower:
                self.logger.info("文本 '%s' 直接匹配到规范名 -> %s", text, canon)
                return {
                    "canonical_name": canon,
                    "uri": info.get("uri"),
                    "matched_synonym": canon,
                }
        # 兜底：若 JSON 为空，可尝试 TTL 或直接回传原文
        self.logger.info("未找到匹配，返回原文作为兜底: %s", text)
        return {
            "canonical_name": text.strip() or "未知商品",
            "uri": None,
            "matched_synonym": None,
        }


_service: OntologyService | None = None


def get_ontology_service() -> OntologyService:
    global _service
    if _service is None:
        _service = OntologyService()
    return _service
