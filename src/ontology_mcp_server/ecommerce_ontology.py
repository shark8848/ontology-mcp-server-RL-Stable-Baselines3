from __future__ import annotations
# Copyright (c) 2025 shark8848
# MIT License
#
# Ontology MCP Server - 电商 AI 助手系统
# 本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI
#
# Author: shark8848
# Repository: https://github.com/shark8848/ontology-mcp-server
"""
电商本体推理服务

基于 RDFLib 实现电商领域的本体推理，包括：
- 用户等级推理 (VIP/SVIP)
- 折扣计算推理
- 物流策略推理
- 退换货规则推理
"""

import ast
import re
from decimal import Decimal
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

from .logger import get_logger

LOGGER = get_logger(__name__)

# 命名空间定义
EC = Namespace("http://example.org/ecommerce#")
RULE = Namespace("http://example.org/rules#")


class EcommerceOntologyService:
    """电商本体推理服务"""
    
    def __init__(self, ontology_path: str = "data/ontology_ecommerce.ttl",
                 rules_path: str = "data/ontology_rules.ttl"):
        """初始化本体推理服务
        
        Args:
            ontology_path: 电商本体文件路径
            rules_path: 业务规则文件路径
        """
        self.ontology_graph = Graph()
        self.rules_graph = Graph()
        
        # 加载本体
        self._load_ontology(ontology_path)
        self._load_rules(rules_path)
        self._user_level_rules = self._load_user_level_rules()
        self._discount_rules = self._load_discount_rules()
        self._shipping_rules = self._load_shipping_rules()
        self._return_rules = self._load_return_rules()
        self._cancellation_rules = self._load_cancellation_rules()
        
        LOGGER.info("电商本体推理服务已初始化")
    
    def _load_ontology(self, path: str):
        """加载电商本体"""
        ontology_file = Path(path)
        if ontology_file.exists():
            try:
                self.ontology_graph.parse(ontology_file, format="turtle")
                LOGGER.info(f"已加载电商本体: {path}, 三元组数: {len(self.ontology_graph)}")
            except Exception as e:
                LOGGER.error(f"加载电商本体失败: {e}")
        else:
            LOGGER.warning(f"电商本体文件不存在: {path}")
    
    def _load_rules(self, path: str):
        """加载业务规则"""
        rules_file = Path(path)
        if rules_file.exists():
            try:
                self.rules_graph.parse(rules_file, format="turtle")
                LOGGER.info(f"已加载业务规则: {path}, 三元组数: {len(self.rules_graph)}")
            except Exception as e:
                LOGGER.error(f"加载业务规则失败: {e}")
        else:
            LOGGER.warning(f"业务规则文件不存在: {path}")

    # ============================================
    # 规则解析辅助
    # ============================================

    def _load_user_level_rules(self) -> List[Dict[str, Any]]:
        """从本体规则图中读取用户等级规则并编译条件。"""

        query = """
        PREFIX rule: <http://example.org/rules#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?rule ?condition ?priority ?action ?label
        WHERE {
            ?rule a rule:UserLevelRule ;
                  rule:condition ?condition .
            OPTIONAL { ?rule rule:priority ?priority }
            OPTIONAL { ?rule rule:action ?action }
            OPTIONAL {
                ?rule rdfs:label ?label .
                FILTER (lang(?label) = "" || langMatches(lang(?label), "en") || langMatches(lang(?label), "zh"))
            }
        }
        """

        rules_map: Dict[str, Dict[str, Any]] = {}
        try:
            rows = self.rules_graph.query(query)
        except Exception as exc:
            LOGGER.warning("用户等级规则查询失败，回退到默认阈值: %s", exc)
            return []

        for row in rows:
            rule_uri = str(row.rule)
            entry = rules_map.setdefault(
                rule_uri,
                {
                    "rule": rule_uri,
                    "condition": None,
                    "priority": 0,
                    "label": None,
                    "action": None,
                    "target_level": None,
                },
            )

            if getattr(row, "condition", None):
                entry["condition"] = str(row.condition)
            if getattr(row, "priority", None) is not None:
                try:
                    entry["priority"] = int(row.priority)
                except Exception:
                    entry["priority"] = 0
            if getattr(row, "label", None) and not entry["label"]:
                entry["label"] = str(row.label)
            if getattr(row, "action", None) and not entry["action"]:
                entry["action"] = str(row.action)

        compiled_rules: List[Dict[str, Any]] = []
        for entry in rules_map.values():
            if not entry.get("condition"):
                LOGGER.warning("用户等级规则 %s 缺少 condition，已忽略", entry["rule"])
                continue

            target = entry.get("target_level") or self._derive_level_from_rule_text(
                entry.get("label"),
                entry.get("action"),
            )
            if not target:
                LOGGER.warning("用户等级规则 %s 未能识别目标等级，已忽略", entry["rule"])
                continue

            try:
                normalized, compiled_obj = self._compile_condition(entry["condition"])
            except ValueError as exc:
                LOGGER.warning("用户等级规则 %s 条件无法编译: %s", entry["rule"], exc)
                continue

            entry["target_level"] = target
            entry["normalized_condition"] = normalized
            entry["compiled_condition"] = compiled_obj
            compiled_rules.append(entry)

        compiled_rules.sort(key=lambda item: item.get("priority", 0), reverse=True)
        LOGGER.info("已加载 %d 条用户等级本体规则", len(compiled_rules))
        return compiled_rules

    def _load_discount_rules(self) -> List[Dict[str, Any]]:
        """从本体规则图中读取折扣规则并编译条件。"""

        query = """
        PREFIX rule: <http://example.org/rules#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?rule ?condition ?priority ?label ?action ?discountRate
        WHERE {
            ?rule a rule:DiscountRule ;
                  rule:condition ?condition ;
                  rule:discountRate ?discountRate .
            OPTIONAL { ?rule rule:priority ?priority }
            OPTIONAL { ?rule rdfs:label ?label }
            OPTIONAL { ?rule rule:action ?action }
        }
        """

        rule_entries: List[Dict[str, Any]] = []
        try:
            rows = self.rules_graph.query(query)
        except Exception as exc:
            LOGGER.warning("折扣规则查询失败，回退到静态策略: %s", exc)
            return []

        for row in rows:
            if not getattr(row, "condition", None):
                LOGGER.warning("折扣规则 %s 缺少 condition，已忽略", row.rule)
                continue
            if not getattr(row, "discountRate", None):
                LOGGER.warning("折扣规则 %s 缺少 discountRate，已忽略", row.rule)
                continue
            try:
                rate = Decimal(str(row.discountRate))
            except Exception as exc:
                LOGGER.warning("折扣规则 %s 的 discountRate 无法解析: %s", row.rule, exc)
                continue

            entry = {
                "rule": str(row.rule),
                "condition": str(row.condition),
                "priority": int(row.priority) if getattr(row, "priority", None) is not None else 0,
                "label": str(row.label) if getattr(row, "label", None) else None,
                "action": str(row.action) if getattr(row, "action", None) else None,
                "discount_rate": rate,
            }

            try:
                normalized, compiled_obj = self._compile_condition(entry["condition"])
            except ValueError as exc:
                LOGGER.warning("折扣规则 %s 条件无法编译: %s", entry["rule"], exc)
                continue

            entry["normalized_condition"] = normalized
            entry["compiled_condition"] = compiled_obj
            rule_entries.append(entry)

        rule_entries.sort(
            key=lambda item: (item.get("discount_rate", Decimal("1")), -item.get("priority", 0))
        )
        LOGGER.info("已加载 %d 条折扣本体规则", len(rule_entries))
        return rule_entries

    def _load_shipping_rules(self) -> List[Dict[str, Any]]:
        """从本体规则图中读取物流规则并编译条件。"""

        query = """
        PREFIX rule: <http://example.org/rules#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?rule ?condition ?priority ?label ?action ?shippingCost ?shippingType
        WHERE {
            ?rule a rule:ShippingRule ;
                  rule:condition ?condition .
            OPTIONAL { ?rule rule:priority ?priority }
            OPTIONAL { ?rule rdfs:label ?label }
            OPTIONAL { ?rule rule:action ?action }
            OPTIONAL { ?rule rule:shippingCost ?shippingCost }
            OPTIONAL { ?rule rule:shippingType ?shippingType }
        }
        """

        entries: List[Dict[str, Any]] = []
        try:
            rows = self.rules_graph.query(query)
        except Exception as exc:
            LOGGER.warning("物流规则查询失败，回退到静态策略: %s", exc)
            return []

        for row in rows:
            if not getattr(row, "condition", None):
                LOGGER.warning("物流规则 %s 缺少 condition，已忽略", row.rule)
                continue

            shipping_cost: Optional[Decimal] = None
            if getattr(row, "shippingCost", None) is not None:
                try:
                    shipping_cost = Decimal(str(row.shippingCost))
                except Exception as exc:
                    LOGGER.warning("物流规则 %s 的 shippingCost 无法解析: %s", row.rule, exc)
                    continue

            entry = {
                "rule": str(row.rule),
                "condition": str(row.condition),
                "priority": int(row.priority) if getattr(row, "priority", None) is not None else 0,
                "label": str(row.label) if getattr(row, "label", None) else None,
                "action": str(row.action) if getattr(row, "action", None) else None,
                "shipping_cost": shipping_cost,
                "shipping_type": str(row.shippingType) if getattr(row, "shippingType", None) else None,
            }

            try:
                normalized, compiled_obj = self._compile_condition(entry["condition"])
            except ValueError as exc:
                LOGGER.warning("物流规则 %s 条件无法编译: %s", entry["rule"], exc)
                continue

            entry["normalized_condition"] = normalized
            entry["compiled_condition"] = compiled_obj
            entry["is_surcharge"] = self._detect_shipping_surcharge(
                normalized,
                entry.get("label"),
                entry.get("action"),
            )
            entries.append(entry)

        entries.sort(key=lambda item: (item.get("is_surcharge", False), -item.get("priority", 0)))
        LOGGER.info("已加载 %d 条物流本体规则", len(entries))
        return entries

    @staticmethod
    def _detect_shipping_surcharge(condition: Optional[str], label: Optional[str], action: Optional[str]) -> bool:
        text = " ".join(filter(None, [condition or "", label or "", action or ""]))
        lowered = text.lower()
        keywords = ["remote", "加收", "surcharge"]
        if condition and "isRemoteArea" in condition:
            return True
        return any(keyword in lowered for keyword in keywords)

    def _load_return_rules(self) -> List[Dict[str, Any]]:
        """从本体规则图中读取退换货规则并编译条件。"""

        query = """
        PREFIX rule: <http://example.org/rules#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?rule ?condition ?priority ?label ?action ?returnPeriodDays ?applicableTo
        WHERE {
            ?rule a rule:ReturnRule ;
                  rule:condition ?condition .
            OPTIONAL { ?rule rule:priority ?priority }
            OPTIONAL { ?rule rdfs:label ?label }
            OPTIONAL { ?rule rule:action ?action }
            OPTIONAL { ?rule rule:returnPeriodDays ?returnPeriodDays }
            OPTIONAL { ?rule rule:applicableTo ?applicableTo }
        }
        """

        entries: List[Dict[str, Any]] = []
        try:
            rows = self.rules_graph.query(query)
        except Exception as exc:
            LOGGER.warning("退换货规则查询失败，回退到静态策略: %s", exc)
            return []

        for row in rows:
            if not getattr(row, "condition", None):
                LOGGER.warning("退换货规则 %s 缺少 condition，已忽略", row.rule)
                continue

            period: Optional[int] = None
            if getattr(row, "returnPeriodDays", None) is not None:
                try:
                    period = int(row.returnPeriodDays)
                except Exception as exc:
                    LOGGER.warning("退换货规则 %s 的 returnPeriodDays 无法解析: %s", row.rule, exc)
                    continue

            entry = {
                "rule": str(row.rule),
                "condition": str(row.condition),
                "priority": int(row.priority) if getattr(row, "priority", None) is not None else 0,
                "label": str(row.label) if getattr(row, "label", None) else None,
                "action": str(row.action) if getattr(row, "action", None) else None,
                "return_period_days": period,
                "applicable_to": str(row.applicableTo) if getattr(row, "applicableTo", None) else None,
            }

            try:
                normalized, compiled_obj = self._compile_condition(entry["condition"])
            except ValueError as exc:
                LOGGER.warning("退换货规则 %s 条件无法编译: %s", entry["rule"], exc)
                continue

            entry["normalized_condition"] = normalized
            entry["compiled_condition"] = compiled_obj
            entries.append(entry)

        entries.sort(key=lambda item: -item.get("priority", 0))
        LOGGER.info("已加载 %d 条退换货本体规则", len(entries))
        return entries

    def _load_cancellation_rules(self) -> List[Dict[str, Any]]:
        """从本体规则图中读取取消规则并编译条件。"""

        query = """
        PREFIX rule: <http://example.org/rules#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?rule ?condition ?priority ?label ?action ?cancelWindowHours ?allowedStatuses ?requiresShipmentCheck
        WHERE {
            ?rule a rule:CancellationRule ;
                  rule:condition ?condition .
            OPTIONAL { ?rule rule:priority ?priority }
            OPTIONAL { ?rule rdfs:label ?label }
            OPTIONAL { ?rule rule:action ?action }
            OPTIONAL { ?rule rule:cancelWindowHours ?cancelWindowHours }
            OPTIONAL { ?rule rule:allowedStatuses ?allowedStatuses }
            OPTIONAL { ?rule rule:requiresShipmentCheck ?requiresShipmentCheck }
        }
        """

        entries: List[Dict[str, Any]] = []
        try:
            rows = self.rules_graph.query(query)
        except Exception as exc:
            LOGGER.warning("取消规则查询失败，回退到静态策略: %s", exc)
            return []

        for row in rows:
            if not getattr(row, "condition", None):
                LOGGER.warning("取消规则 %s 缺少 condition，已忽略", row.rule)
                continue

            cancel_window: Optional[int] = None
            if getattr(row, "cancelWindowHours", None) is not None:
                try:
                    cancel_window = int(row.cancelWindowHours)
                except Exception as exc:
                    LOGGER.warning("取消规则 %s 的 cancelWindowHours 无法解析: %s", row.rule, exc)
                    continue

            entry = {
                "rule": str(row.rule),
                "condition": str(row.condition),
                "priority": int(row.priority) if getattr(row, "priority", None) is not None else 0,
                "label": str(row.label) if getattr(row, "label", None) else None,
                "action": str(row.action) if getattr(row, "action", None) else None,
                "cancel_window_hours": cancel_window,
                "allowed_statuses": self._parse_allowed_statuses(getattr(row, "allowedStatuses", None)),
                "requires_shipment_check": self._parse_bool_literal(getattr(row, "requiresShipmentCheck", None)),
            }

            try:
                normalized, compiled_obj = self._compile_condition(entry["condition"])
            except ValueError as exc:
                LOGGER.warning("取消规则 %s 条件无法编译: %s", entry["rule"], exc)
                continue

            entry["normalized_condition"] = normalized
            entry["compiled_condition"] = compiled_obj
            entries.append(entry)

        entries.sort(key=lambda item: -item.get("priority", 0))
        LOGGER.info("已加载 %d 条取消本体规则", len(entries))
        return entries

    def _build_discount_context(self, user_level: str, order_amount: Decimal, is_first_order: bool) -> Dict[str, Any]:
        order_amount_float = float(order_amount)
        return {
            "orderAmount": order_amount_float,
            "order_amount": order_amount_float,
            "userLevel": user_level,
            "user_level": user_level,
            "isFirstOrder": bool(is_first_order),
            "orderCount": 1 if is_first_order else 2,
        }

    def _build_shipping_context(self, user_level: str, order_amount: Decimal, is_remote_area: bool) -> Dict[str, Any]:
        amount_float = float(order_amount)
        return {
            "userLevel": user_level,
            "user_level": user_level,
            "orderAmount": amount_float,
            "order_amount": amount_float,
            "isRemoteArea": bool(is_remote_area),
            "isRemote": bool(is_remote_area),
        }

    def _build_return_context(
        self,
        user_level: str,
        product_category: str,
        within_days: int,
        is_activated: bool,
        packaging_intact: bool,
    ) -> Dict[str, Any]:
        normalized_category = self._normalize_product_category(product_category)
        return {
            "userLevel": user_level,
            "user_level": user_level,
            "category": normalized_category,
            "productCategory": normalized_category,
            "withinDays": within_days,
            "daysSincePurchase": within_days,
            "isActivated": bool(is_activated),
            "packaging": "intact" if packaging_intact else "opened",
            "packagingIntact": bool(packaging_intact),
        }

    @staticmethod
    def _normalize_product_category(product_category: str) -> str:
        normalized = (product_category or "").strip().lower()
        mapping = {
            "手机": "ElectronicProduct",
            "电子": "ElectronicProduct",
            "electronics": "ElectronicProduct",
            "electronic": "ElectronicProduct",
            "配件": "Accessory",
            "accessory": "Accessory",
            "附件": "Accessory",
            "服务": "Service",
            "service": "Service",
        }
        for key, target in mapping.items():
            if key in normalized:
                return target
        return product_category or "General"

    @staticmethod
    def _parse_bool_literal(value: Any) -> Optional[bool]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        return None

    @staticmethod
    def _parse_allowed_statuses(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        text = str(value)
        parts = [item.strip().lower() for item in text.split(',') if item.strip()]
        return parts or None

    def _build_cancellation_context(
        self,
        order_status: str,
        hours_since_created: float,
        has_shipment: bool,
    ) -> Dict[str, Any]:
        status = (order_status or "pending").strip().lower()
        hours = max(float(hours_since_created or 0.0), 0.0)
        return {
            "orderStatus": status,
            "order_status": status,
            "hoursSinceOrder": hours,
            "hours_since_order": hours,
            "hasShipment": bool(has_shipment),
        }

    @staticmethod
    def _interpret_cancellation_allowed(action: Optional[str], label: Optional[str], cancel_window: Optional[int]) -> bool:
        text = " ".join(filter(None, [action, label]))
        lowered = text.lower()
        block_keywords = ["拒绝", "不可", "不允许", "block", "禁止"]
        allow_keywords = ["允许", "可取消", "cancel", "取消", "申请"]
        if any(keyword in lowered for keyword in block_keywords):
            return False
        if any(keyword in lowered for keyword in allow_keywords):
            return True
        if cancel_window is not None:
            return cancel_window > 0
        return True

    @staticmethod
    def _short_rule_name(rule_uri: Optional[str]) -> Optional[str]:
        if not rule_uri:
            return rule_uri
        if "#" in rule_uri:
            return rule_uri.split("#")[-1]
        trimmed = rule_uri.rstrip("/")
        if "/" in trimmed:
            return trimmed.split("/")[-1]
        return rule_uri

    @staticmethod
    def _derive_level_from_rule_text(label: Optional[str], action: Optional[str]) -> Optional[str]:
        """尝试从标签/动作描述中解析目标等级。"""

        text = " ".join(filter(None, [label or "", action or ""])).upper()
        if not text:
            return None
        if "SVIP" in text:
            return "SVIP"
        if "VIP" in text:
            return "VIP"
        if "REGULAR" in text or "普通" in text:
            return "Regular"
        return None

    @staticmethod
    def _normalize_condition_expr(condition: str) -> str:
        expr = condition.strip()
        expr = re.sub(r"(?i)\bAND\b", " and ", expr)
        expr = re.sub(r"(?i)\bOR\b", " or ", expr)
        expr = re.sub(r"(?i)\bNOT\s+IN\b", " not in ", expr)
        expr = re.sub(r"(?i)\bIN\b", " in ", expr)
        expr = re.sub(r"(?i)\bTRUE\b", "True", expr)
        expr = re.sub(r"(?i)\bFALSE\b", "False", expr)
        expr = re.sub(r"(?<![<>!=])=(?!=)", "==", expr)
        return expr

    def _compile_condition(self, condition: str) -> Tuple[str, Any]:
        expr = self._normalize_condition_expr(condition)
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"语法错误: {exc}") from exc
        self._validate_condition_ast(tree)
        compiled_obj = compile(tree, "<rule_condition>", "eval")
        return expr, compiled_obj

    @staticmethod
    def _validate_condition_ast(tree: ast.AST) -> None:
        disallowed = (
            ast.Call,
            ast.Attribute,
            ast.Subscript,
            ast.Lambda,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Await,
            ast.Yield,
            ast.FunctionDef,
            ast.ClassDef,
            ast.Assign,
            ast.AugAssign,
            ast.Import,
            ast.ImportFrom,
        )
        for node in ast.walk(tree):
            if isinstance(node, disallowed):
                raise ValueError("条件中包含不受支持的表达式")

    @staticmethod
    def _evaluate_compiled_condition(compiled_obj: Any, context: Dict[str, Any]) -> bool:
        try:
            return bool(eval(compiled_obj, {"__builtins__": {}}, context))
        except Exception:
            raise

    @staticmethod
    def _fallback_user_level(total_spent_float: float) -> str:
        if total_spent_float >= 10000:
            return "SVIP"
        if total_spent_float >= 5000:
            return "VIP"
        return "Regular"
    
    # ============================================
    # 用户等级推理
    # ============================================
    
    def infer_user_level(self, total_spent: Decimal) -> str:
        """根据累计消费推理用户等级
        
        Args:
            total_spent: 累计消费金额
            
        Returns:
            str: 用户等级 (Regular/VIP/SVIP)
        """
        total_spent_float = float(total_spent)

        if self._user_level_rules:
            context = {"totalSpent": total_spent_float}
            for rule in self._user_level_rules:
                try:
                    if self._evaluate_compiled_condition(rule["compiled_condition"], context):
                        LOGGER.info(
                            "用户等级推理[推理方式=本体规则]: 命中 %s -> %s (totalSpent=%s, condition=%s)",
                            rule.get("label") or rule["rule"],
                            rule["target_level"],
                            total_spent,
                            rule.get("condition"),
                        )
                        return rule["target_level"]
                except Exception as exc:
                    LOGGER.warning("用户等级规则 %s 评估失败: %s", rule["rule"], exc)

            LOGGER.info(
                "用户等级推理[推理方式=本体规则]: 未匹配任何规则，默认返回 Regular (totalSpent=%s)",
                total_spent,
            )
            return "Regular"

        LOGGER.warning("用户等级推理: 未加载到本体规则，使用阈值回退")
        fallback_level = self._fallback_user_level(total_spent_float)
        LOGGER.info(
            "用户等级推理[推理方式=阈值回退]: totalSpent=%s -> %s",
            total_spent,
            fallback_level,
        )
        return fallback_level
    
    # ============================================
    # 折扣推理
    # ============================================
    
    def infer_discount(self, user_level: str, order_amount: Decimal,
                      is_first_order: bool = False) -> Dict[str, Any]:
        """推理订单折扣
        
        Args:
            user_level: 用户等级 (Regular/VIP/SVIP)
            order_amount: 订单金额
            is_first_order: 是否首单
            
        Returns:
            Dict: {
                'discount_type': str,  # 折扣类型
                'discount_rate': Decimal,  # 折扣率
                'discount_amount': Decimal,  # 折扣金额
                'final_amount': Decimal,  # 最终金额
                'reason': str  # 推理理由
            }
        """
        if self._discount_rules:
            context = self._build_discount_context(user_level, order_amount, is_first_order)
            matched_rules: List[Dict[str, Any]] = []
            for rule in self._discount_rules:
                try:
                    if self._evaluate_compiled_condition(rule["compiled_condition"], context):
                        matched_rules.append(rule)
                except Exception as exc:
                    LOGGER.warning("折扣规则 %s 评估失败: %s", rule["rule"], exc)

            if matched_rules:
                best_rule = min(
                    matched_rules,
                    key=lambda item: (item["discount_rate"], -item.get("priority", 0)),
                )
                discount_rate = best_rule["discount_rate"]
                final_amount = order_amount * discount_rate
                discount_amount = order_amount - final_amount
                reason = (
                    f"命中{best_rule.get('label') or best_rule['rule']}，折扣率"
                    f"{float(discount_rate):.2f}"
                )
                if len(matched_rules) > 1:
                    extra = [r.get("label") or r["rule"] for r in matched_rules if r is not best_rule]
                    reason += f"（同时满足{', '.join(extra)}，已根据折扣力度选择最优）"
                LOGGER.info(
                    "折扣推理[推理方式=本体规则]: %s (user_level=%s, amount=%s)",
                    reason,
                    user_level,
                    order_amount,
                )
                return {
                    'discount_type': best_rule.get('label') or best_rule['rule'],
                    'discount_rate': discount_rate,
                    'discount_amount': discount_amount,
                    'final_amount': final_amount,
                    'reason': reason,
                    'rule_applied': best_rule['rule'],
                }

            LOGGER.info(
                "折扣推理[推理方式=本体规则]: 未匹配到规则，回退静态策略 (user_level=%s, amount=%s)",
                user_level,
                order_amount,
            )

        return self._fallback_discount_inference(user_level, order_amount, is_first_order)

    def _fallback_discount_inference(
        self,
        user_level: str,
        order_amount: Decimal,
        is_first_order: bool,
    ) -> Dict[str, Any]:
        order_amount_float = float(order_amount)
        applicable_discounts = []

        if user_level == "SVIP":
            applicable_discounts.append({
                'type': 'SVIP会员折扣',
                'rate': Decimal('0.90'),
                'priority': 40,
                'rule': 'SVIPDiscountRule'
            })
        elif user_level == "VIP":
            applicable_discounts.append({
                'type': 'VIP会员折扣',
                'rate': Decimal('0.95'),
                'priority': 30,
                'rule': 'VIPDiscountRule'
            })

        if order_amount_float >= 10000:
            applicable_discounts.append({
                'type': '批量折扣(满10000)',
                'rate': Decimal('0.90'),
                'priority': 25,
                'rule': 'VolumeDiscount10kRule'
            })
        elif order_amount_float >= 5000:
            applicable_discounts.append({
                'type': '批量折扣(满5000)',
                'rate': Decimal('0.95'),
                'priority': 15,
                'rule': 'VolumeDiscount5kRule'
            })

        if is_first_order:
            applicable_discounts.append({
                'type': '首单折扣',
                'rate': Decimal('0.98'),
                'priority': 5,
                'rule': 'FirstOrderDiscountRule'
            })

        if not applicable_discounts:
            LOGGER.info(
                "折扣推理[推理方式=规则优选]: 不满足任何折扣条件 (user_level=%s, amount=%s)",
                user_level,
                order_amount,
            )
            return {
                'discount_type': '无折扣',
                'discount_rate': Decimal('1.0'),
                'discount_amount': Decimal('0'),
                'final_amount': order_amount,
                'reason': '不满足任何折扣条件'
            }

        best_discount = min(applicable_discounts, key=lambda x: x['rate'])

        discount_rate = best_discount['rate']
        final_amount = order_amount * discount_rate
        discount_amount = order_amount - final_amount

        reason = f"应用{best_discount['type']}，折扣率{float(discount_rate):.2f}"
        if len(applicable_discounts) > 1:
            other_discounts = [d['type'] for d in applicable_discounts if d != best_discount]
            reason += f"（同时满足{', '.join(other_discounts)}，已自动选择最优）"

        LOGGER.info(f"折扣推理[推理方式=规则优选]: {reason}")

        return {
            'discount_type': best_discount['type'],
            'discount_rate': discount_rate,
            'discount_amount': discount_amount,
            'final_amount': final_amount,
            'reason': reason,
            'rule_applied': best_discount['rule']
        }
    
    # ============================================
    # 物流推理
    # ============================================
    
    def infer_shipping(self, user_level: str, order_amount: Decimal,
                      is_remote_area: bool = False) -> Dict[str, Any]:
        """推理物流策略
        
        Args:
            user_level: 用户等级
            order_amount: 订单金额
            is_remote_area: 是否偏远地区
            
        Returns:
            Dict: {
                'shipping_cost': Decimal,  # 运费
                'shipping_type': str,  # 配送类型
                'free_shipping': bool,  # 是否包邮
                'reason': str  # 推理理由
            }
        """
        if self._shipping_rules:
            context = self._build_shipping_context(user_level, order_amount, is_remote_area)
            matched_rules: List[Dict[str, Any]] = []
            for rule in self._shipping_rules:
                try:
                    if self._evaluate_compiled_condition(rule["compiled_condition"], context):
                        matched_rules.append(rule)
                except Exception as exc:
                    LOGGER.warning("物流规则 %s 评估失败: %s", rule["rule"], exc)

            if matched_rules:
                matched_sorted = sorted(
                    matched_rules,
                    key=lambda item: (item.get("is_surcharge", False), -item.get("priority", 0)),
                )
                shipping_cost = Decimal('0')
                shipping_type = 'Standard'
                reasons: List[str] = []
                for rule in matched_sorted:
                    label = rule.get('label') or rule['rule']
                    reasons.append(f"命中{label}")
                    if rule.get('shipping_type'):
                        shipping_type = rule['shipping_type']
                    cost_value = rule.get('shipping_cost')
                    if cost_value is not None:
                        if rule.get('is_surcharge'):
                            shipping_cost += cost_value
                        else:
                            shipping_cost = cost_value

                free_shipping = shipping_cost == Decimal('0')
                reason = "；".join(reasons)
                LOGGER.info(
                    "物流推理[推理方式=本体规则]: %s (user_level=%s, amount=%s, remote=%s)",
                    reason,
                    user_level,
                    order_amount,
                    is_remote_area,
                )
                return {
                    'shipping_cost': shipping_cost,
                    'shipping_type': shipping_type,
                    'free_shipping': free_shipping,
                    'reason': reason,
                    'estimated_days': 1 if shipping_type == "NextDayDelivery" else 3,
                    'rule_applied': [rule['rule'] for rule in matched_sorted],
                }

            LOGGER.info(
                "物流推理[推理方式=本体规则]: 未匹配到规则，使用静态策略 (user_level=%s, amount=%s)",
                user_level,
                order_amount,
            )

        return self._fallback_shipping_inference(user_level, order_amount, is_remote_area)

    def _fallback_shipping_inference(
        self,
        user_level: str,
        order_amount: Decimal,
        is_remote_area: bool,
    ) -> Dict[str, Any]:
        order_amount_float = float(order_amount)
        shipping_cost = Decimal('0')
        shipping_type = 'Standard'
        free_shipping = False
        reasons = []

        if user_level == "SVIP":
            shipping_type = "NextDayDelivery"
            free_shipping = True
            reasons.append("SVIP用户享受免费次日达服务")
        elif user_level == "VIP":
            free_shipping = True
            reasons.append("VIP用户享受包邮服务")
        elif order_amount_float >= 500:
            free_shipping = True
            reasons.append("订单金额满500元包邮")
        else:
            shipping_cost = Decimal('15')
            reasons.append("普通用户订单不满500元，收取15元运费")

        if is_remote_area and user_level != "SVIP":
            shipping_cost += Decimal('30')
            free_shipping = False
            reasons.append("偏远地区加收30元运费")

        reason = "; ".join(reasons) if reasons else "默认物流策略"
        LOGGER.info("物流推理[推理方式=静态规则]: %s", reason)

        return {
            'shipping_cost': shipping_cost,
            'shipping_type': shipping_type,
            'free_shipping': free_shipping,
            'reason': reason,
            'estimated_days': 1 if shipping_type == "NextDayDelivery" else 3,
            'rule_applied': ['ShippingRuleFallback'],
        }
    
    # ============================================
    # 退换货规则推理
    # ============================================
    
    def infer_return_policy(
        self,
        user_level: str,
        product_category: str,
        is_activated: bool = False,
        packaging_intact: bool = True,
        days_since_purchase: Optional[int] = None,
    ) -> Dict[str, Any]:
        """推理退换货政策
        
        Args:
            user_level: 用户等级
            product_category: 商品分类 (手机/配件/服务)
            is_activated: 电子产品是否已激活
            packaging_intact: 商品包装是否完好
            days_since_purchase: 下单后经过的天数，用于对比规则期限
            
        Returns:
            Dict: {
                'returnable': bool,  # 是否可退货
                'return_period_days': int,  # 退货期限(天)
                'conditions': List[str],  # 退货条件
                'reason': str  # 推理理由
            }
        """
        within_days = days_since_purchase if days_since_purchase is not None else 7

        if self._return_rules:
            context = self._build_return_context(
                user_level,
                product_category,
                within_days,
                is_activated,
                packaging_intact,
            )
            matched_rules: List[Dict[str, Any]] = []
            for rule in self._return_rules:
                try:
                    if self._evaluate_compiled_condition(rule["compiled_condition"], context):
                        matched_rules.append(rule)
                except Exception as exc:
                    LOGGER.warning("退换货规则 %s 评估失败: %s", rule["rule"], exc)

            if matched_rules:
                matched_sorted = sorted(matched_rules, key=lambda item: -item.get("priority", 0))
                returnable = True
                return_period_days: Optional[int] = None
                conditions: List[str] = []
                reasons: List[str] = []
                for rule in matched_sorted:
                    label = rule.get('label') or rule['rule']
                    reasons.append(f"命中{label}")
                    if rule.get('action'):
                        conditions.append(rule['action'])
                    period = rule.get('return_period_days')
                    if period is not None:
                        return_period_days = period
                        if period <= 0:
                            returnable = False
                    if any(keyword in (rule.get('action') or '') for keyword in ["不可退", "不可退款", "不可退货", "拒绝"]):
                        returnable = False

                result = {
                    'returnable': returnable,
                    'return_period_days': return_period_days if return_period_days is not None else (15 if user_level in {"VIP", "SVIP"} else 7),
                    'conditions': conditions,
                    'reason': "；".join(reasons),
                    'rule_applied': [rule['rule'] for rule in matched_sorted],
                }
                LOGGER.info(
                    "退换货推理[推理方式=本体规则]: %s (user_level=%s, category=%s)",
                    result['reason'],
                    user_level,
                    product_category,
                )
                return result

            LOGGER.info(
                "退换货推理[推理方式=本体规则]: 未匹配规则，回退静态策略 (user_level=%s, category=%s)",
                user_level,
                product_category,
            )

        return self._fallback_return_policy(
            user_level,
            product_category,
            is_activated,
            packaging_intact,
        )

    def _fallback_return_policy(
        self,
        user_level: str,
        product_category: str,
        is_activated: bool,
        packaging_intact: bool,
    ) -> Dict[str, Any]:
        returnable = True
        return_period_days = 7
        conditions: List[str] = []
        reasons: List[str] = []

        if product_category == "服务":
            returnable = False
            return_period_days = 0
            reasons.append("服务类商品(如AppleCare+)不可退货")
        elif user_level in ["VIP", "SVIP"]:
            return_period_days = 15
            reasons.append("VIP/SVIP用户享受15天无理由退货")
        else:
            return_period_days = 7
            reasons.append("普通用户享受7天无理由退货")

        if product_category == "手机":
            if is_activated:
                conditions.append("电子产品已激活，需符合质量问题才能退货")
                reasons.append("电子产品已激活，仅限质量问题退货")
            else:
                conditions.append("电子产品未激活，可无理由退货")
        elif product_category == "配件":
            if packaging_intact:
                conditions.append("包装需保持完好")
                reasons.append("配件类商品包装完好可退货")
            else:
                returnable = False
                return_period_days = 0
                reasons.append("配件类商品包装已拆封，不可退货")

        reason = "; ".join(reasons) if reasons else "默认退换货策略"
        LOGGER.info("退换货推理[推理方式=静态规则]: %s", reason)

        return {
            'returnable': returnable,
            'return_period_days': return_period_days,
            'conditions': conditions,
            'reason': reason,
            'rule_applied': ['ReturnRuleFallback'],
        }

    # ============================================
    # 取消规则推理
    # ============================================

    def infer_cancellation_policy(
        self,
        order_status: str,
        hours_since_created: float,
        has_shipment: bool = False,
    ) -> Dict[str, Any]:
        """推理订单是否允许取消及其理由."""
        status = (order_status or "pending").lower()
        hours = max(float(hours_since_created or 0.0), 0.0)

        if self._cancellation_rules:
            context = self._build_cancellation_context(status, hours, has_shipment)
            matched_rules: List[Dict[str, Any]] = []
            for rule in self._cancellation_rules:
                try:
                    if self._evaluate_compiled_condition(rule["compiled_condition"], context):
                        matched_rules.append(rule)
                except Exception as exc:
                    LOGGER.warning("取消规则 %s 评估失败: %s", rule["rule"], exc)

            if matched_rules:
                best_rule = max(matched_rules, key=lambda item: item.get("priority", 0))
                allowed = self._interpret_cancellation_allowed(
                    best_rule.get('action'),
                    best_rule.get('label'),
                    best_rule.get('cancel_window_hours'),
                )
                if best_rule.get('requires_shipment_check') and has_shipment:
                    allowed = False
                deadline = best_rule.get('cancel_window_hours')
                policy = {
                    "status": status,
                    "hours_since_order": hours,
                    "allowed": allowed,
                    "rule": self._short_rule_name(best_rule['rule']),
                    "deadline_hours": deadline,
                    "reason": best_rule.get('action') or (best_rule.get('label') or '取消规则触发'),
                    "allowed_statuses": best_rule.get('allowed_statuses'),
                    "rule_applied": [best_rule['rule']],
                }
                LOGGER.info(
                    "取消规则推理[推理方式=本体规则]: %s (status=%s, hours=%.2f)",
                    policy['reason'],
                    status,
                    hours,
                )
                return policy

            LOGGER.info(
                "取消规则推理[推理方式=本体规则]: 未匹配规则，使用静态策略 (status=%s)",
                status,
            )

        return self._fallback_cancellation_policy(status, hours, has_shipment)

    def _fallback_cancellation_policy(
        self,
        status: str,
        hours: float,
        has_shipment: bool,
    ) -> Dict[str, Any]:
        status_normalized = status.lower()
        policy = {
            "status": status_normalized,
            "hours_since_order": hours,
            "allowed": False,
            "rule": "DefaultCancellationRule",
            "deadline_hours": None,
            "reason": "订单状态不支持取消",
            "rule_applied": ['CancellationRuleFallback'],
        }

        method_label = "[推理方式=静态规则]"

        if status_normalized in {"shipped", "delivered"} or has_shipment:
            policy.update(
                {
                    "allowed": False,
                    "rule": "ShippedCancellationBlockRule",
                    "deadline_hours": 0,
                    "reason": "订单已发货，需走退货流程",
                }
            )
            LOGGER.info("取消规则推理%s %s", method_label, policy["reason"])
            return policy

        if status_normalized == "pending":
            deadline = 24.0
            policy["deadline_hours"] = deadline
            if hours <= deadline:
                policy.update(
                    {
                        "allowed": True,
                        "rule": "Pending24hCancellationRule",
                        "reason": "待支付订单24小时内可直接取消",
                    }
                )
            else:
                policy["reason"] = "已超过24小时待支付取消窗口"
            LOGGER.info("取消规则推理%s %s", method_label, policy["reason"])
            return policy

        if status_normalized == "paid":
            deadline = 12.0
            policy["deadline_hours"] = deadline
            if has_shipment:
                policy.update(
                    {
                        "allowed": False,
                        "rule": "ShippedCancellationBlockRule",
                        "reason": "已生成物流单，无法取消",
                    }
                )
            elif hours <= deadline:
                policy.update(
                    {
                        "allowed": True,
                        "rule": "Paid12hCancellationRule",
                        "reason": "已支付12小时内且未发货，可人工审核后取消",
                    }
                )
            else:
                policy["reason"] = "已超过12小时支付取消窗口"
            LOGGER.info("取消规则推理%s %s", method_label, policy["reason"])
            return policy

        if status_normalized in {"cancelled", "returned"}:
            policy.update(
                {
                    "allowed": False,
                    "rule": "AlreadyTerminatedCancellationRule",
                    "reason": "订单已终止，无需再次取消",
                }
            )
            LOGGER.info("取消规则推理%s %s", method_label, policy["reason"])
            return policy

        policy["reason"] = "当前状态不支持自动取消，请联系客服"
        LOGGER.info("取消规则推理%s %s", method_label, policy["reason"])
        return policy
    
    # ============================================
    # 综合推理
    # ============================================
    
    def infer_order_details(self, user_data: Dict[str, Any],
                           order_data: Dict[str, Any]) -> Dict[str, Any]:
        """综合推理订单详情（折扣+物流）
        
        Args:
            user_data: 用户数据 {
                'user_id': int,
                'user_level': str,
                'total_spent': Decimal,
                'order_count': int
            }
            order_data: 订单数据 {
                'order_amount': Decimal,
                'products': List[Dict],
                'shipping_address': str
            }
            
        Returns:
            Dict: 完整的订单推理结果
        """
        # 1. 推理用户等级（如果需要升级）
        current_level = user_data.get('user_level', 'Regular')
        total_spent = Decimal(str(user_data.get('total_spent', 0)))
        inferred_level = self.infer_user_level(total_spent)
        
        if inferred_level != current_level:
            LOGGER.info(
                f"用户等级推理[推理方式=规则阈值]: 检测到等级变化 {current_level} -> {inferred_level}"
            )
        
        # 2. 推理折扣
        order_amount = Decimal(str(order_data['order_amount']))
        is_first_order = user_data.get('order_count', 0) == 0
        
        discount_info = self.infer_discount(
            user_level=inferred_level,
            order_amount=order_amount,
            is_first_order=is_first_order
        )
        
        # 3. 推理物流
        # 简单判断是否偏远地区（实际应该查询地址库）
        shipping_address = order_data.get('shipping_address', '')
        is_remote = any(area in shipping_address for area in ['西藏', '新疆', '内蒙古'])
        
        shipping_info = self.infer_shipping(
            user_level=inferred_level,
            order_amount=discount_info['final_amount'],  # 用折后金额
            is_remote_area=is_remote
        )

        LOGGER.info(
            "订单综合推理[推理方式=组合推理]: user_id=%s discount_rule=%s shipping_type=%s",
            user_data.get('user_id'),
            discount_info.get('rule_applied', 'N/A'),
            shipping_info.get('shipping_type'),
        )
        
        # 4. 汇总结果
        return {
            'user_level_inference': {
                'original_level': current_level,
                'inferred_level': inferred_level,
                'should_upgrade': inferred_level != current_level
            },
            'discount_inference': discount_info,
            'shipping_inference': shipping_info,
            'final_summary': {
                'original_amount': order_amount,
                'discount_amount': discount_info['discount_amount'],
                'subtotal': discount_info['final_amount'],
                'shipping_cost': shipping_info['shipping_cost'],
                'total_payable': discount_info['final_amount'] + shipping_info['shipping_cost']
            }
        }
    
    # ============================================
    # 查询本体
    # ============================================
    
    def query_rules_by_type(self, rule_type: str) -> List[Dict[str, Any]]:
        """查询特定类型的业务规则
        
        Args:
            rule_type: 规则类型 (UserLevelRule/DiscountRule/ShippingRule/ReturnRule)
            
        Returns:
            List[Dict]: 规则列表
        """
        query = f"""
        PREFIX rule: <http://example.org/rules#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?rule ?label ?condition ?action ?priority
        WHERE {{
            ?rule a rule:{rule_type} ;
                  rdfs:label ?label ;
                  rule:condition ?condition ;
                  rule:action ?action .
            OPTIONAL {{ ?rule rule:priority ?priority }}
        }}
        ORDER BY DESC(?priority)
        """
        
        results = []
        for row in self.rules_graph.query(query):
            results.append({
                'rule': str(row.rule),
                'label': str(row.label),
                'condition': str(row.condition),
                'action': str(row.action),
                'priority': int(row.priority) if row.priority else 0
            })
        
        return results
    
    def get_all_discount_rules(self) -> List[Dict[str, Any]]:
        """获取所有折扣规则"""
        return self.query_rules_by_type("DiscountRule")
    
    def get_all_shipping_rules(self) -> List[Dict[str, Any]]:
        """获取所有物流规则"""
        return self.query_rules_by_type("ShippingRule")
    
    def get_all_return_rules(self) -> List[Dict[str, Any]]:
        """获取所有退换货规则"""
        return self.query_rules_by_type("ReturnRule")
