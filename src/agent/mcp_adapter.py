from __future__ import annotations
# Copyright (c) 2025 shark8848
# MIT License
#
# Ontology MCP Server - 电商 AI 助手系统
# 本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI
#
# Author: shark8848
# Repository: https://github.com/shark8848/ontology-mcp-server
"""MCP HTTP 适配器 + 工具封装。"""

import ast
import json
import operator
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Type, Union, get_args, get_origin

import requests
from pydantic import BaseModel, Field

from .logger import get_logger

logger = get_logger(__name__)


def _sanitize_schema(obj: Any) -> None:
    """递归移除schema中所有additionalProperties字段，避免Gradio解析错误。"""
    if isinstance(obj, dict):
        if "additionalProperties" in obj:
            del obj["additionalProperties"]
        for value in obj.values():
            _sanitize_schema(value)
    elif isinstance(obj, list):
        for item in obj:
            _sanitize_schema(item)


_MATH_EXPR_PATTERN = re.compile(r"^[0-9\.+\-*/()\s]+$")


def _evaluate_math_expression(expr: str) -> float:
    """Safely evaluate simple arithmetic expressions used inside arguments."""

    allowed_bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    allowed_unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("不支持的常量类型")
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_bin_ops:
            left = _eval(node.left)
            right = _eval(node.right)
            return allowed_bin_ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary_ops:
            operand = _eval(node.operand)
            return allowed_unary_ops[type(node.op)](operand)
        raise ValueError("仅支持简单算术表达式")

    tree = ast.parse(expr, mode="eval")
    return float(_eval(tree))


def _parse_argument_string(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ValueError("参数既不是合法 JSON，也无法载入 YAML") from exc
        try:
            data = yaml.safe_load(text)
        except Exception as exc:  # pragma: no cover - YAML parse errors
            raise ValueError("参数 YAML 解析失败") from exc
    if not isinstance(data, dict):
        raise ValueError("工具参数必须为对象类型")
    return data


def _resolve_annotation(annotation: Any) -> Any:
    origin_type = get_origin(annotation)
    if origin_type is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _resolve_annotation(args[0]) if args else Any
    return annotation


def _maybe_eval_numeric_literal(value: Any, expected: Any) -> Any:
    if not isinstance(value, str):
        return value
    expr = value.strip()
    if not expr or not _MATH_EXPR_PATTERN.match(expr):
        return value
    try:
        evaluated = _evaluate_math_expression(expr)
    except Exception:
        return value
    if expected is int and float(evaluated).is_integer():
        return int(evaluated)
    return float(evaluated)


@dataclass
class ToolDefinition:
    """描述一个 MCP 工具及其调用方式。"""

    name: str
    description: str
    args_schema: Type[BaseModel]
    func: Callable[..., Any]

    def to_openai_tool(self) -> Dict[str, Any]:
        # 手动构建简化的 schema，避免 Pydantic 生成的复杂嵌套结构
        properties: Dict[str, Any] = {}
        required: List[str] = []
        
        for field_name, field_info in self.args_schema.model_fields.items():
            annotation = field_info.annotation

            def resolve(annotation_value: Any) -> Any:
                origin_type = get_origin(annotation_value)
                if origin_type is None:
                    return annotation_value
                if origin_type in (list, List):
                    return list
                if origin_type in (dict, Dict):
                    return dict
                if origin_type in (tuple, Tuple):
                    return tuple
                if origin_type is Union:
                    args = [arg for arg in get_args(annotation_value) if arg is not type(None)]
                    return resolve(args[0]) if args else str
                return annotation_value

            resolved = resolve(annotation)

            field_type = "string"
            schema_extra: Dict[str, Any] = {}

            if resolved is bool:
                field_type = "boolean"
            elif resolved is int:
                field_type = "integer"
            elif resolved is float:
                field_type = "number"
            elif resolved is list:
                field_type = "array"
                schema_extra["items"] = {"type": "object"}
            elif resolved is dict:
                field_type = "object"

            properties[field_name] = {
                "type": field_type,
                "description": field_info.description or "",
                **schema_extra,
            }
            
            if field_info.is_required():
                required.append(field_name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def parse_arguments(self, raw_args: str | Dict[str, Any]) -> Dict[str, Any]:
        data: Any = raw_args

        if isinstance(data, dict) and "_raw" in data and isinstance(data["_raw"], (str, dict)):
            # LangChain/Gradio 有时会包装一层 `_raw`，需要拆开
            data = data["_raw"]

        if isinstance(data, str):
            try:
                data = _parse_argument_string(data)
            except ValueError as exc:
                raise ValueError(f"工具 {self.name} 参数解析失败: {exc}") from exc

        if data is None:
            data = {}

        if not isinstance(data, dict):
            raise ValueError(f"工具 {self.name} 参数必须是对象类型")

        normalized = dict(data)
        for field_name, field_info in self.args_schema.model_fields.items():
            if field_name not in normalized:
                continue
            target_type = _resolve_annotation(field_info.annotation)
            if target_type in (int, float):
                normalized[field_name] = _maybe_eval_numeric_literal(normalized[field_name], target_type)

        model = self.args_schema.model_validate(normalized or {})
        return model.model_dump()

    def invoke(self, parsed_args: Dict[str, Any]) -> Any:
        return self.func(**parsed_args)


def _load_yaml_config() -> dict:
    try:
        import yaml
    except Exception:
        return {}
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    if not cfg_path.exists():
        alt = Path(__file__).resolve().parents[1] / "agent" / "config.yaml"
        if alt.exists():
            cfg_path = alt
        else:
            return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class MCPAdapter:
    """封装 MCP Server 的 HTTP 访问，并提供 LangChain Tool。"""

    def __init__(self, base_url: str | None = None, timeout: int = 10) -> None:
        cfg = _load_yaml_config()
        self.base_url = base_url or os.getenv("MCP_BASE_URL") or cfg.get("MCP_BASE_URL") or "http://localhost:8000"
        self.timeout = timeout

    # ------------------------------------------------------------------
    # HTTP 基础操作
    # ------------------------------------------------------------------
    def invoke(self, tool: str, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        url = f"{self.base_url.rstrip('/')}/invoke"
        body = {"tool": tool, "payload": payload}
        logger.debug("POST %s tool=%s", url, tool)
        resp = requests.post(url, json=body, timeout=self.timeout)
        if resp.status_code != 200:
            logger.warning("MCP invoke failed %s %s", resp.status_code, resp.text[:200])
            return False, {"status_code": resp.status_code, "text": resp.text}
        return True, resp.json()

    def capabilities(self) -> Tuple[bool, Any]:
        url = f"{self.base_url.rstrip('/')}/capabilities"
        logger.debug("GET %s", url)
        resp = requests.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            logger.warning("MCP capabilities failed %s %s", resp.status_code, resp.text[:200])
            return False, {"status_code": resp.status_code, "text": resp.text}
        return True, resp.json()

    # ------------------------------------------------------------------
    # LangChain Tool 工厂
    # ------------------------------------------------------------------
    def _invoke_or_raise(self, tool: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        ok, data = self.invoke(tool, payload)
        if not ok:
            raise RuntimeError(f"调用 {tool} 失败: {data}")
        return data

    def create_tools(self) -> List[ToolDefinition]:
        """根据 MCP 能力构造工具定义列表。"""

        adapter = self

        class ExplainDiscountInput(BaseModel):
            is_vip: bool = Field(..., description="是否为 VIP 客户")
            amount: float = Field(..., description="订单金额，单位：元")

        class NormalizeProductInput(BaseModel):
            text: str = Field(..., description="用户输入的商品描述文本")

        class ValidateOrderInput(BaseModel):
            data: str = Field(..., description="待校验的 RDF/TTL 文本")
            format: str = Field(
                default="turtle",
                description="RDF 格式（turtle|json-ld 等），默认为 turtle",
            )

        class SearchProductsInput(BaseModel):
            keyword: str | None = Field(default=None, description="模糊搜索关键字")
            category: str | None = Field(default=None, description="商品分类过滤")
            brand: str | None = Field(default=None, description="品牌过滤")
            min_price: float | None = Field(default=None, description="最低价格")
            max_price: float | None = Field(default=None, description="最高价格")
            available_only: bool = Field(default=True, description="仅返回可用商品")
            limit: int = Field(default=20, description="最大返回数量")

        class GetProductDetailInput(BaseModel):
            product_id: int = Field(..., description="商品 ID")

        class CheckStockInput(BaseModel):
            product_id: int = Field(..., description="商品 ID")
            quantity: int = Field(..., description="期望购买数量")

        class ProductRecommendationInput(BaseModel):
            product_id: int | None = Field(default=None, description="参考的商品 ID")
            category: str | None = Field(default=None, description="候选分类过滤")
            limit: int = Field(default=5, description="返回推荐数")

        class ProductReviewsInput(BaseModel):
            product_id: int = Field(..., description="商品 ID")
            limit: int = Field(default=10, description="最大评价条数")

        class AddToCartInput(BaseModel):
            user_id: int = Field(..., description="用户 ID")
            product_id: int = Field(..., description="商品 ID")
            quantity: int = Field(default=1, description="加入购物车数量")

        class ViewCartInput(BaseModel):
            user_id: int = Field(..., description="用户 ID")

        class RemoveFromCartInput(BaseModel):
            user_id: int = Field(..., description="用户 ID")
            product_id: int = Field(..., description="商品 ID")

        class OrderItemInput(BaseModel):
            product_id: int = Field(..., description="商品 ID")
            quantity: int = Field(..., description="购买数量")
            unit_price: float | None = Field(default=None, description="单价，可为空表示使用数据库价格")

        class CreateOrderInput(BaseModel):
            user_id: int = Field(..., description="用户 ID")
            items: List[OrderItemInput] = Field(..., description="订单项列表")
            shipping_address: str = Field(..., description="收货地址")
            contact_phone: str = Field(..., description="联系电话")

        class OrderIdInput(BaseModel):
            order_id: int = Field(..., description="订单 ID")

        class UserOrdersInput(BaseModel):
            user_id: int = Field(..., description="用户 ID")
            status: str | None = Field(default=None, description="订单状态过滤")

        class ProcessPaymentInput(BaseModel):
            order_id: int = Field(..., description="订单 ID")
            payment_method: str = Field(..., description="支付方式")
            amount: float = Field(..., description="支付金额")

        class TrackShipmentInput(BaseModel):
            tracking_no: str = Field(..., description="运单号")

        class SupportTicketInput(BaseModel):
            user_id: int = Field(..., description="用户 ID")
            subject: str = Field(..., description="工单主题")
            description: str = Field(..., description="问题描述")
            order_id: int | None = Field(default=None, description="关联订单 ID")
            category: str = Field(default="售后", description="工单分类")
            priority: str = Field(default="medium", description="工单优先级")
            initial_message: str | None = Field(default=None, description="首条留言内容")

        class ProcessReturnInput(BaseModel):
            order_id: int = Field(..., description="订单 ID")
            user_id: int = Field(..., description="用户 ID")
            return_type: str = Field(default="return", description="退换货类型")
            reason: str = Field(default="", description="退货原因")
            product_category: str = Field(default="手机", description="商品分类")
            is_activated: bool = Field(default=False, description="电子产品是否已激活")

        class UserProfileInput(BaseModel):
            user_id: int = Field(..., description="用户 ID")

        def _explain_discount_tool(is_vip: bool, amount: float) -> str:
            result = adapter._invoke_or_raise(
                "ontology.explain_discount",
                {"is_vip": is_vip, "amount": amount},
            )
            return json.dumps(result, ensure_ascii=False)

        def _normalize_product_tool(text: str) -> str:
            result = adapter._invoke_or_raise(
                "ontology.normalize_product",
                {"text": text},
            )
            return json.dumps(result, ensure_ascii=False)

        def _validate_order_tool(data: str, format: str = "turtle") -> str:
            result = adapter._invoke_or_raise(
                "ontology.validate_order",
                {"data": data, "format": format},
            )
            return json.dumps(result, ensure_ascii=False)

        def _search_products_tool(**kwargs: Any) -> str:
            result = adapter._invoke_or_raise("commerce.search_products", kwargs)
            return json.dumps(result, ensure_ascii=False)

        def _get_product_detail_tool(product_id: int) -> str:
            result = adapter._invoke_or_raise(
                "commerce.get_product_detail",
                {"product_id": product_id},
            )
            return json.dumps(result, ensure_ascii=False)

        def _check_stock_tool(product_id: int, quantity: int) -> str:
            result = adapter._invoke_or_raise(
                "commerce.check_stock",
                {"product_id": product_id, "quantity": quantity},
            )
            return json.dumps(result, ensure_ascii=False)

        def _product_recommendation_tool(**kwargs: Any) -> str:
            result = adapter._invoke_or_raise("commerce.get_product_recommendations", kwargs)
            return json.dumps(result, ensure_ascii=False)

        def _product_reviews_tool(product_id: int, limit: int = 10) -> str:
            result = adapter._invoke_or_raise(
                "commerce.get_product_reviews",
                {"product_id": product_id, "limit": limit},
            )
            return json.dumps(result, ensure_ascii=False)

        def _add_to_cart_tool(user_id: int, product_id: int, quantity: int = 1) -> str:
            result = adapter._invoke_or_raise(
                "commerce.add_to_cart",
                {"user_id": user_id, "product_id": product_id, "quantity": quantity},
            )
            return json.dumps(result, ensure_ascii=False)

        def _view_cart_tool(user_id: int) -> str:
            result = adapter._invoke_or_raise(
                "commerce.view_cart",
                {"user_id": user_id},
            )
            return json.dumps(result, ensure_ascii=False)

        def _remove_from_cart_tool(user_id: int, product_id: int) -> str:
            result = adapter._invoke_or_raise(
                "commerce.remove_from_cart",
                {"user_id": user_id, "product_id": product_id},
            )
            return json.dumps(result, ensure_ascii=False)

        def _create_order_tool(user_id: int, items: List[Dict[str, Any]], shipping_address: str, contact_phone: str) -> str:
            result = adapter._invoke_or_raise(
                "commerce.create_order",
                {
                    "user_id": user_id,
                    "items": items,
                    "shipping_address": shipping_address,
                    "contact_phone": contact_phone,
                },
            )
            return json.dumps(result, ensure_ascii=False)

        def _get_order_detail_tool(order_id: int) -> str:
            result = adapter._invoke_or_raise(
                "commerce.get_order_detail",
                {"order_id": order_id},
            )
            return json.dumps(result, ensure_ascii=False)

        def _cancel_order_tool(order_id: int) -> str:
            result = adapter._invoke_or_raise(
                "commerce.cancel_order",
                {"order_id": order_id},
            )
            return json.dumps(result, ensure_ascii=False)

        def _get_user_orders_tool(user_id: int, status: str | None = None) -> str:
            payload = {"user_id": user_id}
            if status is not None:
                payload["status"] = status
            result = adapter._invoke_or_raise("commerce.get_user_orders", payload)
            return json.dumps(result, ensure_ascii=False)

        def _process_payment_tool(order_id: int, payment_method: str, amount: float) -> str:
            result = adapter._invoke_or_raise(
                "commerce.process_payment",
                {"order_id": order_id, "payment_method": payment_method, "amount": amount},
            )
            return json.dumps(result, ensure_ascii=False)

        def _track_shipment_tool(tracking_no: str) -> str:
            result = adapter._invoke_or_raise(
                "commerce.track_shipment",
                {"tracking_no": tracking_no},
            )
            return json.dumps(result, ensure_ascii=False)

        def _get_shipment_status_tool(order_id: int) -> str:
            result = adapter._invoke_or_raise(
                "commerce.get_shipment_status",
                {"order_id": order_id},
            )
            return json.dumps(result, ensure_ascii=False)

        def _create_support_ticket_tool(**kwargs: Any) -> str:
            result = adapter._invoke_or_raise("commerce.create_support_ticket", kwargs)
            return json.dumps(result, ensure_ascii=False)

        def _process_return_tool(**kwargs: Any) -> str:
            result = adapter._invoke_or_raise("commerce.process_return", kwargs)
            return json.dumps(result, ensure_ascii=False)

        def _get_user_profile_tool(user_id: int) -> str:
            result = adapter._invoke_or_raise(
                "commerce.get_user_profile",
                {"user_id": user_id},
            )
            return json.dumps(result, ensure_ascii=False)

        tools = [
            ToolDefinition(
                name="ontology_explain_discount",
                description="解释订单折扣规则，输入是否为 VIP 以及订单金额，返回折扣应用与来源。",
                func=_explain_discount_tool,
                args_schema=ExplainDiscountInput,
            ),
            ToolDefinition(
                name="ontology_normalize_product",
                description="将用户描述的商品文本归一化为本体中的标准概念。",
                func=_normalize_product_tool,
                args_schema=NormalizeProductInput,
            ),
            ToolDefinition(
                name="ontology_validate_order",
                description="使用 SHACL 形状校验 RDF 数据，检查是否符合本体约束。",
                func=_validate_order_tool,
                args_schema=ValidateOrderInput,
            ),
        ]

        tools.extend(
            [
                ToolDefinition(
                    name="commerce_search_products",
                    description="搜索电商商品，支持关键字、分类、品牌与价格过滤。",
                    func=lambda **kwargs: _search_products_tool(**kwargs),
                    args_schema=SearchProductsInput,
                ),
                ToolDefinition(
                    name="commerce_get_product_detail",
                    description="根据商品 ID 获取详细信息。",
                    func=_get_product_detail_tool,
                    args_schema=GetProductDetailInput,
                ),
                ToolDefinition(
                    name="commerce_check_stock",
                    description="检查商品库存是否满足指定数量。",
                    func=_check_stock_tool,
                    args_schema=CheckStockInput,
                ),
                ToolDefinition(
                    name="commerce_get_product_recommendations",
                    description="获取同类目或相关商品推荐列表。",
                    func=lambda **kwargs: _product_recommendation_tool(**kwargs),
                    args_schema=ProductRecommendationInput,
                ),
                ToolDefinition(
                    name="commerce_get_product_reviews",
                    description="查看指定商品的用户评价记录。",
                    func=_product_reviews_tool,
                    args_schema=ProductReviewsInput,
                ),
                ToolDefinition(
                    name="commerce_add_to_cart",
                    description="将商品加入用户购物车。",
                    func=_add_to_cart_tool,
                    args_schema=AddToCartInput,
                ),
                ToolDefinition(
                    name="commerce_view_cart",
                    description="查看用户购物车内的商品列表。",
                    func=_view_cart_tool,
                    args_schema=ViewCartInput,
                ),
                ToolDefinition(
                    name="commerce_remove_from_cart",
                    description="从购物车移除指定商品。",
                    func=_remove_from_cart_tool,
                    args_schema=RemoveFromCartInput,
                ),
                ToolDefinition(
                    name="commerce_create_order",
                    description="创建订单并应用本体推理的折扣与物流策略。",
                    func=lambda **kwargs: _create_order_tool(**kwargs),
                    args_schema=CreateOrderInput,
                ),
                ToolDefinition(
                    name="commerce_get_order_detail",
                    description="获取订单详情，包括用户与物流信息。",
                    func=_get_order_detail_tool,
                    args_schema=OrderIdInput,
                ),
                ToolDefinition(
                    name="commerce_cancel_order",
                    description="取消待处理或已支付的订单。",
                    func=_cancel_order_tool,
                    args_schema=OrderIdInput,
                ),
                ToolDefinition(
                    name="commerce_get_user_orders",
                    description="按状态筛选并返回用户的订单列表。",
                    func=lambda **kwargs: _get_user_orders_tool(**kwargs),
                    args_schema=UserOrdersInput,
                ),
                ToolDefinition(
                    name="commerce_process_payment",
                    description="创建支付记录并更新订单支付状态。",
                    func=_process_payment_tool,
                    args_schema=ProcessPaymentInput,
                ),
                ToolDefinition(
                    name="commerce_track_shipment",
                    description="根据运单号查询物流进度。",
                    func=_track_shipment_tool,
                    args_schema=TrackShipmentInput,
                ),
                ToolDefinition(
                    name="commerce_get_shipment_status",
                    description="按订单 ID 获取物流状态。",
                    func=_get_shipment_status_tool,
                    args_schema=OrderIdInput,
                ),
                ToolDefinition(
                    name="commerce_create_support_ticket",
                    description="创建客服工单并可附初始留言。",
                    func=lambda **kwargs: _create_support_ticket_tool(**kwargs),
                    args_schema=SupportTicketInput,
                ),
                ToolDefinition(
                    name="commerce_process_return",
                    description="依据本体规则发起退换货申请。",
                    func=lambda **kwargs: _process_return_tool(**kwargs),
                    args_schema=ProcessReturnInput,
                ),
                ToolDefinition(
                    name="commerce_get_user_profile",
                    description="获取用户画像信息与等级推理。",
                    func=_get_user_profile_tool,
                    args_schema=UserProfileInput,
                ),
            ]
        )

        return tools
