from __future__ import annotations
# Copyright (c) 2025 shark8848
# MIT License
#
# Ontology MCP Server - 电商 AI 助手系统
# 本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI
#
# Author: shark8848
# Repository: https://github.com/shark8848/ontology-mcp-server
"""对话状态管理模块 - 跟踪购物会话状态"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .logger import get_logger

logger = get_logger(__name__)


class ConversationStage(str, Enum):
    """对话阶段枚举"""
    GREETING = "greeting"           # 初次问候
    BROWSING = "browsing"           # 浏览商品
    SELECTING = "selecting"         # 选择商品
    CART_MANAGEMENT = "cart"        # 购物车管理
    CHECKOUT = "checkout"           # 结算中
    ORDER_TRACKING = "tracking"     # 订单跟踪
    CUSTOMER_SERVICE = "service"    # 售后服务
    IDLE = "idle"                   # 空闲状态


@dataclass
class UserContext:
    """用户上下文信息"""
    user_id: Optional[int] = None
    is_vip: bool = False
    username: Optional[str] = None
    last_viewed_products: List[int] = field(default_factory=list)
    cart_item_count: int = 0
    recent_order_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "is_vip": self.is_vip,
            "username": self.username,
            "last_viewed_products": self.last_viewed_products,
            "cart_item_count": self.cart_item_count,
            "recent_order_id": self.recent_order_id,
        }


@dataclass
class SessionState:
    """会话状态"""
    session_id: str
    stage: ConversationStage = ConversationStage.GREETING
    user_context: UserContext = field(default_factory=UserContext)
    current_product_id: Optional[int] = None
    current_order_id: Optional[int] = None
    intent_history: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    
    def update_stage(self, new_stage: ConversationStage, reason: str = "") -> None:
        """更新对话阶段"""
        old_stage = self.stage
        self.stage = new_stage
        self.last_active = datetime.now()
        logger.info(
            "会话阶段变更: %s -> %s (session=%s, reason=%s)",
            old_stage.value,
            new_stage.value,
            self.session_id,
            reason or "未指定",
        )
    
    def add_intent(self, intent: str) -> None:
        """记录用户意图"""
        self.intent_history.append(intent)
        self.last_active = datetime.now()
        # 只保留最近10条意图
        if len(self.intent_history) > 10:
            self.intent_history = self.intent_history[-10:]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "stage": self.stage.value,
            "user_context": self.user_context.to_dict(),
            "current_product_id": self.current_product_id,
            "current_order_id": self.current_order_id,
            "intent_history": self.intent_history,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }


class ConversationStateManager:
    """对话状态管理器"""
    
    def __init__(self):
        self.state: Optional[SessionState] = None
        logger.info("对话状态管理器初始化")
    
    def initialize_session(self, session_id: str) -> SessionState:
        """初始化新会话"""
        self.state = SessionState(session_id=session_id)
        logger.info("创建新会话: %s", session_id)
        return self.state
    
    def get_state(self) -> Optional[SessionState]:
        """获取当前会话状态"""
        return self.state
    
    def update_user_context(
        self,
        *,
        user_id: Optional[int] = None,
        is_vip: Optional[bool] = None,
        username: Optional[str] = None,
    ) -> None:
        """更新用户上下文"""
        if not self.state:
            logger.warning("尝试更新用户上下文但会话未初始化")
            return
        
        if user_id is not None:
            self.state.user_context.user_id = user_id
        if is_vip is not None:
            self.state.user_context.is_vip = is_vip
        if username is not None:
            self.state.user_context.username = username
        
        logger.debug("用户上下文已更新: %s", self.state.user_context.to_dict())
    
    def infer_stage_from_intent(self, user_input: str, tool_calls: List[Dict[str, Any]]) -> ConversationStage:
        """从用户输入和工具调用推断对话阶段
        
        Args:
            user_input: 用户输入文本
            tool_calls: 本轮调用的工具列表
            
        Returns:
            ConversationStage: 推断的对话阶段
        """
        input_lower = user_input.lower()
        tool_names = [call.get("tool", "") for call in tool_calls]

        def _log_and_return(stage: ConversationStage, reason: str) -> ConversationStage:
            logger.info(
                "对话阶段推理[推理方式=工具优先+关键词回退]: stage=%s reason=%s tools=%s",
                stage.value,
                reason,
                tool_names,
            )
            return stage
        
        # 基于工具调用推断
        if any("search_products" in t for t in tool_names):
            return _log_and_return(ConversationStage.BROWSING, "tool=search_products")
        
        if any("get_product_detail" in t for t in tool_names):
            return _log_and_return(ConversationStage.SELECTING, "tool=get_product_detail")
        
        if any(t in ["add_to_cart", "view_cart", "remove_from_cart"] for t in tool_names):
            return _log_and_return(ConversationStage.CART_MANAGEMENT, "tool=cart_ops")
        
        if any("create_order" in t for t in tool_names):
            return _log_and_return(ConversationStage.CHECKOUT, "tool=create_order")
        
        if any(t in ["process_payment", "get_order_detail", "track_shipment"] for t in tool_names):
            return _log_and_return(ConversationStage.ORDER_TRACKING, "tool=order_tracking")
        
        if any(t in ["create_support_ticket", "process_return"] for t in tool_names):
            return _log_and_return(ConversationStage.CUSTOMER_SERVICE, "tool=customer_service")
        
        # 基于关键词推断
        browse_keywords = ["搜索", "找", "看看", "推荐", "有什么", "商品"]
        if any(kw in input_lower for kw in browse_keywords):
            return _log_and_return(ConversationStage.BROWSING, "keyword=browse")
        
        cart_keywords = ["购物车", "加入", "加购", "移除"]
        if any(kw in input_lower for kw in cart_keywords):
            return _log_and_return(ConversationStage.CART_MANAGEMENT, "keyword=cart")
        
        order_keywords = ["下单", "购买", "结算", "支付"]
        if any(kw in input_lower for kw in order_keywords):
            return _log_and_return(ConversationStage.CHECKOUT, "keyword=checkout")
        
        tracking_keywords = ["订单", "物流", "快递", "发货"]
        if any(kw in input_lower for kw in tracking_keywords):
            return _log_and_return(ConversationStage.ORDER_TRACKING, "keyword=tracking")
        
        service_keywords = ["退货", "换货", "售后", "客服", "投诉"]
        if any(kw in input_lower for kw in service_keywords):
            return _log_and_return(ConversationStage.CUSTOMER_SERVICE, "keyword=service")
        
        # 默认返回当前阶段或空闲
        default_stage = self.state.stage if self.state else ConversationStage.IDLE
        return _log_and_return(default_stage, "fallback=current_stage")
    
    def update_from_tool_results(self, tool_log: List[Dict[str, Any]]) -> None:
        """从工具调用结果更新状态
        
        Args:
            tool_log: 工具调用日志列表
        """
        if not self.state:
            return
        
        for entry in tool_log:
            tool_name = entry.get("tool", "")
            observation = entry.get("observation", "")
            
            # 更新购物车数量
            if tool_name == "view_cart":
                try:
                    import json
                    result = json.loads(observation)
                    if isinstance(result, dict) and "items" in result:
                        self.state.user_context.cart_item_count = len(result["items"])
                except Exception:
                    pass
            
            # 更新订单ID
            if tool_name == "create_order":
                try:
                    import json
                    result = json.loads(observation)
                    if isinstance(result, dict) and "order" in result:
                        order_data = result["order"]
                        if isinstance(order_data, dict) and "order_id" in order_data:
                            self.state.current_order_id = order_data["order_id"]
                            self.state.user_context.recent_order_id = order_data["order_id"]
                except Exception:
                    pass
            
            # 更新商品浏览历史
            if tool_name == "get_product_detail":
                try:
                    import json
                    inp = entry.get("input", {})
                    if isinstance(inp, dict) and "product_id" in inp:
                        product_id = inp["product_id"]
                        if product_id not in self.state.user_context.last_viewed_products:
                            self.state.user_context.last_viewed_products.append(product_id)
                            # 只保留最近5个
                            if len(self.state.user_context.last_viewed_products) > 5:
                                self.state.user_context.last_viewed_products = (
                                    self.state.user_context.last_viewed_products[-5:]
                                )
                        self.state.current_product_id = product_id
                except Exception:
                    pass
    
    def get_context_summary(self) -> str:
        """生成当前状态的文本摘要
        
        Returns:
            str: 状态摘要文本
        """
        if not self.state:
            return "无活跃会话"
        
        summary_parts = [f"阶段: {self.state.stage.value}"]
        
        if self.state.user_context.is_vip:
            summary_parts.append("VIP客户")
        
        if self.state.user_context.cart_item_count > 0:
            summary_parts.append(f"购物车: {self.state.user_context.cart_item_count}件")
        
        if self.state.current_order_id:
            summary_parts.append(f"当前订单: #{self.state.current_order_id}")
        
        if self.state.user_context.last_viewed_products:
            summary_parts.append(
                f"浏览过: {len(self.state.user_context.last_viewed_products)}个商品"
            )
        
        return " | ".join(summary_parts)
    
    def clear_session(self) -> None:
        """清除当前会话"""
        if self.state:
            logger.info("清除会话: %s", self.state.session_id)
        self.state = None
