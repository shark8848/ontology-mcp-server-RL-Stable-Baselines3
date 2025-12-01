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
"""用户上下文动态提取器

功能：
1. 从对话中动态提取关键信息（用户ID、订单号、手机号等）
2. 保持信息唯一性和最新性
3. 自动注入到下一轮对话的提示词中
"""

import re
import json
from typing import Dict, Any, Optional, Set, List
from datetime import datetime
from dataclasses import dataclass, field, asdict

from agent.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class UserContext:
    """用户上下文信息"""
    # 核心身份信息
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    
    # 联系方式
    phone: Optional[str] = None
    address: Optional[str] = None
    
    # 订单相关
    recent_order_id: Optional[str] = None
    order_ids: Set[str] = field(default_factory=set)
    
    # 商品相关
    viewed_product_ids: Set[int] = field(default_factory=set)
    recent_product_id: Optional[int] = None
    
    # 时间戳
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 额外元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（处理Set类型）"""
        data = asdict(self)
        data['order_ids'] = list(self.order_ids)
        data['viewed_product_ids'] = list(self.viewed_product_ids)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserContext':
        """从字典创建实例"""
        if 'order_ids' in data and isinstance(data['order_ids'], list):
            data['order_ids'] = set(data['order_ids'])
        if 'viewed_product_ids' in data and isinstance(data['viewed_product_ids'], list):
            data['viewed_product_ids'] = set(data['viewed_product_ids'])
        return cls(**data)
    
    def merge(self, other: 'UserContext'):
        """合并另一个上下文的信息（保留最新的非空值）"""
        if other.user_id is not None:
            self.user_id = other.user_id
        if other.user_name:
            self.user_name = other.user_name
        if other.phone:
            self.phone = other.phone
        if other.address:
            self.address = other.address
        if other.recent_order_id:
            self.recent_order_id = other.recent_order_id
        if other.recent_product_id is not None:
            self.recent_product_id = other.recent_product_id
        
        # 合并集合
        self.order_ids.update(other.order_ids)
        self.viewed_product_ids.update(other.viewed_product_ids)
        
        # 更新时间戳
        self.last_updated = datetime.now().isoformat()
        
        # 合并元数据
        self.metadata.update(other.metadata)
    
    def to_prompt_context(self) -> str:
        """生成用于注入提示词的上下文"""
        lines = []
        
        if self.user_id is not None:
            lines.append(f"- 用户ID: {self.user_id}")
        if self.user_name:
            lines.append(f"- 用户姓名: {self.user_name}")
        if self.phone:
            lines.append(f"- 联系电话: {self.phone}")
        if self.address:
            lines.append(f"- 配送地址: {self.address}")
        
        # 订单信息（去重，只显示有效订单号）
        valid_orders = [oid for oid in self.order_ids if oid.startswith('ORD') and len(oid) >= 18]
        if self.recent_order_id and self.recent_order_id in valid_orders:
            lines.append(f"- 最近订单: {self.recent_order_id}")
        if len(valid_orders) > 1:
            # 排序并取最近3个（不包括recent_order_id）
            other_orders = [o for o in sorted(valid_orders) if o != self.recent_order_id][-2:]
            if other_orders:
                lines.append(f"- 历史订单: {', '.join(other_orders)}")
        
        # 商品信息（去重，只显示有效商品ID）
        valid_products = [pid for pid in self.viewed_product_ids if 1 <= pid <= 9999]
        if self.recent_product_id is not None and self.recent_product_id in valid_products:
            lines.append(f"- 当前关注商品ID: {self.recent_product_id}")
        if len(valid_products) > 1:
            # 排序并取最近5个（不包括recent_product_id）
            other_products = [p for p in sorted(valid_products) if p != self.recent_product_id][-4:]
            if other_products:
                lines.append(f"- 浏览过的商品ID: {', '.join(map(str, other_products))}")
        
        if not lines:
            return ""
        
        return "**用户上下文信息**:\n" + "\n".join(lines)
    
    def is_empty(self) -> bool:
        """判断是否为空上下文"""
        return (
            self.user_id is None and
            not self.user_name and
            not self.phone and
            not self.address and
            not self.recent_order_id and
            not self.order_ids and
            not self.viewed_product_ids
        )


class UserContextExtractor:
    """用户上下文提取器
    
    从对话内容、工具调用、Agent响应中提取关键信息
    """
    
    # 正则表达式模式
    PATTERNS = {
        'user_id': [
            r'用户\s*ID[：:\s]*(\d+)',  # 支持"用户ID 1"、"用户ID:1"等
            r'user_id[：:=\s]+(\d+)',
            r'用户编号[：:\s]*(\d+)',
            r'用户[：:\s]+(\d{1,6})\b',  # "用户 1" 或 "用户:1"，限制1-6位
            r'(?:会员|账号|帐户|account)号?[：:\s]*(\d{1,10})',
            r'(?:用户|会员|账号|帐户)\s*(?:ID|编号)?\s*(?:是|为|=)\s*(\d{1,10})',
            r'(?:我的|本人的)?\s*用户\s*ID\s*(?:是|为)\s*(\d{1,10})',
            r'(?:ID|account id)\s*(?:是|为|=)\s*(\d{1,10})',
            r'user\s*id\s*(?:is|=)\s*(\d{1,10})',
        ],
        'phone': [
            r'(?:电话|手机|联系方式)[：:：\s]+(1[3-9]\d{9})',
            r'(?:phone|mobile|contact_phone)[：:=\s]+(1[3-9]\d{9})',
            r'\b(1[3-9]\d{9})\b',  # 直接匹配手机号
            r'(?:注册|绑定|预留)?(?:电话|手机|手机号|联系电话)\s*(?:是|为|=)?\s*(1[3-9]\d{9})',
        ],
        'order_id': [
            # 只匹配完整的订单号格式（ORD开头 + 至少15位数字）
            r'\b(ORD\d{15,})\b',
            r'订单号?[：:：\s]+(ORD\d{15,})',
            r'order_id[：:=\s]+(ORD\d{15,})',
        ],
        'product_id': [
            r'商品\s*ID[：:：\s]*(\d{1,4})\b',  # 限制1-4位数字
            r'product_id[：:=\s]+(\d{1,4})\b',
        ],
        'address': [
            r'(?:地址|配送地址)[：:：\s]+([^，。,\n]{4,50})',
            r'(?:address|shipping_address)[：:=\s]+([^,\n]{4,50})',
            r'(?:送到|寄到)[：:：\s]+([^，。,"\n]{4,50})',
        ],
    }

    USER_ID_KEYS = {"user_id", "userid", "userId", "uid", "member_id", "account_id", "customer_id"}
    PHONE_KEYS = {"phone", "mobile", "contact_phone", "contact", "tel", "telephone"}
    ADDRESS_KEYS = {"address", "shipping_address", "delivery_address", "addr"}
    ORDER_ID_KEYS = {"order_id", "orderid", "order_no", "order_number", "orderno", "order"}
    PRODUCT_ID_KEYS = {"product_id", "productid", "sku_id", "sku", "item_id"}
    PRODUCT_COLLECTION_KEYS = {"items", "products", "cart_items", "results"}
    
    def __init__(self):
        """初始化提取器"""
        self.compiled_patterns = {}
        for key, patterns in self.PATTERNS.items():
            self.compiled_patterns[key] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in patterns
            ]
    
    def extract_from_text(self, text: str) -> UserContext:
        """从文本中提取信息
        
        Args:
            text: 输入文本
            
        Returns:
            UserContext: 提取的上下文
        """
        context = UserContext()
        
        # 提取用户ID
        for pattern in self.compiled_patterns['user_id']:
            match = pattern.search(text)
            if match:
                try:
                    context.user_id = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # 提取手机号
        for pattern in self.compiled_patterns['phone']:
            match = pattern.search(text)
            if match:
                phone = match.group(1)
                if self._is_valid_phone(phone):
                    context.phone = phone
                    break
        
        # 提取订单号（可能有多个）- 只保留有效格式
        for pattern in self.compiled_patterns['order_id']:
            for match in pattern.finditer(text):
                order_id = match.group(1)
                # 验证订单号格式：必须是ORD开头且至少15位数字
                if order_id.startswith('ORD') and len(order_id) >= 18:
                    context.order_ids.add(order_id)
                    context.recent_order_id = order_id  # 最后一个作为最近订单
        
        # 提取商品ID（可能有多个）- 只保留合理范围的ID
        for pattern in self.compiled_patterns['product_id']:
            for match in pattern.finditer(text):
                try:
                    product_id = int(match.group(1))
                    # 只保留1-9999范围内的商品ID（排除订单号等长数字）
                    if 1 <= product_id <= 9999:
                        context.viewed_product_ids.add(product_id)
                        context.recent_product_id = product_id  # 最后一个作为当前商品
                except (ValueError, IndexError):
                    continue
        
        # 提取地址
        for pattern in self.compiled_patterns['address']:
            match = pattern.search(text)
            if match:
                address = match.group(1).strip()
                if len(address) >= 5:  # 至少5个字符
                    context.address = address
                    break
        
        return context

    @staticmethod
    def is_valid_order_id(order_id: str) -> bool:
        """简单校验订单号是否为有效格式（ORD开头且长度合理）"""
        if not isinstance(order_id, str):
            return False
        return order_id.startswith('ORD') and len(order_id) >= 18
    
    def extract_from_tool_calls(self, tool_calls: list) -> UserContext:
        """从工具调用中提取信息
        
        Args:
            tool_calls: 工具调用列表
            
        Returns:
            UserContext: 提取的上下文
        """
        context = UserContext()
        
        for tool_call in tool_calls:
            tool_name = str(tool_call.get('tool', '')).lower()
            tool_input = tool_call.get('input')
            observation = tool_call.get('observation')

            self._harvest_from_payload(tool_input, context)
            self._harvest_from_payload(observation, context)

            if 'create_order' in tool_name:
                extracted = self.extract_from_text(str(observation))
                if extracted.recent_order_id:
                    if extracted.recent_order_id.startswith('ORD') and len(extracted.recent_order_id) >= 18:
                        context.order_ids.add(extracted.recent_order_id)
                        context.recent_order_id = extracted.recent_order_id
                if isinstance(tool_input, str):
                    extracted_input = self.extract_from_text(tool_input)
                    if extracted_input.recent_order_id:
                        if extracted_input.recent_order_id.startswith('ORD') and len(extracted_input.recent_order_id) >= 18:
                            context.order_ids.add(extracted_input.recent_order_id)
                            context.recent_order_id = extracted_input.recent_order_id
        
        return context
    
    def extract_from_conversation(
        self,
        user_input: str,
        agent_response: str,
        tool_calls: list = None
    ) -> UserContext:
        """从完整对话中提取信息
        
        Args:
            user_input: 用户输入
            agent_response: Agent响应
            tool_calls: 工具调用列表
            
        Returns:
            UserContext: 提取的上下文
        """
        context = UserContext()
        
        # 从用户输入提取
        user_context = self.extract_from_text(user_input)
        context.merge(user_context)
        
        # 从Agent响应提取
        response_context = self.extract_from_text(agent_response)
        context.merge(response_context)
        
        # 从工具调用提取
        if tool_calls:
            tool_context = self.extract_from_tool_calls(tool_calls)
            context.merge(tool_context)
        
        if not context.is_empty():
            LOGGER.debug(
                "提取用户上下文: user_id=%s, phone=%s, order_id=%s, product_ids=%s",
                context.user_id, context.phone, context.recent_order_id,
                len(context.viewed_product_ids)
            )
        
        return context
    
    def _harvest_from_payload(self, payload: Any, context: UserContext):
        """从任意数据结构中提取信息."""
        if payload is None:
            return

        if isinstance(payload, str):
            stripped = payload.strip()
            if not stripped:
                return
            parsed = self._try_parse_json(stripped)
            if parsed is not None and parsed is not payload:
                self._harvest_from_payload(parsed, context)
                return
            nested = self.extract_from_text(stripped)
            context.merge(nested)
            return

        if isinstance(payload, dict):
            self._harvest_from_mapping(payload, context)
            return

        if isinstance(payload, list):
            for item in payload:
                self._harvest_from_payload(item, context)
            return

        if isinstance(payload, (int, float)):
            self._record_numeric_candidate(payload, context)

    def _harvest_from_mapping(self, data: Dict[str, Any], context: UserContext):
        for key, value in data.items():
            lowered = str(key).strip().lower()
            if not lowered:
                continue

            if lowered == "result" and isinstance(value, str):
                parsed = self._try_parse_json(value)
                if parsed is not None:
                    self._harvest_from_payload(parsed, context)
                    continue

            if lowered in self.PRODUCT_COLLECTION_KEYS and isinstance(value, list):
                self._harvest_from_payload(value, context)
                continue

            self._process_structured_field(lowered, value, context)

    def _process_structured_field(self, key: str, value: Any, context: UserContext):
        if key in self.USER_ID_KEYS:
            user_id = self._safe_int(value)
            if user_id is not None:
                context.user_id = user_id
            return

        if key in self.PHONE_KEYS:
            phone = str(value).strip()
            if self._is_valid_phone(phone):
                context.phone = phone
            return

        if key in self.ADDRESS_KEYS:
            address = str(value).strip()
            if len(address) >= 4:
                context.address = address
            return

        if key in self.ORDER_ID_KEYS:
            order_id = str(value).strip()
            if self.is_valid_order_id(order_id):
                context.order_ids.add(order_id)
                context.recent_order_id = order_id
            return

        if key in self.PRODUCT_ID_KEYS:
            product_id = self._safe_int(value)
            self._record_product_id(product_id, context)
            return

        if isinstance(value, (dict, list)):
            self._harvest_from_payload(value, context)
            return

        if isinstance(value, str):
            nested = self.extract_from_text(value)
            context.merge(nested)

    def _record_product_id(self, product_id: Optional[int], context: UserContext):
        if product_id is None:
            return
        if 1 <= product_id <= 9999:
            context.viewed_product_ids.add(product_id)
            context.recent_product_id = product_id

    def _record_numeric_candidate(self, value: float, context: UserContext):
        int_value = self._safe_int(value)
        self._record_product_id(int_value, context)

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _try_parse_json(value: str) -> Optional[Any]:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    
    @staticmethod
    def _is_valid_phone(phone: str) -> bool:
        """验证手机号格式"""
        # 中国大陆手机号：1开头，第二位3-9，共11位
        return bool(re.match(r'^1[3-9]\d{9}$', phone))


class UserContextManager:
    """用户上下文管理器
    
    维护会话级别的用户上下文，支持持久化
    """
    
    def __init__(self, session_id: str):
        """初始化管理器
        
        Args:
            session_id: 会话ID
        """
        self.session_id = session_id
        self.context = UserContext()
        self.extractor = UserContextExtractor()
        self.history: list = []  # 历史上下文快照
        
        LOGGER.info("初始化用户上下文管理器: session=%s", session_id)
    
    def update_from_conversation(
        self,
        user_input: str,
        agent_response: str,
        tool_calls: list = None
    ):
        """从对话中更新上下文
        
        Args:
            user_input: 用户输入
            agent_response: Agent响应
            tool_calls: 工具调用列表
        """
        # 提取新信息
        new_context = self.extractor.extract_from_conversation(
            user_input, agent_response, tool_calls
        )
        
        # 合并到当前上下文（保留历史快照）
        if not new_context.is_empty():
            self.history.append(self.context.to_dict())
            self.context.merge(new_context)
            LOGGER.info("用户上下文已更新: %s", self._format_summary())

    def ingest_tool_call(self, tool_name: str, tool_input: Any, observation: Any):
        """基于单次工具交互更新上下文（实时应用场景）"""
        snapshot = [{"tool": tool_name, "input": tool_input, "observation": observation}]
        extracted = self.extractor.extract_from_tool_calls(snapshot)
        if extracted.is_empty():
            return
        self.history.append(self.context.to_dict())
        self.context.merge(extracted)
        LOGGER.info("用户上下文快速更新: %s", self._format_summary())

    def ingest_free_text(self, text: str):
        """直接基于原始文本内容更新上下文（无须等待完整对话存档）。"""
        if not text:
            return
        extracted = self.extractor.extract_from_text(text)
        if extracted.is_empty():
            return
        self.history.append(self.context.to_dict())
        self.context.merge(extracted)
        LOGGER.info("用户上下文已根据文本更新: %s", self._format_summary())

    def set_recent_order(self, order_id: str):
        """显式设置最近订单号（带验证），并记录历史快照

        Args:
            order_id: 订单号字符串
        """
        if not order_id:
            return
        # 仅接受有效的 ORD 格式
        if UserContextExtractor.is_valid_order_id(order_id):
            # 记录快照
            self.history.append(self.context.to_dict())
            self.context.order_ids.add(order_id)
            self.context.recent_order_id = order_id
            self.context.last_updated = datetime.now().isoformat()
            LOGGER.info("用户上下文最近订单已显式更新: %s", order_id)
        else:
            LOGGER.debug("忽略无效订单号设置请求: %s", order_id)
    
    def get_context(self) -> UserContext:
        """获取当前上下文"""
        return self.context
    
    def get_prompt_injection(self) -> str:
        """获取用于注入提示词的文本"""
        if self.context.is_empty():
            return ""
        return self.context.to_prompt_context()
    
    def clear(self):
        """清空上下文"""
        self.context = UserContext()
        self.history.clear()
        LOGGER.info("用户上下文已清空")
    
    def save_to_json(self, filepath: str):
        """保存到JSON文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'session_id': self.session_id,
                    'context': self.context.to_dict(),
                    'history': self.history,
                }, f, ensure_ascii=False, indent=2)
            LOGGER.info("用户上下文已保存: %s", filepath)
        except Exception as e:
            LOGGER.error("保存用户上下文失败: %s", e)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'UserContextManager':
        """从JSON文件加载"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            manager = cls(data['session_id'])
            manager.context = UserContext.from_dict(data['context'])
            manager.history = data.get('history', [])
            
            LOGGER.info("用户上下文已加载: %s", filepath)
            return manager
        except Exception as e:
            LOGGER.error("加载用户上下文失败: %s", e)
            raise
    
    def _format_summary(self) -> str:
        """格式化摘要（用于日志）"""
        parts = []
        if self.context.user_id is not None:
            parts.append(f"user_id={self.context.user_id}")
        if self.context.phone:
            parts.append(f"phone={self.context.phone[-4:]}")  # 只显示后4位
        if self.context.recent_order_id:
            parts.append(f"order={self.context.recent_order_id[-8:]}")  # 只显示后8位
        if self.context.recent_product_id is not None:
            parts.append(f"product={self.context.recent_product_id}")
        return ", ".join(parts) if parts else "empty"
