"""
Copyright (c) 2025 shark8848
MIT License

Ontology MCP Server - 电商 AI 助手系统
本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI

Author: shark8848
Repository: https://github.com/shark8848/ontology-mcp-server
"""

"""
多轮意图识别与跟踪

功能：
1. 跟踪用户在多轮对话中的意图演变
2. 识别复合意图（如：先问价格，再问库存 → 购买意向）
3. 基于意图历史预测下一步可能的操作
4. 支持意图分层（主意图 + 子意图）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import datetime
import re


class IntentCategory(Enum):
    """意图类别"""
    # 浏览类
    SEARCH = "search"                 # 搜索商品
    BROWSE = "browse"                 # 浏览商品列表
    VIEW_DETAIL = "view_detail"       # 查看商品详情
    
    # 咨询类
    PRICE_INQUIRY = "price_inquiry"   # 询问价格
    STOCK_INQUIRY = "stock_inquiry"   # 询问库存
    SPEC_INQUIRY = "spec_inquiry"     # 询问规格参数
    
    # 购买类
    ADD_TO_CART = "add_to_cart"       # 加入购物车
    VIEW_CART = "view_cart"           # 查看购物车
    CHECKOUT = "checkout"             # 结账下单
    
    # 订单类
    ORDER_STATUS = "order_status"     # 查询订单状态
    ORDER_TRACK = "order_track"       # 物流跟踪
    
    # 服务类
    VIP_INQUIRY = "vip_inquiry"       # VIP 咨询
    RECOMMENDATION = "recommendation" # 寻求推荐
    CHART_REQUEST = "chart_request"   # 数据可视化需求
    
    # 其他
    GREETING = "greeting"             # 问候
    UNKNOWN = "unknown"               # 未知意图


class IntentConfidence(Enum):
    """意图识别置信度"""
    HIGH = 0.8      # 高置信度
    MEDIUM = 0.5    # 中等置信度
    LOW = 0.3       # 低置信度


@dataclass
class Intent:
    """单个意图"""
    category: IntentCategory
    confidence: float  # 0-1
    extracted_entities: Dict[str, Any] = field(default_factory=dict)  # 提取的实体
    turn_id: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    raw_input: str = ""
    
    def __str__(self):
        return f"{self.category.value}({self.confidence:.2f})"


@dataclass
class CompositeIntent:
    """复合意图（多轮意图的组合）"""
    name: str
    sub_intents: List[Intent] = field(default_factory=list)
    confidence: float = 0.0
    description: str = ""
    
    def add_intent(self, intent: Intent):
        """添加子意图"""
        self.sub_intents.append(intent)
        # 更新置信度（使用平均值）
        if self.sub_intents:
            self.confidence = sum(i.confidence for i in self.sub_intents) / len(self.sub_intents)
    
    def __str__(self):
        return f"{self.name}: {[str(i) for i in self.sub_intents]}"


class IntentRecognizer:
    """意图识别器（基于规则 + 关键词匹配）"""
    
    # 意图识别规则（关键词映射）
    INTENT_PATTERNS = {
        IntentCategory.SEARCH: [
            r"搜索|找|查找|有没有|给我找",
            r"search|find",
        ],
        IntentCategory.VIEW_DETAIL: [
            r"看看|查看|详情|介绍|怎么样",
            r"detail|view|show",
        ],
        IntentCategory.PRICE_INQUIRY: [
            r"多少钱|价格|多贵|便宜|贵",
            r"price|cost|expensive",
        ],
        IntentCategory.STOCK_INQUIRY: [
            r"有货|库存|还有吗|能买到吗",
            r"stock|inventory|available",
        ],
        IntentCategory.SPEC_INQUIRY: [
            r"参数|规格|配置|性能",
            r"spec|parameter|configuration",
        ],
        IntentCategory.ADD_TO_CART: [
            r"加购|加入购物车|放进|买这个",
            r"add.*cart|buy",
        ],
        IntentCategory.VIEW_CART: [
            r"购物车|看看购物车|有什么",
            r"cart|shopping.*cart",
        ],
        IntentCategory.CHECKOUT: [
            r"下单|结账|支付|购买|确认订单",
            r"checkout|order|pay|purchase",
        ],
        IntentCategory.ORDER_STATUS: [
            r"订单|订单状态|我的订单",
            r"order.*status|my.*order",
        ],
        IntentCategory.ORDER_TRACK: [
            r"物流|快递|到哪了|配送",
            r"track|delivery|shipping",
        ],
        IntentCategory.VIP_INQUIRY: [
            r"会员|vip|升级|权益",
            r"vip|member|premium",
        ],
        IntentCategory.RECOMMENDATION: [
            r"推荐|建议|什么好|帮我选",
            r"recommend|suggest|best",
        ],
        IntentCategory.CHART_REQUEST: [
            r"趋势图|柱状图|饼状图|折线图|图表|统计图|可视化|对比图",
            r"chart|graph|plot|trend|bar|pie|visualize",
        ],
        IntentCategory.GREETING: [
            r"你好|嗨|hello|hi|早上好|晚上好",
        ],
    }

    CHART_KEYWORDS = {
        "trend": ["趋势", "trend", "折线", "line"],
        "bar": ["柱", "bar"],
        "pie": ["饼", "占比", "pie"],
        "comparison": ["对比", "比较", "compare"],
    }
    
    def recognize(self, user_input: str, turn_id: int = 0) -> List[Intent]:
        """识别用户意图（允许多标签）"""
        user_input_lower = user_input.lower()
        matches: List[Intent] = []

        # 遍历所有意图模式
        for intent_category, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    entities = self._extract_entities(user_input)
                    matches.append(
                        Intent(
                            category=intent_category,
                            confidence=IntentConfidence.HIGH.value,
                            extracted_entities=entities,
                            turn_id=turn_id,
                            raw_input=user_input,
                        )
                    )
                    break

        if not matches:
            matches.append(
                Intent(
                    category=IntentCategory.UNKNOWN,
                    confidence=IntentConfidence.LOW.value,
                    turn_id=turn_id,
                    raw_input=user_input,
                )
            )

        return matches
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """提取实体（商品名、数量、价格等）"""
        entities = {}
        
        # 提取数量
        quantity_match = re.search(r"(\d+)\s*个|(\d+)\s*件", text)
        if quantity_match:
            entities["quantity"] = int(quantity_match.group(1) or quantity_match.group(2))
        
        # 提取价格范围
        price_match = re.search(r"(\d+)\s*[-到]\s*(\d+)\s*元", text)
        if price_match:
            entities["price_range"] = (int(price_match.group(1)), int(price_match.group(2)))
        
        # 提取商品ID（如果明确提到）
        product_id_match = re.search(r"prod_\w+|商品\s*(\d+)", text)
        if product_id_match:
            entities["product_id"] = product_id_match.group(0)
        
        for chart_type, words in self.CHART_KEYWORDS.items():
            if any(word in text for word in words):
                entities.setdefault("chart_types", set()).add(chart_type)
        if "chart_types" in entities:
            entities["chart_types"] = sorted(entities["chart_types"])
        return entities


class IntentTracker:
    """多轮意图跟踪器"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.intent_history: List[Intent] = []
        self.intent_labels: List[Intent] = []
        self.recognizer = IntentRecognizer()
        self.composite_intents: List[CompositeIntent] = []
    
    def track_intent(self, user_input: str, turn_id: int) -> Intent:
        """跟踪当前意图（返回主意图，并保留所有标签）"""
        intents = self.recognizer.recognize(user_input, turn_id)
        self.intent_labels.extend(intents)
        primary = intents[0]
        self.intent_history.append(primary)
        self._detect_composite_intents()
        return primary
    
    def _detect_composite_intents(self):
        """检测复合意图"""
        if len(self.intent_history) < 2:
            return
        
        # 最近的意图序列
        recent_intents = self.intent_history[-5:]  # 最近5轮
        
        # 检测购买意向（咨询 → 加购/下单）
        self._detect_purchase_intent(recent_intents)
        
        # 检测比较意向（多次查看详情）
        self._detect_comparison_intent(recent_intents)
        
        # 检测售后意向（订单 → 物流）
        self._detect_after_sales_intent(recent_intents)
    
    def _detect_purchase_intent(self, intents: List[Intent]):
        """检测购买意向"""
        inquiry_intents = [
            i for i in intents 
            if i.category in [IntentCategory.PRICE_INQUIRY, IntentCategory.STOCK_INQUIRY, IntentCategory.SPEC_INQUIRY]
        ]
        purchase_intents = [
            i for i in intents 
            if i.category in [IntentCategory.ADD_TO_CART, IntentCategory.CHECKOUT]
        ]
        
        if inquiry_intents and purchase_intents:
            # 检查是否已存在
            if not any(c.name == "purchase_intent" for c in self.composite_intents):
                composite = CompositeIntent(
                    name="purchase_intent",
                    description="用户先咨询商品信息，然后进行购买操作",
                )
                for intent in inquiry_intents + purchase_intents:
                    composite.add_intent(intent)
                self.composite_intents.append(composite)
    
    def _detect_comparison_intent(self, intents: List[Intent]):
        """检测比较意向"""
        view_intents = [i for i in intents if i.category == IntentCategory.VIEW_DETAIL]
        
        if len(view_intents) >= 2:
            # 检查是否已存在
            if not any(c.name == "comparison_intent" for c in self.composite_intents):
                composite = CompositeIntent(
                    name="comparison_intent",
                    description="用户正在比较多个商品",
                )
                for intent in view_intents:
                    composite.add_intent(intent)
                self.composite_intents.append(composite)
    
    def _detect_after_sales_intent(self, intents: List[Intent]):
        """检测售后意向"""
        order_intents = [
            i for i in intents 
            if i.category in [IntentCategory.ORDER_STATUS, IntentCategory.ORDER_TRACK]
        ]
        
        if len(order_intents) >= 1:
            # 检查是否已存在
            if not any(c.name == "after_sales_intent" for c in self.composite_intents):
                composite = CompositeIntent(
                    name="after_sales_intent",
                    description="用户关注订单和物流信息",
                )
                for intent in order_intents:
                    composite.add_intent(intent)
                self.composite_intents.append(composite)
    
    def get_current_intent(self) -> Optional[Intent]:
        """获取当前意图"""
        return self.intent_history[-1] if self.intent_history else None
    
    def get_intent_sequence(self, last_n: int = 5) -> List[Intent]:
        """获取最近的意图序列"""
        return self.intent_history[-last_n:]
    
    def get_composite_intents(self) -> List[CompositeIntent]:
        """获取识别出的复合意图"""
        return self.composite_intents
    
    def predict_next_intent(self) -> List[IntentCategory]:
        """预测下一步可能的意图"""
        if not self.intent_history:
            return [IntentCategory.GREETING, IntentCategory.SEARCH]
        
        current_intent = self.get_current_intent()
        
        # 基于规则的意图预测
        prediction_rules = {
            IntentCategory.SEARCH: [IntentCategory.VIEW_DETAIL, IntentCategory.BROWSE],
            IntentCategory.VIEW_DETAIL: [IntentCategory.PRICE_INQUIRY, IntentCategory.STOCK_INQUIRY, IntentCategory.ADD_TO_CART],
            IntentCategory.PRICE_INQUIRY: [IntentCategory.ADD_TO_CART, IntentCategory.VIEW_CART],
            IntentCategory.STOCK_INQUIRY: [IntentCategory.ADD_TO_CART],
            IntentCategory.ADD_TO_CART: [IntentCategory.VIEW_CART, IntentCategory.CHECKOUT],
            IntentCategory.VIEW_CART: [IntentCategory.CHECKOUT],
            IntentCategory.CHECKOUT: [IntentCategory.ORDER_STATUS, IntentCategory.ORDER_TRACK],
        }
        
        return prediction_rules.get(current_intent.category, [])
    
    def get_summary(self) -> Dict[str, Any]:
        """获取意图跟踪摘要"""
        intent_distribution = {}
        for intent in self.intent_history:
            category = intent.category.value
            intent_distribution[category] = intent_distribution.get(category, 0) + 1
        
        return {
            "session_id": self.session_id,
            "total_turns": len(self.intent_history),
            "intent_distribution": intent_distribution,
            "intent_labels": [i.category.value for i in self.intent_labels[-5:]],
            "composite_intents": [
                {
                    "name": c.name,
                    "description": c.description,
                    "confidence": c.confidence,
                    "sub_intents": [str(i) for i in c.sub_intents],
                }
                for c in self.composite_intents
            ],
            "current_intent": str(self.get_current_intent()) if self.get_current_intent() else None,
            "predicted_next": [i.value for i in self.predict_next_intent()],
        }
