"""
Copyright (c) 2025 shark8848
MIT License

状态提取器 - 将对话上下文转换为 RL 状态向量

状态空间设计（128维）：
- 用户上下文编码 (32维)
- 对话上下文编码 (64维)
- 商品库状态编码 (32维)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json


@dataclass
class StateComponents:
    """状态向量各组件"""
    user_context: np.ndarray       # 32维
    conversation_context: np.ndarray  # 64维
    product_state: np.ndarray      # 32维
    
    def to_vector(self) -> np.ndarray:
        """合并为完整状态向量"""
        return np.concatenate([
            self.user_context,
            self.conversation_context,
            self.product_state
        ])
    
    def __repr__(self) -> str:
        return (
            f"StateComponents(user={self.user_context.shape}, "
            f"conversation={self.conversation_context.shape}, "
            f"product={self.product_state.shape})"
        )


class StateExtractor:
    """状态提取器"""
    
    # 状态空间维度
    USER_CONTEXT_DIM = 32
    CONVERSATION_CONTEXT_DIM = 64
    PRODUCT_STATE_DIM = 32
    TOTAL_DIM = USER_CONTEXT_DIM + CONVERSATION_CONTEXT_DIM + PRODUCT_STATE_DIM
    
    # 对话阶段映射
    STAGE_MAP = {
        "greeting": 0,
        "browsing": 1,
        "selecting": 2,
        "cart": 3,
        "checkout": 4,
        "tracking": 5,
        "service": 6,
        "idle": 7,
    }
    
    # 意图类别映射（根据 intent_tracker.py）
    INTENT_MAP = {
        "greeting": 0,
        "search": 1,
        "browse": 2,
        "view_detail": 3,
        "add_to_cart": 4,
        "view_cart": 5,
        "checkout": 6,
        "payment": 7,
        "track_order": 8,
        "review": 9,
        "return": 10,
        "support": 11,
        "recommendation": 12,
        "unknown": 13,
    }
    
    def __init__(self, use_text_embedding: bool = False):
        """
        初始化状态提取器
        
        Args:
            use_text_embedding: 是否使用文本嵌入（需要 sentence-transformers）
        """
        self.use_text_embedding = use_text_embedding
        self.text_encoder = None
        
        if use_text_embedding:
            try:
                from sentence_transformers import SentenceTransformer
                self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("Warning: sentence-transformers not installed, using simple encoding")
                self.use_text_embedding = False
    
    def extract(
        self,
        user_input: str,
        agent_state: Dict[str, Any],
        conversation_state: Optional[Dict[str, Any]] = None,
        quality_metrics: Optional[Dict[str, Any]] = None,
        intent_analysis: Optional[Dict[str, Any]] = None,
        tool_log: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        提取状态向量
        
        Args:
            user_input: 用户输入文本
            agent_state: Agent 状态（包含记忆、工具等信息）
            conversation_state: 对话状态（阶段、用户上下文等）
            quality_metrics: 质量指标
            intent_analysis: 意图分析
            tool_log: 工具调用历史
            
        Returns:
            np.ndarray: 128维状态向量
        """
        # 1. 用户上下文编码 (32维)
        user_context = self._encode_user_context(conversation_state)
        
        # 2. 对话上下文编码 (64维)
        conversation_context = self._encode_conversation_context(
            user_input, tool_log, quality_metrics, intent_analysis
        )
        
        # 3. 商品库状态编码 (32维)
        product_state = self._encode_product_state(agent_state, tool_log)
        
        # 合并
        components = StateComponents(
            user_context=user_context,
            conversation_context=conversation_context,
            product_state=product_state
        )
        
        return components.to_vector()
    
    def _encode_user_context(self, conversation_state: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        编码用户上下文 (32维)
        
        包含：
        - VIP 等级 one-hot (3维): Regular=0, VIP=1, SVIP=2
        - 购物车商品数量 (1维): 归一化到 [0,1]
        - 历史订单数 (1维): 归一化
        - 浏览商品特征 (16维): 平均价格、类别统计等
        - 对话阶段 one-hot (8维)
        - 意图历史编码 (3维): 最近3个意图的平均特征
        """
        vector = np.zeros(self.USER_CONTEXT_DIM, dtype=np.float32)
        
        if conversation_state is None:
            return vector
        
        user_ctx = conversation_state.get("user_context", {})
        
        # VIP 等级 one-hot (0-2)
        is_vip = user_ctx.get("is_vip", False)
        vip_level = 2 if is_vip else 0  # 简化：VIP=2, Regular=0
        if vip_level < 3:
            vector[vip_level] = 1.0
        
        # 购物车商品数量 (3)
        cart_count = user_ctx.get("cart_item_count", 0)
        vector[3] = min(cart_count / 10.0, 1.0)  # 归一化，假设最多10件
        
        # 历史订单数 (4)
        # 这里简化处理，实际应从数据库查询
        vector[4] = 0.0  # 占位
        
        # 浏览商品特征 (5-20): 平均价格、类别分布等
        viewed_products = user_ctx.get("last_viewed_products", [])
        if viewed_products:
            # 简化：用浏览商品数量表示活跃度
            vector[5] = min(len(viewed_products) / 5.0, 1.0)
        
        # 对话阶段 one-hot (21-28)
        stage = conversation_state.get("stage", "idle")
        stage_idx = self.STAGE_MAP.get(stage, 7)  # 默认 idle
        if 0 <= stage_idx < 8:
            vector[21 + stage_idx] = 1.0
        
        # 意图历史 (29-31): 最近意图的加权平均
        intent_history = conversation_state.get("intent_history", [])
        if intent_history:
            # 取最近3个意图
            recent_intents = intent_history[-3:]
            for i, intent in enumerate(recent_intents):
                vector[29 + i] = 0.5  # 简化：存在意图则标记
        
        return vector
    
    def _encode_conversation_context(
        self,
        user_input: str,
        tool_log: Optional[List[Dict[str, Any]]],
        quality_metrics: Optional[Dict[str, Any]],
        intent_analysis: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """
        编码对话上下文 (64维)
        
        包含：
        - 用户输入文本嵌入 (32维): BERT 嵌入或简单特征
        - 工具调用历史编码 (16维): 最近使用的工具统计
        - 质量指标向量 (8维): 效率、完成度、流畅度等
        - 意图置信度向量 (8维): 当前意图和历史意图分布
        """
        vector = np.zeros(self.CONVERSATION_CONTEXT_DIM, dtype=np.float32)
        
        # 1. 用户输入文本嵌入 (0-31)
        if self.use_text_embedding and self.text_encoder:
            try:
                embedding = self.text_encoder.encode(user_input, convert_to_numpy=True)
                # 降维到32维（取前32维或平均池化）
                if len(embedding) >= 32:
                    vector[0:32] = embedding[:32]
                else:
                    vector[0:len(embedding)] = embedding
            except Exception as e:
                print(f"Text embedding failed: {e}")
                vector[0:32] = self._simple_text_features(user_input)
        else:
            vector[0:32] = self._simple_text_features(user_input)
        
        # 2. 工具调用历史 (32-47)
        if tool_log:
            # 统计最近工具调用
            recent_tools = [entry.get("tool", "") for entry in tool_log[-5:]]
            
            # 工具类别编码
            tool_categories = {
                "search": 0, "detail": 1, "cart": 2, "order": 3,
                "payment": 4, "tracking": 5, "service": 6
            }
            
            for tool in recent_tools:
                for category, idx in tool_categories.items():
                    if category in tool.lower():
                        vector[32 + idx] = min(vector[32 + idx] + 0.2, 1.0)
            
            # 工具调用总数 (39)
            vector[39] = min(len(tool_log) / 10.0, 1.0)
        
        # 3. 质量指标 (48-55)
        if quality_metrics:
            efficiency = quality_metrics.get("efficiency", {})
            task_completion = quality_metrics.get("task_completion", {})
            conversation_quality = quality_metrics.get("conversation_quality", {})
            
            # 平均响应时间（归一化）
            avg_response_time = efficiency.get("avg_response_time", 0)
            vector[48] = max(0, 1.0 - avg_response_time / 10.0)  # 10s为基准
            
            # 平均工具调用数（归一化）
            avg_tool_calls = efficiency.get("avg_tool_calls", 0)
            vector[49] = max(0, 1.0 - avg_tool_calls / 5.0)
            
            # 任务成功率
            vector[50] = task_completion.get("success_rate", 0)
            
            # 对话流畅度
            vector[51] = 1.0 - conversation_quality.get("clarification_rate", 0)
            vector[52] = conversation_quality.get("proactive_rate", 0)
            
            # 质量分数（归一化）
            vector[53] = quality_metrics.get("quality_score", 0) / 100.0
        
        # 4. 意图置信度 (56-63)
        if intent_analysis:
            current_intent_data = intent_analysis.get("current_intent")
            intent_category = "unknown"
            confidence = 0.0

            if isinstance(current_intent_data, dict):
                confidence = current_intent_data.get("confidence", 0.0)
                intent_category = current_intent_data.get("category", "unknown")
            elif isinstance(current_intent_data, str):
                intent_category = current_intent_data
            elif current_intent_data is not None:
                # 无法识别的类型，尽量转换为字符串
                intent_category = str(current_intent_data)

            vector[56] = confidence

            intent_idx = self.INTENT_MAP.get(intent_category, 13)
            if intent_idx < 8:
                vector[57 + intent_idx] = 1.0
        
        return vector
    
    def _encode_product_state(
        self,
        agent_state: Dict[str, Any],
        tool_log: Optional[List[Dict[str, Any]]],
    ) -> np.ndarray:
        """
        编码商品库状态 (32维)
        
        包含：
        - 热门商品特征 (16维): 价格分布、类别分布
        - 库存状态统计 (8维): 缺货率、库存总量等
        - 推荐商品特征 (8维): 推荐分数、相关性等
        """
        vector = np.zeros(self.PRODUCT_STATE_DIM, dtype=np.float32)
        
        # 从工具调用结果提取商品信息
        if tool_log:
            for entry in tool_log:
                tool_name = entry.get("tool", "")
                observation = entry.get("observation", "")
                
                # 解析商品搜索结果
                if "search_products" in tool_name:
                    try:
                        result = json.loads(observation)
                        if isinstance(result, dict):
                            products = result.get("products", [])
                            if products:
                                # 平均价格（归一化）
                                prices = [p.get("price", 0) for p in products if isinstance(p, dict)]
                                if prices:
                                    avg_price = sum(prices) / len(prices)
                                    vector[0] = min(avg_price / 10000.0, 1.0)  # 假设最高1万
                                
                                # 商品数量
                                vector[1] = min(len(products) / 20.0, 1.0)
                    except Exception:
                        pass
                
                # 解析库存信息
                elif "check_stock" in tool_name:
                    try:
                        result = json.loads(observation)
                        if isinstance(result, dict):
                            stock = result.get("stock", 0)
                            vector[16] = min(stock / 100.0, 1.0)  # 归一化库存
                    except Exception:
                        pass
        
        return vector
    
    def _simple_text_features(self, text: str) -> np.ndarray:
        """
        简单文本特征（当没有 sentence-transformers 时使用）
        
        特征包括：
        - 文本长度
        - 关键词匹配
        - 问号/感叹号数量
        """
        features = np.zeros(32, dtype=np.float32)
        
        # 文本长度（归一化）
        features[0] = min(len(text) / 100.0, 1.0)
        
        # 关键词匹配（购物相关）
        keywords = {
            "搜索": 1, "查找": 1, "推荐": 2, "购买": 3, "加入": 4,
            "购物车": 4, "下单": 5, "支付": 6, "订单": 7, "物流": 8,
            "退货": 9, "客服": 10
        }
        
        for keyword, idx in keywords.items():
            if keyword in text and idx < 32:
                features[idx] = 1.0
        
        # 问号/感叹号
        features[31] = min(text.count("?") + text.count("？") + 
                          text.count("!") + text.count("！"), 1.0)
        
        return features
    
    @staticmethod
    def get_state_space_dim() -> int:
        """获取状态空间维度"""
        return StateExtractor.TOTAL_DIM
    
    @staticmethod
    def get_empty_state() -> np.ndarray:
        """获取空状态向量"""
        return np.zeros(StateExtractor.TOTAL_DIM, dtype=np.float32)
