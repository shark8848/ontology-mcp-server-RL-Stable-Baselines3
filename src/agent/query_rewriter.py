"""
Copyright (c) 2025 shark8848
MIT License

Ontology MCP Server - 电商 AI 助手系统
本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI

Author: shark8848
Repository: https://github.com/shark8848/ontology-mcp-server
"""

"""
查询改写器 - Query Rewriter

功能:
1. 理解用户模糊查询的真实意图
2. 扩展关键词和同义词
3. 优化检索策略
4. 提升商品推荐准确率
"""

import json
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from functools import lru_cache

from .intent_tracker import Intent, IntentCategory

logger = logging.getLogger(__name__)


@dataclass
class RewrittenQuery:
    """改写后的查询"""
    original_query: str
    understood_intent: str  # LLM 理解的用户意图
    category: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    expanded_keywords: List[str] = field(default_factory=list)
    user_preference: Optional[str] = None  # 用户偏好: 性价比高、品质优、热销等
    price_range: Optional[Dict[str, float]] = None  # {"min": 100, "max": 5000}
    brands: List[str] = field(default_factory=list)
    search_strategy: str = "broad"  # broad, specific, hybrid
    confidence: float = 0.8
    reasoning: str = ""  # 改写原因


class QueryRewriter:
    """查询改写器 - 使用 LLM 优化用户查询"""
    
    # 同义词映射表
    SYNONYM_MAP = {
        "电子产品": ["手机", "笔记本电脑", "平板电脑", "耳机", "智能手表", "数码产品"],
        "手机": ["智能手机", "iPhone", "安卓手机", "移动电话"],
        "笔记本": ["笔记本电脑", "laptop", "便携电脑"],
        "耳机": ["蓝牙耳机", "无线耳机", "头戴式耳机", "入耳式耳机"],
        "家电": ["电器", "家用电器", "厨房电器", "生活电器"],
        "服装": ["衣服", "服饰", "穿搭", "时装"],
        "食品": ["食物", "零食", "美食", "小吃"],
        "图书": ["书籍", "读物", "书本"],
        "运动": ["健身", "户外", "体育用品"],
        "美妆": ["化妆品", "护肤品", "彩妆"],
        "玩具": ["儿童玩具", "益智玩具", "游戏玩具"],
    }
    
    # 品牌关键词
    BRAND_KEYWORDS = {
        "苹果", "Apple", "iPhone", "iPad", "MacBook",
        "华为", "Huawei", "小米", "Xiaomi", "OPPO", "vivo",
        "三星", "Samsung", "联想", "Lenovo", "戴尔", "Dell",
        "惠普", "HP", "索尼", "Sony", "佳能", "Canon",
    }
    
    # 偏好关键词
    PREFERENCE_KEYWORDS = {
        "性价比": ["实惠", "便宜", "划算", "省钱"],
        "品质": ["高端", "高档", "品质好", "质量好"],
        "热销": ["畅销", "爆款", "人气", "流行"],
        "新品": ["最新", "新款", "刚上市"],
        "轻薄": ["便携", "轻巧", "小巧"],
        "续航": ["电池", "待机", "耐用"],
    }
    
    def __init__(self, llm, config: Dict[str, Any] = None):
        self.llm = llm
        self.config = config or {}
        self.enable_cache = self.config.get("enable_cache", True)
        self.enable_synonym_expansion = self.config.get("enable_synonym_expansion", True)
        
        logger.info("查询改写器初始化完成")
    
    @lru_cache(maxsize=500)
    def _cached_llm_rewrite(self, query: str, intent_str: str) -> str:
        """缓存的 LLM 改写调用"""
        prompt = self._build_rewrite_prompt(query, intent_str)
        try:
            response = self.llm.predict(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM 查询改写失败: {e}")
            return self._fallback_rewrite(query)
    
    def _build_rewrite_prompt(self, query: str, intent_str: str) -> str:
        """构建改写 prompt"""
        return f"""你是一个电商查询优化专家。请分析用户查询并优化搜索策略。

用户查询: {query}
识别意图: {intent_str}

请返回 JSON 格式 (严格遵守格式):
{{
  "understood_intent": "用户真实意图的自然语言描述",
  "category": "商品类别(如: 电子产品, 服装, 食品)",
  "keywords": ["主要关键词1", "主要关键词2"],
  "user_preference": "用户偏好(如: 性价比高, 品质优, 热销, 新品, 轻薄, 续航长)",
  "price_range": {{"min": 最低价, "max": 最高价}} 或 null,
  "brands": ["品牌1", "品牌2"] 或 [],
  "search_strategy": "broad/specific/hybrid",
  "confidence": 0.0-1.0,
  "reasoning": "改写原因"
}}

示例:
输入: "有什么好的电子产品推荐？"
输出:
{{
  "understood_intent": "用户想要浏览热销、高性价比的电子产品,没有明确具体类别",
  "category": "电子产品",
  "keywords": ["手机", "笔记本电脑", "平板电脑", "耳机", "智能手表"],
  "user_preference": "热销、性价比高",
  "price_range": null,
  "brands": [],
  "search_strategy": "broad",
  "confidence": 0.9,
  "reasoning": "用户使用'好的'表示关注品质和口碑,'电子产品'范围较广需要扩展具体品类"
}}

输入: "2000块左右的华为手机有哪些"
输出:
{{
  "understood_intent": "用户想购买华为品牌的手机,预算约2000元",
  "category": "手机",
  "keywords": ["华为手机", "智能手机"],
  "user_preference": "性价比",
  "price_range": {{"min": 1500, "max": 2500}},
  "brands": ["华为", "Huawei"],
  "search_strategy": "specific",
  "confidence": 0.95,
  "reasoning": "用户明确了品牌、价格和品类,可以精准检索"
}}

现在请处理上面的用户查询。只返回 JSON,不要其他内容。
"""
    
    def _fallback_rewrite(self, query: str) -> str:
        """降级改写策略 (LLM 失败时)"""
        return json.dumps({
            "understood_intent": f"用户查询: {query}",
            "category": "电子产品",
            "keywords": [query],
            "user_preference": None,
            "price_range": None,
            "brands": [],
            "search_strategy": "broad",
            "confidence": 0.5,
            "reasoning": "LLM改写失败,使用原始查询"
        }, ensure_ascii=False)
    
    def rewrite(self, query: str, intent: Intent) -> RewrittenQuery:
        """
        改写查询
        
        Args:
            query: 用户原始查询
            intent: 识别出的意图
        
        Returns:
            改写后的查询对象
        """
        intent_str = f"{intent.category.value} (置信度: {intent.confidence:.2f})"
        
        # 1. LLM 改写
        if self.enable_cache:
            llm_response = self._cached_llm_rewrite(query, intent_str)
        else:
            llm_response = self._cached_llm_rewrite.__wrapped__(self, query, intent_str)
        
        # 2. 解析 LLM 响应
        try:
            llm_response = llm_response.strip()
            if llm_response.startswith("```json"):
                llm_response = llm_response[7:]
            if llm_response.startswith("```"):
                llm_response = llm_response[3:]
            if llm_response.endswith("```"):
                llm_response = llm_response[:-3]
            llm_response = llm_response.strip()
            
            parsed = json.loads(llm_response)
        except json.JSONDecodeError as e:
            logger.error(f"LLM 响应 JSON 解析失败: {e}, 响应: {llm_response[:200]}")
            parsed = json.loads(self._fallback_rewrite(query))
        
        # 3. 提取品牌 (从原始查询和 LLM 结果合并)
        brands = set(parsed.get("brands", []))
        brands.update(self._extract_brands(query))
        
        # 4. 扩展同义词
        keywords = parsed.get("keywords", [])
        expanded_keywords = self._expand_keywords(keywords) if self.enable_synonym_expansion else keywords
        
        # 5. 构建改写结果
        rewritten = RewrittenQuery(
            original_query=query,
            understood_intent=parsed.get("understood_intent", ""),
            category=parsed.get("category"),
            keywords=keywords,
            expanded_keywords=list(expanded_keywords),
            user_preference=parsed.get("user_preference"),
            price_range=parsed.get("price_range"),
            brands=list(brands),
            search_strategy=parsed.get("search_strategy", "broad"),
            confidence=parsed.get("confidence", 0.8),
            reasoning=parsed.get("reasoning", "")
        )
        
        logger.info(
            f"查询改写完成: '{query}' → "
            f"类别={rewritten.category}, "
            f"关键词={rewritten.keywords[:3]}, "
            f"策略={rewritten.search_strategy}"
        )
        
        return rewritten
    
    def _extract_brands(self, query: str) -> Set[str]:
        """从查询中提取品牌"""
        brands = set()
        query_lower = query.lower()
        for brand in self.BRAND_KEYWORDS:
            if brand.lower() in query_lower:
                brands.add(brand)
        return brands
    
    def _expand_keywords(self, keywords: List[str]) -> Set[str]:
        """扩展关键词同义词"""
        expanded = set(keywords)
        for keyword in keywords:
            # 查找同义词
            if keyword in self.SYNONYM_MAP:
                expanded.update(self.SYNONYM_MAP[keyword])
            
            # 查找部分匹配
            for key, synonyms in self.SYNONYM_MAP.items():
                if keyword in key or key in keyword:
                    expanded.update(synonyms)
        
        return expanded
    
    def build_search_queries(self, rewritten: RewrittenQuery) -> List[Dict[str, Any]]:
        """
        根据改写结果构建多个检索查询
        
        Returns:
            检索查询列表,按优先级排序
        """
        queries = []
        
        strategy = rewritten.search_strategy
        
        if strategy == "specific":
            # 精确检索: 品牌 + 类别 + 价格
            if rewritten.brands:
                for brand in rewritten.brands:
                    queries.append({
                        "type": "specific",
                        "keyword": f"{brand} {rewritten.category or ''}".strip(),
                        "brand": brand,
                        "category": rewritten.category,
                        "min_price": rewritten.price_range.get("min") if rewritten.price_range else None,
                        "max_price": rewritten.price_range.get("max") if rewritten.price_range else None,
                        "priority": 1
                    })
            else:
                queries.append({
                    "type": "specific",
                    "category": rewritten.category,
                    "min_price": rewritten.price_range.get("min") if rewritten.price_range else None,
                    "max_price": rewritten.price_range.get("max") if rewritten.price_range else None,
                    "priority": 1
                })
        
        elif strategy == "broad":
            # 广泛检索: 扩展关键词
            for keyword in rewritten.expanded_keywords[:5]:  # 最多5个关键词
                queries.append({
                    "type": "keyword",
                    "keyword": keyword,
                    "category": rewritten.category,
                    "priority": 2
                })
            
            # 类别推荐
            if rewritten.category:
                queries.append({
                    "type": "recommendation",
                    "category": rewritten.category,
                    "limit": 10,
                    "priority": 3
                })
        
        else:  # hybrid
            # 混合策略: 先精确,再广泛
            if rewritten.brands:
                queries.append({
                    "type": "specific",
                    "brand": rewritten.brands[0],
                    "category": rewritten.category,
                    "priority": 1
                })
            
            for keyword in rewritten.keywords[:3]:
                queries.append({
                    "type": "keyword",
                    "keyword": keyword,
                    "category": rewritten.category,
                    "priority": 2
                })
            
            if rewritten.category:
                queries.append({
                    "type": "recommendation",
                    "category": rewritten.category,
                    "priority": 3
                })
        
        # 按优先级排序
        queries.sort(key=lambda x: x["priority"])
        
        return queries
    
    def format_enhanced_prompt(self, original_query: str, rewritten: RewrittenQuery) -> str:
        """
        格式化增强的 prompt
        
        用于注入到 Agent 的上下文中,引导 LLM 更好地调用工具
        """
        enhanced = f"""用户查询: {original_query}

【系统理解】
{rewritten.understood_intent}

【推荐检索策略】
- 商品类别: {rewritten.category or '不限'}
- 关键词: {', '.join(rewritten.keywords[:5])}
"""
        
        if rewritten.expanded_keywords:
            enhanced += f"- 扩展关键词: {', '.join(rewritten.expanded_keywords[:10])}\n"
        
        if rewritten.brands:
            enhanced += f"- 指定品牌: {', '.join(rewritten.brands)}\n"
        
        if rewritten.price_range:
            enhanced += f"- 价格范围: {rewritten.price_range['min']}~{rewritten.price_range['max']} 元\n"
        
        if rewritten.user_preference:
            enhanced += f"- 用户偏好: {rewritten.user_preference}\n"
        
        enhanced += f"- 检索策略: {rewritten.search_strategy}\n"
        
        enhanced += f"\n【建议工具调用】\n"
        search_queries = self.build_search_queries(rewritten)
        for i, sq in enumerate(search_queries[:3], 1):
            if sq["type"] == "specific":
                enhanced += f"{i}. commerce.search_products("
                params = []
                if sq.get("keyword"):
                    params.append(f'keyword="{sq["keyword"]}"')
                if sq.get("brand"):
                    params.append(f'brand="{sq["brand"]}"')
                if sq.get("category"):
                    params.append(f'category="{sq["category"]}"')
                if sq.get("min_price"):
                    params.append(f'min_price={sq["min_price"]}')
                if sq.get("max_price"):
                    params.append(f'max_price={sq["max_price"]}')
                enhanced += ", ".join(params) + ")\n"
            
            elif sq["type"] == "keyword":
                enhanced += f'{i}. commerce.search_products(keyword="{sq["keyword"]}"'
                if sq.get("category"):
                    enhanced += f', category="{sq["category"]}"'
                enhanced += ")\n"
            
            elif sq["type"] == "recommendation":
                enhanced += f'{i}. commerce.get_product_recommendations(category="{sq["category"]}", limit={sq.get("limit", 10)})\n'
        
        enhanced += f"\n请根据以上分析,调用合适的工具返回推荐结果。"
        
        return enhanced
