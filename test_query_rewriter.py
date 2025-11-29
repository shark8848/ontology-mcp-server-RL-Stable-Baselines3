"""
测试查询改写器
"""
import yaml
from src.agent.query_rewriter import QueryRewriter
from src.agent.intent_tracker import Intent, IntentCategory
from src.agent.llm_deepseek import create_deepseek_chat_model

def load_config():
    """加载配置"""
    with open("src/agent/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_query_rewriter():
    """测试查询改写器"""
    print("\n" + "="*80)
    print("测试查询改写器")
    print("="*80)
    
    config = load_config()
    llm = create_deepseek_chat_model(config)
    
    rewriter_config = config.get("query_rewriter", {})
    rewriter = QueryRewriter(llm, rewriter_config)
    
    # 测试用例
    test_cases = [
        {
            "query": "有什么好的电子产品推荐？",
            "intent": Intent(
                category=IntentCategory.RECOMMENDATION,
                confidence=0.87,
                turn_id=1,
                raw_input="有什么好的电子产品推荐？"
            )
        },
        {
            "query": "2000块左右的华为手机有哪些",
            "intent": Intent(
                category=IntentCategory.SEARCH,
                confidence=0.9,
                turn_id=2,
                raw_input="2000块左右的华为手机有哪些"
            )
        },
        {
            "query": "推荐几个性价比高的笔记本",
            "intent": Intent(
                category=IntentCategory.RECOMMENDATION,
                confidence=0.85,
                turn_id=3,
                raw_input="推荐几个性价比高的笔记本"
            )
        },
        {
            "query": "有没有好用的蓝牙耳机",
            "intent": Intent(
                category=IntentCategory.SEARCH,
                confidence=0.8,
                turn_id=4,
                raw_input="有没有好用的蓝牙耳机"
            )
        },
        {
            "query": "苹果最新款的手机",
            "intent": Intent(
                category=IntentCategory.SEARCH,
                confidence=0.9,
                turn_id=5,
                raw_input="苹果最新款的手机"
            )
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"测试 {i}: {test_case['query']}")
        print(f"{'='*80}")
        
        # 执行改写
        rewritten = rewriter.rewrite(test_case['query'], test_case['intent'])
        
        # 打印结果
        print(f"\n【原始查询】")
        print(f"  {test_case['query']}")
        
        print(f"\n【系统理解】")
        print(f"  {rewritten.understood_intent}")
        
        print(f"\n【改写结果】")
        print(f"  类别: {rewritten.category}")
        print(f"  关键词: {', '.join(rewritten.keywords)}")
        
        if rewritten.expanded_keywords:
            print(f"  扩展关键词: {', '.join(rewritten.expanded_keywords[:10])}")
        
        if rewritten.brands:
            print(f"  品牌: {', '.join(rewritten.brands)}")
        
        if rewritten.price_range:
            print(f"  价格范围: {rewritten.price_range['min']}-{rewritten.price_range['max']} 元")
        
        if rewritten.user_preference:
            print(f"  用户偏好: {rewritten.user_preference}")
        
        print(f"  检索策略: {rewritten.search_strategy}")
        print(f"  置信度: {rewritten.confidence:.2f}")
        
        print(f"\n【改写原因】")
        print(f"  {rewritten.reasoning}")
        
        print(f"\n【建议检索策略】")
        search_queries = rewriter.build_search_queries(rewritten)
        for j, sq in enumerate(search_queries[:3], 1):
            print(f"  {j}. 类型={sq['type']}, 优先级={sq['priority']}")
            for k, v in sq.items():
                if k not in ['type', 'priority']:
                    print(f"     {k}={v}")
        
        print(f"\n【增强 Prompt 示例】")
        enhanced_prompt = rewriter.format_enhanced_prompt(test_case['query'], rewritten)
        print(enhanced_prompt[:500] + "..." if len(enhanced_prompt) > 500 else enhanced_prompt)

def test_synonym_expansion():
    """测试同义词扩展"""
    print("\n" + "="*80)
    print("测试同义词扩展")
    print("="*80)
    
    config = load_config()
    llm = create_deepseek_chat_model(config)
    rewriter = QueryRewriter(llm, config.get("query_rewriter", {}))
    
    test_keywords = ["电子产品", "手机", "耳机", "笔记本"]
    
    for keyword in test_keywords:
        expanded = rewriter._expand_keywords([keyword])
        print(f"\n{keyword} → {', '.join(list(expanded)[:10])}")

def test_brand_extraction():
    """测试品牌提取"""
    print("\n" + "="*80)
    print("测试品牌提取")
    print("="*80)
    
    config = load_config()
    llm = create_deepseek_chat_model(config)
    rewriter = QueryRewriter(llm, config.get("query_rewriter", {}))
    
    test_queries = [
        "华为手机",
        "苹果 iPhone",
        "小米笔记本",
        "索尼耳机",
        "三星平板"
    ]
    
    for query in test_queries:
        brands = rewriter._extract_brands(query)
        print(f"{query} → 品牌: {', '.join(brands) if brands else '(无)'}")

if __name__ == "__main__":
    print("="*80)
    print("查询改写器测试")
    print("="*80)
    
    # 测试 1: 查询改写
    test_query_rewriter()
    
    # 测试 2: 同义词扩展
    test_synonym_expansion()
    
    # 测试 3: 品牌提取
    test_brand_extraction()
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)
