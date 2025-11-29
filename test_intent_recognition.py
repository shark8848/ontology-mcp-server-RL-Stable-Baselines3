"""
测试新的配置化意图识别系统
"""
import yaml
from src.agent.intent_tracker import (
    HybridIntentRecognizer,
    RuleBasedRecognizer,
    IntentCategory
)
from src.agent.llm_deepseek import create_deepseek_chat_model

def load_config():
    """加载配置"""
    with open("src/agent/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_rule_based():
    """测试规则匹配"""
    print("\n" + "="*60)
    print("测试 1: 规则匹配识别器")
    print("="*60)
    
    config = load_config()
    rule_config = config.get("intent_recognition", {}).get("rule", {})
    recognizer = RuleBasedRecognizer(rule_config)
    
    test_cases = [
        "你们有什么好的电子产品推荐？",
        "我的购物车里有什么",
        "搜索笔记本电脑",
        "这个多少钱",
        "加入购物车",
    ]
    
    for user_input in test_cases:
        intents = recognizer.recognize(user_input)
        intent = intents[0]
        print(f"输入: {user_input}")
        print(f"识别结果: {intent.category.value} (置信度: {intent.confidence:.2f})")
        print()

def test_llm_based():
    """测试 LLM 识别"""
    print("\n" + "="*60)
    print("测试 2: LLM 识别器")
    print("="*60)
    
    config = load_config()
    llm = create_deepseek_chat_model(config)
    
    from src.agent.intent_tracker import LLMIntentRecognizer
    llm_config = config.get("intent_recognition", {}).get("llm", {})
    recognizer = LLMIntentRecognizer(llm, llm_config)
    
    test_cases = [
        "你们有什么好的电子产品推荐？",
        "我想看看购物车",
        "有没有性价比高的手机",
        "帮我推荐个笔记本",
    ]
    
    for user_input in test_cases:
        intents = recognizer.recognize(user_input)
        intent = intents[0]
        print(f"输入: {user_input}")
        print(f"识别结果: {intent.category.value} (置信度: {intent.confidence:.2f})")
        if intent.extracted_entities:
            print(f"提取实体: {intent.extracted_entities}")
        print()

def test_hybrid():
    """测试混合识别器"""
    print("\n" + "="*60)
    print("测试 3: 混合识别器 (按优先级)")
    print("="*60)
    
    config = load_config()
    llm = create_deepseek_chat_model(config)
    
    intent_config = config.get("intent_recognition", {})
    print(f"优先级顺序: {intent_config.get('priority', [])}")
    print(f"高置信度阈值: {intent_config.get('high_confidence_threshold', 0.85)}")
    print()
    
    recognizer = HybridIntentRecognizer(llm, intent_config)
    
    test_cases = [
        "你们有什么好的电子产品推荐？",  # 应该被LLM正确识别
        "我的购物车里有什么",           # 规则即可识别
        "有没有好用的智能手表",         # 需要LLM理解
        "这款产品库存还有吗",           # 规则可识别
        "推荐几个性价比高的产品",       # LLM更准确
    ]
    
    for user_input in test_cases:
        print(f"\n输入: {user_input}")
        intents = recognizer.recognize(user_input)
        intent = intents[0]
        print(f"最终结果: {intent.category.value} (置信度: {intent.confidence:.2f})")
        if intent.extracted_entities:
            print(f"提取实体: {intent.extracted_entities}")

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "="*60)
    print("测试 4: 边界情况")
    print("="*60)
    
    config = load_config()
    llm = create_deepseek_chat_model(config)
    recognizer = HybridIntentRecognizer(llm, config.get("intent_recognition", {}))
    
    test_cases = [
        "asdfghjkl",                    # 无意义输入
        "",                              # 空字符串
        "help me buy something good",    # 英文
        "我想要一个既便宜又好用的，还要配送快的，最好还有售后服务的产品",  # 复杂需求
    ]
    
    for user_input in test_cases:
        if not user_input:
            user_input = "(空字符串)"
        print(f"\n输入: {user_input}")
        intents = recognizer.recognize(user_input if user_input != "(空字符串)" else "")
        intent = intents[0]
        print(f"识别结果: {intent.category.value} (置信度: {intent.confidence:.2f})")

if __name__ == "__main__":
    print("开始测试配置化意图识别系统...")
    
    # 测试 1: 规则匹配
    test_rule_based()
    
    # 测试 2: LLM 识别
    test_llm_based()
    
    # 测试 3: 混合识别
    test_hybrid()
    
    # 测试 4: 边界情况
    test_edge_cases()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
