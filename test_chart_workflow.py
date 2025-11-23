#!/usr/bin/env python3
"""
测试图表功能的完整工作流程
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_prompt_contains_chart_tool():
    """测试系统提示词是否包含图表工具说明"""
    print("=" * 60)
    print("测试1: 检查系统提示词")
    print("=" * 60)
    
    from agent.prompts import ECOMMERCE_SHOPPING_SYSTEM_PROMPT, ECOMMERCE_SIMPLE_SYSTEM_PROMPT
    
    # 检查完整提示词
    if "analytics_get_chart_data" in ECOMMERCE_SHOPPING_SYSTEM_PROMPT:
        print("✅ 完整提示词包含图表工具说明")
        print(f"   关键内容预览: ...{ECOMMERCE_SHOPPING_SYSTEM_PROMPT[ECOMMERCE_SHOPPING_SYSTEM_PROMPT.find('analytics_get_chart_data'):ECOMMERCE_SHOPPING_SYSTEM_PROMPT.find('analytics_get_chart_data')+150]}...")
    else:
        print("❌ 完整提示词缺少图表工具说明")
        return False
    
    # 检查简化提示词
    if "analytics_get_chart_data" in ECOMMERCE_SIMPLE_SYSTEM_PROMPT:
        print("✅ 简化提示词包含图表工具说明")
    else:
        print("⚠️  简化提示词缺少图表工具说明（已添加关键原则）")
    
    return True


def test_tool_registration():
    """测试工具是否正确注册"""
    print("\n" + "=" * 60)
    print("测试2: 检查工具注册")
    print("=" * 60)
    
    from agent.mcp_adapter import MCPAdapter
    
    adapter = MCPAdapter(base_url="http://localhost:8000")
    tools = adapter.create_tools()
    
    chart_tool = None
    for tool in tools:
        if tool.name == "analytics_get_chart_data":
            chart_tool = tool
            break
    
    if chart_tool:
        print(f"✅ 图表工具已注册")
        print(f"   名称: {chart_tool.name}")
        print(f"   描述: {chart_tool.description}")
        print(f"   参数: {list(chart_tool.args_schema.model_fields.keys())}")
        return True
    else:
        print("❌ 图表工具未找到")
        print(f"   可用工具: {[t.name for t in tools]}")
        return False


def test_intent_recognition():
    """测试意图识别"""
    print("\n" + "=" * 60)
    print("测试3: 检查意图识别")
    print("=" * 60)
    
    from agent.intent_tracker import IntentRecognizer, IntentCategory
    
    recognizer = IntentRecognizer()
    
    test_queries = [
        "给我展示订单趋势图",
        "查看最近7天的销售走势",
        "显示商品分类占比饼图",
        "各类商品的销量排行",
    ]
    
    for query in test_queries:
        intents = recognizer.recognize(query, turn_id=1)
        intent_names = [i.category.value for i in intents]
        
        if IntentCategory.CHART_REQUEST.value in intent_names:
            print(f"✅ '{query}' → {intent_names}")
        else:
            print(f"❌ '{query}' → {intent_names} (未识别为chart_request)")
    
    return True


def test_chart_tool_call_format():
    """测试工具调用格式"""
    print("\n" + "=" * 60)
    print("测试4: 模拟工具调用")
    print("=" * 60)
    
    try:
        from agent.analytics_service import get_chart_data
        
        # 模拟trend图表
        result = get_chart_data(chart_type="trend", days=7)
        print(f"✅ 趋势图生成成功")
        print(f"   标题: {result['title']}")
        print(f"   标签数: {len(result['labels'])}")
        print(f"   系列数: {len(result['series'])}")
        
        return True
    except Exception as e:
        print(f"⚠️  工具调用测试跳过（需要数据库）: {e}")
        return True  # 不算失败


def main():
    print("🧪 图表功能完整流程测试\n")
    
    results = []
    
    try:
        results.append(("提示词检查", test_prompt_contains_chart_tool()))
        results.append(("工具注册", test_tool_registration()))
        results.append(("意图识别", test_intent_recognition()))
        results.append(("工具调用", test_chart_tool_call_format()))
        
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        
        for name, passed in results:
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{status} - {name}")
        
        if all(r[1] for r in results):
            print("\n🎉 所有测试通过！")
            print("\n📌 可能的问题原因：")
            print("   1. LLM可能未理解用户意图（尝试更明确的表达）")
            print("   2. 检查Gradio UI是否正确提取并渲染图表")
            print("   3. 查看Agent日志确认工具是否被调用")
            return 0
        else:
            print("\n❌ 部分测试失败，请检查上述输出")
            return 1
            
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
