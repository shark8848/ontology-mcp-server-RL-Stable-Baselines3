#!/usr/bin/env python3
"""
最小化测试：LLM是否会调用图表工具
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_llm_tool_call():
    """测试LLM是否真的会调用analytics_get_chart_data"""
    print("=" * 70)
    print("🔬 测试LLM图表工具调用")
    print("=" * 70)
    
    # 导入
    from agent.llm_deepseek import get_default_chat_model
    from agent.mcp_adapter import MCPAdapter
    
    # 初始化
    llm = get_default_chat_model()
    adapter = MCPAdapter()
    tools = adapter.create_tools()
    tool_specs = [t.to_openai_tool() for t in tools]
    
    print(f"\n✅ LLM 模型: {llm.model}")
    print(f"✅ API URL: {llm.client.base_url}")
    print(f"✅ 工具数量: {len(tool_specs)}")
    
    # 确认图表工具存在
    chart_tool = None
    for spec in tool_specs:
        if spec["function"]["name"] == "analytics_get_chart_data":
            chart_tool = spec
            break
    
    if not chart_tool:
        print("❌ 图表工具不存在!")
        return False
    
    print("\n✅ 图表工具定义:")
    print(json.dumps(chart_tool, indent=2, ensure_ascii=False))
    
    # 构建消息
    system_prompt = """你是电商助手。你有以下工具：

- analytics_get_chart_data: 生成数据可视化图表（趋势图、柱状图、饼图、对比图）
  参数: chart_type（trend/pie/bar/comparison）、days（天数）、top_n（排名数量）

**重要规则**：
- 用户说"柱状图"、"排行" → 调用 analytics_get_chart_data(chart_type="bar")
- 用户说"趋势图"、"走势" → 调用 analytics_get_chart_data(chart_type="trend")
- 用户说"饼图"、"占比" → 调用 analytics_get_chart_data(chart_type="pie")
- 用户说"对比"、"比较" → 调用 analytics_get_chart_data(chart_type="comparison")

示例：
用户："显示销量前10的商品柱状图"
你应该：analytics_get_chart_data(chart_type="bar", top_n=10)

用户："展示最近7天订单趋势"
你应该：analytics_get_chart_data(chart_type="trend", days=7)

**必须调用工具，不要只用文字描述！**"""

    user_message = "显示销量前10的商品柱状图"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    print("\n" + "=" * 70)
    print("📤 发送给LLM")
    print("=" * 70)
    print(f"\n用户消息: {user_message}")
    print(f"系统提示长度: {len(system_prompt)} 字符")
    print(f"工具: {len(tool_specs)} 个")
    
    # 调用LLM
    print("\n⏳ 调用LLM...")
    try:
        result = llm.generate(messages, tools=tool_specs)
    except Exception as e:
        print(f"\n❌ LLM调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 分析响应
    print("\n" + "=" * 70)
    print("📥 LLM响应")
    print("=" * 70)
    
    content = result.get("content", "")
    tool_calls = result.get("tool_calls", [])
    
    print(f"\n内容长度: {len(content)} 字符")
    if content:
        print(f"内容预览: {content[:200]}...")
    
    print(f"\n工具调用: {len(tool_calls)} 个")
    
    if tool_calls:
        print("\n✅ LLM调用了工具!")
        for i, call in enumerate(tool_calls, 1):
            print(f"\n  工具 {i}:")
            print(f"    ID: {call.get('id')}")
            print(f"    名称: {call.get('name')}")
            print(f"    参数: {json.dumps(call.get('arguments', {}), ensure_ascii=False)}")
            
            # 检查是否是图表工具
            if call.get('name') == 'analytics_get_chart_data':
                args = call.get('arguments', {})
                chart_type = args.get('chart_type')
                print(f"\n    🎉 调用了图表工具!")
                print(f"    图表类型: {chart_type}")
                
                # 检查参数正确性
                if chart_type == 'bar':
                    print("    ✅ 图表类型正确 (bar)")
                else:
                    print(f"    ⚠️  图表类型可能不正确: {chart_type} (期望 bar)")
                
                top_n = args.get('top_n')
                if top_n:
                    print(f"    ✅ top_n参数: {top_n}")
                else:
                    print("    ⚠️  缺少top_n参数")
                
                return True
        
        print("\n⚠️  LLM调用了工具，但不是图表工具")
        return False
    else:
        print("\n❌ LLM没有调用任何工具!")
        print(f"\nLLM只返回了文字:\n{content}")
        return False


def test_multiple_queries():
    """测试多个查询"""
    queries = [
        "显示销量前10的商品柱状图",
        "展示最近7天订单趋势图",
        "生成商品分类销量饼图",
    ]
    
    results = {}
    
    for query in queries:
        print(f"\n\n{'=' * 70}")
        print(f"测试: {query}")
        print("=" * 70)
        
        result = test_specific_query(query)
        results[query] = result
        
        if result:
            print(f"\n✅ 成功")
        else:
            print(f"\n❌ 失败")
    
    print("\n\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    
    for query, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {query}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n成功率: {success_count}/{total_count} ({success_count/total_count*100:.0f}%)")
    
    return success_count == total_count


def test_specific_query(query: str) -> bool:
    """测试特定查询"""
    from agent.llm_deepseek import get_default_chat_model
    from agent.mcp_adapter import MCPAdapter
    
    llm = get_default_chat_model()
    adapter = MCPAdapter()
    tools = adapter.create_tools()
    tool_specs = [t.to_openai_tool() for t in tools]
    
    system_prompt = """你是电商助手。用户要求图表时，必须调用 analytics_get_chart_data 工具。

关键词：柱状图、排行 → chart_type="bar"
关键词：趋势图、走势 → chart_type="trend"
关键词：饼图、占比 → chart_type="pie"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    try:
        result = llm.generate(messages, tools=tool_specs)
        tool_calls = result.get("tool_calls", [])
        
        for call in tool_calls:
            if call.get('name') == 'analytics_get_chart_data':
                return True
        
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False


def main():
    print("\n🔬 图表工具调用测试\n")
    
    # 先测试单个查询
    success = test_llm_tool_call()
    
    if success:
        print("\n\n✅ 基础测试通过！LLM会调用图表工具。")
        print("\n继续测试多个查询...")
        test_multiple_queries()
    else:
        print("\n\n❌ 基础测试失败！LLM不调用图表工具。")
        print("\n可能的原因:")
        print("  1. System prompt 不够明确")
        print("  2. LLM 模型本身的行为问题")
        print("  3. 工具定义描述不够清楚")
        print("  4. LLM temperature 过高导致不稳定")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
