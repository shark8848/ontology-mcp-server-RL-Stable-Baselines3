#!/usr/bin/env python3
"""
测试LLM是否真的会调用图表工具
直接与Agent交互，查看实际tool calls
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_agent_tool_call():
    """直接测试Agent的工具调用行为"""
    print("=" * 70)
    print("🔬 测试Agent图表工具调用")
    print("=" * 70)
    
    from agent.react_agent import LangChainAgent
    
    # 创建Agent实例
    agent = LangChainAgent(
        enable_system_prompt=True,
        enable_conversation_state=False,
        enable_quality_tracking=False,
        enable_intent_tracking=False,
        enable_recommendation=False,
    )
    
    # 测试查询
    test_queries = [
        "显示销量前10的商品柱状图",
        "展示最近7天的订单趋势图",
        "生成商品分类销量饼图",
        "比较不同用户的消费对比图",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"测试 {i}/{len(test_queries)}: {query}")
        print("=" * 70)
        
        try:
            # 运行Agent
            result = agent.run(query)
            
            # 分析结果
            print("\n📊 Agent运行结果:")
            print(f"  - 回复长度: {len(result.get('response', ''))} 字符")
            print(f"  - 包含图表: {'charts' in result and result['charts']}")
            
            if "charts" in result and result["charts"]:
                print(f"  - 图表数量: {len(result['charts'])}")
                for chart in result["charts"]:
                    print(f"    * {chart.get('title', 'Untitled')}")
                print("\n✅ 成功生成图表！")
            else:
                print("\n❌ 未生成图表")
                
                # 显示Agent的回复
                response = result.get("response", "")
                if response:
                    print(f"\n  Agent回复: {response[:200]}...")
                
            # 检查tool_log
            if "tool_log" in result:
                print(f"\n🔧 工具调用记录: {len(result['tool_log'])} 条")
                for tool_entry in result["tool_log"]:
                    tool_name = tool_entry.get("tool_name", "unknown")
                    print(f"  - {tool_name}")
                    if tool_name == "analytics_get_chart_data":
                        print("    ✅ 调用了图表工具!")
                        
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print()


def test_direct_llm_call():
    """直接测试LLM的原始响应"""
    print("\n" + "=" * 70)
    print("🔬 直接测试LLM的tool calling行为")
    print("=" * 70)
    
    from agent.llm_deepseek import get_deepseek_chat_model
    from agent.mcp_adapter import MCPAdapter
    
    # 获取LLM
    llm = get_deepseek_chat_model()
    
    # 获取工具
    adapter = MCPAdapter()
    tools = adapter.create_tools()
    openai_tools = [t.to_openai_tool() for t in tools]
    
    # 构建消息
    system_prompt = """你是电商助手。当用户要求生成图表时，必须调用 analytics_get_chart_data 工具。

关键词映射：
- 柱状图、排行 → chart_type="bar"
- 趋势图、走势 → chart_type="trend"
- 饼图、占比 → chart_type="pie"
- 对比、比较 → chart_type="comparison"

示例：用户说"显示销量前10的商品柱状图"，你应该调用：
analytics_get_chart_data(chart_type="bar", top_n=10)"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "显示销量前10的商品柱状图"}
    ]
    
    print("\n📤 发送给LLM:")
    print(f"  - System prompt: {len(system_prompt)} 字符")
    print(f"  - User message: {messages[1]['content']}")
    print(f"  - Tools: {len(openai_tools)} 个")
    print(f"  - 包含图表工具: {any(t['function']['name'] == 'analytics_get_chart_data' for t in openai_tools)}")
    
    try:
        # 调用LLM
        from langchain_core.messages import SystemMessage, HumanMessage
        lc_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="显示销量前10的商品柱状图")
        ]
        
        # bind tools
        llm_with_tools = llm.bind(functions=openai_tools)
        
        print("\n⏳ 调用LLM...")
        response = llm_with_tools.invoke(lc_messages)
        
        print("\n📥 LLM响应:")
        print(f"  - Content: {response.content[:200] if response.content else '(empty)'}...")
        
        # 检查tool calls
        if hasattr(response, 'additional_kwargs'):
            kwargs = response.additional_kwargs
            if 'function_call' in kwargs:
                print("\n✅ LLM请求调用工具:")
                print(f"  - 函数名: {kwargs['function_call'].get('name')}")
                print(f"  - 参数: {kwargs['function_call'].get('arguments')}")
            elif 'tool_calls' in kwargs:
                print("\n✅ LLM请求调用工具:")
                for tool_call in kwargs['tool_calls']:
                    print(f"  - 函数名: {tool_call['function']['name']}")
                    print(f"  - 参数: {tool_call['function']['arguments']}")
            else:
                print("\n❌ LLM没有调用任何工具")
                print(f"  - additional_kwargs: {kwargs}")
        else:
            print("\n⚠️  响应中没有additional_kwargs字段")
            print(f"  - 响应类型: {type(response)}")
            print(f"  - 响应属性: {dir(response)}")
        
    except Exception as e:
        print(f"\n❌ 直接调用LLM失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "=" * 70)
    print("🔬 图表工具调用测试")
    print("=" * 70)
    print("\n这个脚本将：")
    print("  1. 测试Agent是否调用图表工具")
    print("  2. 直接测试LLM的tool calling行为\n")
    
    # 第一部分：测试Agent
    print("\n第一部分：测试完整Agent")
    print("-" * 70)
    test_agent_tool_call()
    
    # 第二部分：直接测试LLM
    print("\n第二部分：直接测试LLM")
    print("-" * 70)
    test_direct_llm_call()
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
