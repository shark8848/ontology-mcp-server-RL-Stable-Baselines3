#!/usr/bin/env python3
"""
对比实际Agent使用的prompt和测试prompt
找出为什么实际环境中不调用工具
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def compare_prompts():
    """对比两种prompt"""
    print("=" * 70)
    print("🔍 对比System Prompt")
    print("=" * 70)
    
    from agent.react_agent import LangChainAgent
    from agent.prompts import ECOMMERCE_SHOPPING_SYSTEM_PROMPT
    
    # 创建Agent获取实际使用的prompt
    agent = LangChainAgent(
        enable_system_prompt=True,
        enable_conversation_state=False,
        enable_quality_tracking=False,
        enable_intent_tracking=False,
        enable_recommendation=False,
    )
    
    actual_prompt = agent.prompt_manager.get_system_prompt() if agent.prompt_manager else ""
    
    print("\n1. 实际Agent使用的Prompt:")
    print("-" * 70)
    print(f"长度: {len(actual_prompt)} 字符")
    
    # 检查关键内容
    checks = {
        "analytics_get_chart_data": "图表工具名称",
        "必须调用": "强制规则",
        "chart_type": "参数说明",
        "trend": "趋势图",
        "bar": "柱状图",
        "pie": "饼图",
        "comparison": "对比图",
    }
    
    print("\n关键词检查:")
    for keyword, desc in checks.items():
        count = actual_prompt.count(keyword)
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {desc} ({keyword}): {count} 次")
    
    # 找出图表相关段落
    print("\n图表相关段落:")
    print("-" * 70)
    
    lines = actual_prompt.split("\n")
    chart_section_start = -1
    chart_section_end = -1
    
    for i, line in enumerate(lines):
        if "analytics_get_chart_data" in line and chart_section_start == -1:
            chart_section_start = max(0, i - 2)
        if chart_section_start != -1 and chart_section_end == -1:
            # 找到下一个大标题或结束
            if i > chart_section_start + 2 and (line.startswith("#") or not line.strip()):
                if i - chart_section_start > 5:
                    chart_section_end = i
                    break
    
    if chart_section_start != -1:
        if chart_section_end == -1:
            chart_section_end = min(len(lines), chart_section_start + 15)
        chart_section = "\n".join(lines[chart_section_start:chart_section_end])
        print(chart_section)
    else:
        print("⚠️  未找到图表相关段落")
    
    print("\n" + "=" * 70)
    
    # 测试用的简化prompt
    test_prompt = """你是电商助手。用户要求图表时，必须调用 analytics_get_chart_data 工具。

关键词：柱状图、排行 → chart_type="bar"
关键词：趋势图、走势 → chart_type="trend"
关键词：饼图、占比 → chart_type="pie"

示例：用户说"显示销量前10的商品柱状图"，你应该调用 analytics_get_chart_data(chart_type="bar", top_n=10)

**必须调用工具，不要只用文字描述！**"""

    print("\n2. 测试使用的简化Prompt:")
    print("-" * 70)
    print(test_prompt)
    print(f"\n长度: {len(test_prompt)} 字符")
    
    # 对比
    print("\n" + "=" * 70)
    print("📊 对比分析")
    print("=" * 70)
    
    print(f"\n长度差异:")
    print(f"  实际: {len(actual_prompt)} 字符")
    print(f"  测试: {len(test_prompt)} 字符")
    print(f"  差异: {len(actual_prompt) - len(test_prompt)} 字符 ({len(actual_prompt)/len(test_prompt):.1f}x)")
    
    # 关键区别
    print("\n可能的问题:")
    
    # 1. Prompt太长
    if len(actual_prompt) > 3000:
        print("  ⚠️  Prompt可能太长 (>3000字符)，LLM可能忽略后半部分")
    
    # 2. 图表工具位置
    chart_tool_pos = actual_prompt.find("analytics_get_chart_data")
    if chart_tool_pos > len(actual_prompt) * 0.7:
        print(f"  ⚠️  图表工具描述在Prompt的后70%位置 (第{chart_tool_pos}字符)")
        print("     LLM可能更关注前面的内容")
    
    # 3. 其他工具太多
    tool_count = actual_prompt.count("(") - actual_prompt.count("(用户")
    if tool_count > 15:
        print(f"  ⚠️  Prompt中提到太多工具/函数 (~{tool_count}个)")
        print("     可能分散LLM注意力")
    
    # 4. 缺少示例
    if "显示销量" not in actual_prompt and "柱状图" not in actual_prompt:
        print("  ⚠️  缺少具体的图表使用示例")
    
    # 5. 规则不够明确
    if actual_prompt.count("必须调用") < 2:
        print("  ⚠️  '必须调用'强调不够 (只有1次)")
    
    return actual_prompt, test_prompt


def test_with_actual_prompt():
    """使用实际Agent的prompt测试LLM"""
    print("\n" + "=" * 70)
    print("🧪 使用实际Agent Prompt测试LLM")
    print("=" * 70)
    
    from agent.react_agent import LangChainAgent
    from agent.llm_deepseek import get_default_chat_model
    from agent.mcp_adapter import MCPAdapter
    
    # 获取实际prompt
    agent = LangChainAgent(
        enable_system_prompt=True,
        enable_conversation_state=False,
        enable_quality_tracking=False,
        enable_intent_tracking=False,
        enable_recommendation=False,
    )
    
    actual_prompt = agent.prompt_manager.get_system_prompt() if agent.prompt_manager else ""
    
    # 获取LLM和工具
    llm = get_default_chat_model()
    adapter = MCPAdapter()
    tools = adapter.create_tools()
    tool_specs = [t.to_openai_tool() for t in tools]
    
    # 构建消息
    messages = [
        {"role": "system", "content": actual_prompt},
        {"role": "user", "content": "显示销量前10的商品柱状图"}
    ]
    
    print(f"\nPrompt长度: {len(actual_prompt)} 字符")
    print(f"工具数量: {len(tool_specs)}")
    print("用户消息: 显示销量前10的商品柱状图")
    
    # 调用LLM
    print("\n⏳ 调用LLM...")
    try:
        result = llm.generate(messages, tools=tool_specs)
        
        content = result.get("content", "")
        tool_calls = result.get("tool_calls", [])
        
        print(f"\n📥 LLM响应:")
        print(f"  内容: {content[:200] if content else '(空)'}...")
        print(f"  工具调用: {len(tool_calls)} 个")
        
        if tool_calls:
            for call in tool_calls:
                name = call.get("name")
                print(f"\n  ✅ 调用了工具: {name}")
                
                if name == "analytics_get_chart_data":
                    print("    🎉 成功！LLM调用了图表工具")
                    return True
        else:
            print("\n  ❌ LLM没有调用任何工具")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n🔍 诊断：为什么实际Agent不调用图表工具\n")
    
    # 1. 对比prompts
    actual_prompt, test_prompt = compare_prompts()
    
    # 2. 使用实际prompt测试
    success = test_with_actual_prompt()
    
    print("\n" + "=" * 70)
    print("💡 结论")
    print("=" * 70)
    
    if success:
        print("\n✅ 使用实际Agent的Prompt，LLM依然会调用图表工具")
        print("\n这说明问题可能在:")
        print("  1. Agent运行时的其他干扰因素")
        print("  2. 对话历史context干扰了LLM决策")
        print("  3. 用户的实际查询表达不够明确")
        print("  4. 记忆检索返回的内容影响了LLM判断")
        print("\n建议:")
        print("  - 查看实际对话中agent.run()的完整execution_log")
        print("  - 检查注入的context_prefix是否包含误导信息")
        print("  - 尝试更明确的查询如'调用图表工具生成柱状图'")
    else:
        print("\n❌ 使用实际Agent的Prompt，LLM不调用图表工具")
        print("\n问题出在Prompt本身!")
        print("  - Prompt太长或结构不合理")
        print("  - 图表工具描述被淹没在大量其他信息中")
        print("  - 需要优化Prompt，让图表工具更突出")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
