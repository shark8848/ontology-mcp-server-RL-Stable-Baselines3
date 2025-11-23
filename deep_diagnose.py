#!/usr/bin/env python3
"""
深度诊断：检查LLM实际收到的系统提示词
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_actual_prompt():
    """测试Agent实际使用的系统提示词"""
    print("=" * 70)
    print("🔍 检查LLM实际收到的系统提示词")
    print("=" * 70)
    
    from agent.react_agent import LangChainAgent
    
    # 创建Agent实例（模拟实际运行）
    agent = LangChainAgent(
        enable_system_prompt=True,
        enable_conversation_state=False,
        enable_quality_tracking=False,
        enable_intent_tracking=False,
        enable_recommendation=False,
    )
    
    # 获取实际的系统提示词
    if agent.prompt_manager:
        actual_prompt = agent.prompt_manager.get_system_prompt()
        
        print("\n📄 系统提示词长度:", len(actual_prompt), "字符")
        print("\n🔍 关键内容检查:")
        
        checks = {
            "analytics_get_chart_data": "图表工具名称",
            "**必须调用**": "强制调用规则",
            "chart_type=\"trend\"": "趋势图示例",
            "chart_type=\"bar\"": "柱状图示例",
            "用户要求看图表时": "图表触发条件",
        }
        
        all_found = True
        for keyword, desc in checks.items():
            if keyword in actual_prompt:
                print(f"  ✅ {desc}: 找到")
            else:
                print(f"  ❌ {desc}: 未找到")
                all_found = False
        
        if all_found:
            print("\n✅ 系统提示词内容完整\n")
        else:
            print("\n❌ 系统提示词内容不完整\n")
            return False
        
        # 显示图表相关的完整段落
        print("\n📋 图表工具相关段落:")
        print("-" * 70)
        
        # 查找并显示图表相关内容
        lines = actual_prompt.split("\n")
        in_chart_section = False
        chart_lines = []
        
        for i, line in enumerate(lines):
            if "数据可视化工具" in line or "analytics_get_chart_data" in line:
                in_chart_section = True
                # 显示前后几行
                start = max(0, i - 2)
                end = min(len(lines), i + 10)
                chart_lines = lines[start:end]
                break
        
        if chart_lines:
            for line in chart_lines:
                print(line)
        else:
            print("⚠️  未找到图表工具相关段落")
        
        print("-" * 70)
        
        return True
    else:
        print("❌ Agent未启用prompt_manager")
        return False


def test_tool_in_openai_format():
    """测试工具在OpenAI格式中的描述"""
    print("\n" + "=" * 70)
    print("🔍 检查工具的OpenAI格式定义")
    print("=" * 70)
    
    from agent.mcp_adapter import MCPAdapter
    
    adapter = MCPAdapter()
    tools = adapter.create_tools()
    
    chart_tool = None
    for tool in tools:
        if tool.name == "analytics_get_chart_data":
            chart_tool = tool
            break
    
    if not chart_tool:
        print("❌ 未找到图表工具")
        return False
    
    # 转换为OpenAI格式
    openai_tool = chart_tool.to_openai_tool()
    
    print("\n📋 工具定义（OpenAI格式）:")
    print("-" * 70)
    import json
    print(json.dumps(openai_tool, indent=2, ensure_ascii=False))
    print("-" * 70)
    
    # 检查关键字段
    func = openai_tool.get("function", {})
    desc = func.get("description", "")
    
    if "trend" in desc and "pie" in desc and "bar" in desc:
        print("\n✅ 工具描述包含所有图表类型")
        return True
    else:
        print("\n⚠️  工具描述可能不完整")
        return False


def test_mock_llm_call():
    """模拟LLM调用，查看实际传递的messages"""
    print("\n" + "=" * 70)
    print("🔍 模拟LLM调用（查看实际messages）")
    print("=" * 70)
    
    from agent.react_agent import LangChainAgent
    
    agent = LangChainAgent(
        enable_system_prompt=True,
        enable_conversation_state=False,
        enable_quality_tracking=False,
        enable_intent_tracking=False,
        enable_recommendation=False,
    )
    
    # 构建一个简单的测试查询
    test_input = "显示销量前10的商品柱状图"
    
    print(f"\n📝 测试输入: {test_input}")
    print("\n📤 发送给LLM的消息结构:")
    print("-" * 70)
    
    # 获取系统提示词
    if agent.prompt_manager:
        system_prompt = agent.prompt_manager.get_system_prompt()
        
        # 显示system message
        print("\n1. System Message:")
        print(f"   长度: {len(system_prompt)} 字符")
        
        # 显示图表相关部分
        if "analytics_get_chart_data" in system_prompt:
            print("   ✅ 包含图表工具说明")
            
            # 提取相关段落
            start = system_prompt.find("数据可视化工具")
            if start > 0:
                snippet = system_prompt[start:start+300]
                print(f"\n   预览:\n   {snippet[:200]}...")
        else:
            print("   ❌ 不包含图表工具说明")
        
        # 显示user message
        print("\n2. User Message:")
        print(f"   内容: {test_input}")
        
        # 显示tools
        print("\n3. Tools (可用工具列表):")
        print(f"   工具数量: {len(agent.tools)}")
        chart_tool_exists = any(t.name == "analytics_get_chart_data" for t in agent.tools)
        if chart_tool_exists:
            print("   ✅ 包含 analytics_get_chart_data")
        else:
            print("   ❌ 不包含 analytics_get_chart_data")
        
        print("-" * 70)
        
        return True
    else:
        print("❌ 无法获取系统提示词")
        return False


def main():
    print("\n" + "=" * 70)
    print("🔬 深度诊断：LLM提示词传递检查")
    print("=" * 70)
    print("\n这个脚本将检查LLM实际收到的系统提示词和工具定义\n")
    
    results = {}
    
    try:
        results["系统提示词内容"] = test_actual_prompt()
        results["工具OpenAI格式"] = test_tool_in_openai_format()
        results["LLM调用结构"] = test_mock_llm_call()
        
        print("\n" + "=" * 70)
        print("📊 诊断结果汇总")
        print("=" * 70)
        
        for name, passed in results.items():
            status = "✅ 正常" if passed else "❌ 异常"
            print(f"  {status} - {name}")
        
        if all(results.values()):
            print("\n" + "=" * 70)
            print("✅ 所有检查通过 - 配置正确")
            print("=" * 70)
            print("\n📌 结论：")
            print("  - 系统提示词包含图表工具说明")
            print("  - 工具已正确注册并可用")
            print("  - LLM可以接收到完整信息")
            print("\n🤔 如果图表仍然不生成，可能的原因：")
            print("  1. LLM主动选择不调用工具（认为文字描述更合适）")
            print("  2. 用户表达不够明确（缺少'图表'、'柱状图'等关键词）")
            print("  3. LLM温度参数过高，导致行为不稳定")
            print("\n💡 解决方案：")
            print("  1. 使用更明确的表达：'生成柱状图'、'展示趋势图'")
            print("  2. 在查询中强调'图表'或'可视化'")
            print("  3. 检查config.yaml中的temperature设置（建议≤0.7）")
            return 0
        else:
            print("\n❌ 发现问题，请检查上述异常项")
            return 1
            
    except Exception as e:
        print(f"\n❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
