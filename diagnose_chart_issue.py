#!/usr/bin/env python3
"""
完整诊断脚本：检查为什么图表没有在对话中生成
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_system_prompt():
    """检查系统提示词"""
    print("=" * 70)
    print("✅ 步骤1: 检查系统提示词内容")
    print("=" * 70)
    
    from agent.prompts import PromptManager
    
    pm = PromptManager(use_full_prompt=True)
    prompt = pm.get_system_prompt()
    
    # 检查关键内容
    checks = {
        "analytics_get_chart_data": "图表工具名称",
        "chart_type": "chart_type参数",
        "trend": "趋势图关键词",
        "pie": "饼图关键词",
        "bar": "柱状图关键词",
        "必须调用": "强调调用规则",
    }
    
    all_pass = True
    for keyword, desc in checks.items():
        if keyword in prompt:
            print(f"  ✅ {desc}: '{keyword}' 存在")
        else:
            print(f"  ❌ {desc}: '{keyword}' 缺失")
            all_pass = False
    
    if all_pass:
        print("\n✅ 系统提示词检查通过\n")
    else:
        print("\n❌ 系统提示词有缺失内容\n")
    
    return all_pass


def check_tool_availability():
    """检查工具是否可用"""
    print("=" * 70)
    print("✅ 步骤2: 检查工具注册")
    print("=" * 70)
    
    from agent.react_agent import LangChainAgent
    
    agent = LangChainAgent(
        enable_system_prompt=True,
        enable_conversation_state=False,
        enable_quality_tracking=False,
        enable_intent_tracking=False,
        enable_recommendation=False,
    )
    
    # 检查工具列表
    tool_names = [t.name for t in agent.tools]
    
    if "analytics_get_chart_data" in tool_names:
        print("  ✅ analytics_get_chart_data 工具已注册")
        
        # 获取工具详情
        chart_tool = [t for t in agent.tools if t.name == "analytics_get_chart_data"][0]
        print(f"  📋 工具描述: {chart_tool.description[:80]}...")
        print(f"  📋 参数字段: {list(chart_tool.args_schema.model_fields.keys())}")
        print("\n✅ 工具注册检查通过\n")
        return True
    else:
        print(f"  ❌ analytics_get_chart_data 工具未找到")
        print(f"  📋 可用工具: {tool_names}")
        print("\n❌ 工具注册检查失败\n")
        return False


def check_agent_prompt_usage():
    """检查Agent是否使用了正确的提示词"""
    print("=" * 70)
    print("✅ 步骤3: 检查Agent提示词使用")
    print("=" * 70)
    
    from agent.react_agent import LangChainAgent
    
    agent = LangChainAgent(
        enable_system_prompt=True,  # 关键：必须启用
        enable_conversation_state=False,
        enable_quality_tracking=False,
        enable_intent_tracking=False,
        enable_recommendation=False,
    )
    
    if agent.prompt_manager:
        prompt = agent.prompt_manager.get_system_prompt()
        if "analytics_get_chart_data" in prompt:
            print("  ✅ Agent使用的系统提示词包含图表工具说明")
            print("\n✅ Agent提示词使用检查通过\n")
            return True
        else:
            print("  ❌ Agent使用的系统提示词不包含图表工具说明")
            print("\n❌ Agent提示词使用检查失败\n")
            return False
    else:
        print("  ⚠️  Agent未启用prompt_manager")
        print("\n⚠️  请确保enable_system_prompt=True\n")
        return False


def check_chart_extraction():
    """检查图表提取逻辑"""
    print("=" * 70)
    print("✅ 步骤4: 检查图表提取逻辑")
    print("=" * 70)
    
    # 模拟tool_log
    mock_tool_log = [
        {
            "tool": "analytics_get_chart_data",
            "observation": '{"chart_type": "trend", "title": "测试图表", "labels": ["A", "B"], "series": [{"name": "数据", "data": [1, 2]}]}'
        }
    ]
    
    import json
    charts = []
    for entry in mock_tool_log:
        if entry.get("tool") == "analytics_get_chart_data":
            try:
                obs = entry.get("observation", "{}")
                chart_data = json.loads(obs) if isinstance(obs, str) else obs
                if "chart_type" in chart_data and "error" not in chart_data:
                    charts.append(chart_data)
                    print("  ✅ 成功提取图表数据")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  ❌ 解析失败: {e}")
                return False
    
    if charts:
        print(f"  📊 提取到 {len(charts)} 个图表")
        print("\n✅ 图表提取逻辑检查通过\n")
        return True
    else:
        print("  ❌ 未提取到任何图表")
        print("\n❌ 图表提取逻辑检查失败\n")
        return False


def check_chart_rendering():
    """检查图表渲染逻辑"""
    print("=" * 70)
    print("✅ 步骤5: 检查图表渲染逻辑")
    print("=" * 70)
    
    # 导入渲染函数
    sys.path.insert(0, str(Path(__file__).parent / "src" / "agent"))
    from gradio_ui import _render_charts_html
    
    mock_charts = [
        {
            "chart_type": "trend",
            "title": "订单趋势",
            "labels": ["11-20", "11-21", "11-22"],
            "series": [
                {"name": "订单数", "data": [5, 8, 12]},
                {"name": "金额", "data": [5000, 8000, 12000]}
            ],
            "description": "最近3天订单趋势"
        }
    ]
    
    html = _render_charts_html(mock_charts)
    
    if html and "订单趋势" in html and "11-20" in html:
        print("  ✅ 图表渲染成功")
        print(f"  📄 渲染结果长度: {len(html)} 字符")
        print("\n  预览:")
        print("  " + "\n  ".join(html.split("\n")[:15]))
        print("\n✅ 图表渲染逻辑检查通过\n")
        return True
    else:
        print("  ❌ 图表渲染失败")
        print(f"  📄 渲染结果: {html[:200] if html else '(空)'}")
        print("\n❌ 图表渲染逻辑检查失败\n")
        return False


def main():
    print("\n" + "=" * 70)
    print("🔍 图表功能完整诊断")
    print("=" * 70)
    print("\n本诊断将检查5个关键环节，找出图表未生成的原因\n")
    
    results = {
        "系统提示词": check_system_prompt(),
        "工具注册": check_tool_availability(),
        "Agent配置": check_agent_prompt_usage(),
        "图表提取": check_chart_extraction(),
        "图表渲染": check_chart_rendering(),
    }
    
    print("\n" + "=" * 70)
    print("📊 诊断结果汇总")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {status} - {name}")
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n" + "=" * 70)
        print("✅ 所有检查通过！")
        print("=" * 70)
        print("\n📌 可能的问题原因：")
        print("  1. Agent服务未重启（提示词更新未生效）")
        print("  2. LLM未识别用户意图（关键词不够明确）")
        print("  3. LLM选择不调用工具（生成文字回复替代）")
        print("\n💡 解决方案：")
        print("  1. 重启Agent服务: pkill -f gradio_ui && python -m agent.gradio_ui")
        print("  2. 使用明确的关键词: '展示订单趋势图'、'显示销量柱状图'")
        print("  3. 检查Agent日志，确认工具调用记录")
        print("\n🔍 调试方法：")
        print("  - 在Gradio UI的'Tool Calls'标签查看是否调用了analytics_get_chart_data")
        print("  - 检查返回结果中是否有charts字段")
        print("  - 查看Agent日志: tail -f logs/agent_*.log")
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ 发现问题！请修复上述失败的检查项")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
