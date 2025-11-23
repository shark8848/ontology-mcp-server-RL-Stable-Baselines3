#!/usr/bin/env python3
"""
重启后验证脚本：测试图表功能是否正常工作
使用方法: python3 verify_chart_fix.py
"""

import sys
import requests
from pathlib import Path

# 配置
GRADIO_URL = "http://localhost:7860"  # Gradio UI地址
TEST_QUERIES = [
    "展示最近7天的订单趋势图",
    "显示销量前10的商品柱状图",
    "给我看商品分类占比饼图",
]


def check_service_running():
    """检查Gradio服务是否运行"""
    try:
        response = requests.get(GRADIO_URL, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def print_box(title, content, color="blue"):
    """打印美化的文本框"""
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "end": "\033[0m"
    }
    
    lines = content.split("\n")
    max_len = max(len(line) for line in lines)
    border = "=" * (max_len + 4)
    
    print(f"\n{colors.get(color, '')}{border}")
    print(f"  {title}")
    print(f"{border}{colors['end']}")
    for line in lines:
        print(f"  {line}")
    print()


def main():
    print("\n" + "=" * 70)
    print("🔍 图表功能验证脚本")
    print("=" * 70)
    
    # 步骤1: 检查服务状态
    print("\n📍 步骤1: 检查Gradio服务状态")
    if check_service_running():
        print_box(
            "✅ 服务正常运行",
            f"Gradio UI: {GRADIO_URL}",
            "green"
        )
    else:
        print_box(
            "❌ 服务未运行",
            f"请先启动服务:\npython3 -m agent.gradio_ui",
            "red"
        )
        return 1
    
    # 步骤2: 提供测试指南
    print("📍 步骤2: 手动测试图表功能")
    print_box(
        "测试用例",
        "\n".join([f"{i+1}. {q}" for i, q in enumerate(TEST_QUERIES)]),
        "blue"
    )
    
    print("📋 验证清单:")
    checklist = [
        f"在浏览器打开: {GRADIO_URL}",
        "依次输入上述测试用例",
        "检查回复中是否有 '📊 数据可视化' 标题",
        "检查是否有Markdown表格",
        "切换到'Tool Calls'标签，查看是否调用了analytics_get_chart_data"
    ]
    for item in checklist:
        print(f"  [ ] {item}")
    
    # 步骤3: 对比检查
    print("\n📍 步骤3: 对比修复前后")
    
    print("\n❌ 修复前的回复特征:")
    print("  - 包含大段文字分析")
    print("  - 出现'系统无法生成图表'或类似表述")
    print("  - 没有Markdown表格")
    print("  - Tool Calls标签中没有图表工具调用")
    
    print("\n✅ 修复后的预期回复:")
    print("  - 简短的引导文字")
    print("  - '---' 分割线")
    print("  - '## 📊 数据可视化' 标题")
    print("  - Markdown表格（| 项目 | ... |）")
    print("  - Tool Calls标签显示analytics_get_chart_data调用")
    
    # 步骤4: 日志检查
    print("\n📍 步骤4: 检查日志（可选）")
    
    log_dir = Path("logs")
    if log_dir.exists():
        agent_logs = list(log_dir.glob("agent_*.log"))
        if agent_logs:
            latest_log = max(agent_logs, key=lambda p: p.stat().st_mtime)
            print(f"\n  最新日志文件: {latest_log}")
            print(f"\n  查看最后50行:")
            print(f"  $ tail -n 50 {latest_log}")
            print(f"\n  搜索图表调用:")
            print(f"  $ grep 'analytics_get_chart_data' {latest_log}")
        else:
            print("  ⚠️  未找到agent日志文件")
    else:
        print("  ⚠️  logs目录不存在")
    
    # 步骤5: 问题排查
    print("\n📍 步骤5: 如果仍然不工作")
    print_box(
        "问题排查步骤",
        """1. 确认已重启服务（查看进程启动时间）
2. 检查MCP Server是否运行（http://localhost:8000/health）
3. 查看Gradio UI控制台输出是否有错误
4. 尝试更明确的关键词（如'展示趋势图'而非'分析趋势'）
5. 重新运行诊断: python3 diagnose_chart_issue.py""",
        "yellow"
    )
    
    # 总结
    print("\n" + "=" * 70)
    print("📊 验证步骤总结")
    print("=" * 70)
    print("\n1. ✅ 检查服务状态")
    print("2. 🧪 手动测试（在浏览器中）")
    print("3. 🔍 对比修复前后的回复特征")
    print("4. 📋 （可选）检查日志文件")
    print("5. 🔧 （如需）问题排查")
    
    print("\n" + "=" * 70)
    print("💡 提示")
    print("=" * 70)
    print("\n如果所有测试都通过，说明图表功能已正常工作！")
    print("如果仍有问题，请查看诊断脚本获取详细帮助。")
    print("\n✨ 祝使用愉快！\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
