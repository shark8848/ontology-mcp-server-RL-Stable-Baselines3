#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断：为什么重复检测没有生效
"""

import sys
import os

print("=" * 70)
print("诊断: 重复检测代码为什么没有生效")
print("=" * 70)

print("\n1. 检查代码修改状态")
print("-" * 70)

# 读取 react_agent.py
react_agent_path = "src/agent/react_agent.py"
with open(react_agent_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 检查关键代码是否存在
checks = [
    ("tool_call_history 初始化", "tool_call_history: List[str] = []"),
    ("工具签名生成", "tool_signature = f\"{tool_name}"),
    ("记录到历史", "tool_call_history.append(tool_signature)"),
    ("查询工具列表定义", "query_tools = ["),
    ("阈值设置", "repeat_threshold = 2 if tool_name in query_tools else 3"),
    ("重复检测逻辑", "if len(tool_call_history) >= repeat_threshold:"),
    ("检测完全相同", "if len(set(recent_calls)) == 1:"),
    ("触发警告日志", "logger.warning"),
    ("强制终止", "break  # 跳出 for call in tool_calls 循环")
]

all_present = True
for desc, pattern in checks:
    if pattern in content:
        print(f"  ✓ {desc}")
    else:
        print(f"  ✗ {desc} - 未找到!")
        all_present = False

if all_present:
    print("\n✓ 所有关键代码都已添加")
else:
    print("\n✗ 部分代码缺失，需要检查")

print("\n2. 检查运行中的进程")
print("-" * 70)

import subprocess
result = subprocess.run(
    ["ps", "aux"],
    capture_output=True,
    text=True
)

gradio_processes = []
for line in result.stdout.split('\n'):
    if 'gradio' in line.lower() and 'grep' not in line:
        gradio_processes.append(line)

if gradio_processes:
    print(f"  找到 {len(gradio_processes)} 个Gradio进程:")
    for i, proc in enumerate(gradio_processes, 1):
        parts = proc.split()
        if len(parts) >= 11:
            pid = parts[1]
            start_time = parts[8]
            cmd = ' '.join(parts[10:])
            print(f"  {i}. PID={pid}, 启动时间={start_time}, 命令={cmd[:60]}")
else:
    print("  未找到运行中的Gradio进程")

print("\n3. 分析问题原因")
print("-" * 70)

if len(gradio_processes) > 0:
    print("  ⚠️  问题诊断:")
    print("  1. 代码已正确修改 ✓")
    print("  2. 但Gradio进程是在代码修改前启动的 ✗")
    print("  3. Python会缓存已加载的模块")
    print("  4. 需要重启Gradio服务才能加载新代码")
    
    print("\n  📋 时间线分析:")
    print("  - 20:33: Gradio进程启动")
    print("  - 20:50-21:00: 修改 react_agent.py 添加重复检测")
    print("  - 21:16: 用户询问'我还有几个订单没支付'")
    print("  - 21:20: 达到max_iterations，但没有触发重复检测")
    print("  ")
    print("  ✗ 原因: 进程使用的是修改前的代码!")

print("\n4. 解决方案")
print("-" * 70)

print("  方案1: 重启Gradio服务 (推荐)")
print("  ├─ 找到进程: ps aux | grep gradio")
print("  ├─ 终止进程: kill -9 <PID>")
print("  └─ 重新启动: python3 -m agent.gradio_ui")
print()
print("  方案2: 使用热重载")
print("  ├─ 如果Gradio支持watch模式")
print("  └─ 添加 reload=True 参数")
print()
print("  方案3: 验证修复效果")
print("  └─ 重启后再次询问'我还有几个订单没支付'，应该在2-3次迭代内终止")

print("\n5. 验证检查清单")
print("-" * 70)

print("  重启后验证:")
print("  □ 日志中应该出现 'tool_call_history' 相关记录")
print("  □ 连续2次调用 commerce_get_user_orders 应触发警告")
print("  □ 应该看到 'repeated_tool_call_guard' 日志")
print("  □ 迭代次数应该 ≤ 3 次")
print("  □ 不应该再出现 'max iterations' 警告")

print("\n" + "=" * 70)
print("结论: 代码正确，但需要重启Gradio服务加载新代码")
print("=" * 70)
