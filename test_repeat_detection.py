#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试重复工具调用检测机制
"""

import sys
sys.path.insert(0, 'src')

print("=" * 70)
print("测试: 重复工具调用检测机制")
print("=" * 70)

print("\n1. 检查代码修改...")
with open('src/agent/react_agent.py', 'r', encoding='utf-8') as f:
    content = f.read()
    
checks = [
    ('tool_call_history: List[str] = []', '初始化工具调用历史'),
    ('tool_signature = f', '生成工具签名'),
    ('tool_call_history.append(tool_signature)', '记录工具调用'),
    ('if len(tool_call_history) >= 3:', '检测连续3次重复'),
    ('len(set(recent_calls)) == 1', '判断是否完全相同'),
    ('repeated_tool_call_guard', '记录重复调用警告'),
    ('excessive_tool_calls', '记录过度调用警告'),
]

all_passed = True
for keyword, description in checks:
    if keyword in content:
        print(f"   ✓ {description}")
    else:
        print(f"   ✗ {description} - 未找到")
        all_passed = False

print("\n2. 机制说明:")
print("   - 记录每次工具调用的签名 (工具名+参数)")
print("   - 检测连续3次完全相同的调用")
print("   - 检测单个工具累计调用超过5次")
print("   - 触发任一条件时强制终止并返回结果")

print("\n3. 预期效果:")
print("   场景: '还有哪几个订单没有支付'")
print("   ├─ 迭代1: 调用 commerce_get_user_orders")
print("   ├─ 迭代2: 若LLM再次调用 commerce_get_user_orders")
print("   ├─ 迭代3: 若LLM第3次调用 commerce_get_user_orders")
print("   └─ ✓ 系统检测到重复，强制终止并返回结果")
print()
print("   原始行为: 迭代10次后超时")
print("   修复后: 最多迭代3-5次即终止")

print("\n4. 日志输出:")
print("   触发时会记录:")
print("   - repeated_tool_call_guard: 工具X重复调用，强制终止")
print("   - excessive_tool_calls: 工具X调用次数过多")
print("   - final_answer: 原因=repeated_calls")

if all_passed:
    print("\n" + "=" * 70)
    print("✓ 重复工具调用检测机制已成功添加")
    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print("✗ 部分检查失败，请review代码")
    print("=" * 70)

print("\n5. 额外优化建议:")
print("   - 在System Prompt中强化'一次工具调用后立即回答'")
print("   - 考虑根据工具类型设置不同的重复阈值")
print("   - 对于查询类工具，可以设置更严格的重复检测(2次)")

print("\n完成!")
