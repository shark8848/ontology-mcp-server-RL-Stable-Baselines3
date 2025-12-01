#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实际测试重复检测功能
通过模拟用户查询来验证重复检测是否生效
"""

import sys
sys.path.insert(0, 'src')

from agent.react_agent import OntologyMCPAgent
from agent.logger import LOGGER
import json

print("=" * 70)
print("实时测试: 重复检测功能验证")
print("=" * 70)

# 初始化Agent
print("\n1. 初始化Agent...")
agent = OntologyMCPAgent()
print("✓ Agent初始化完成")

# 测试用例: 这个查询之前会导致重复调用 commerce_get_user_orders
test_query = "我还有几个订单没有支付"

print(f"\n2. 测试查询: '{test_query}'")
print("-" * 70)
print("⚠️  如果重复检测生效，应该:")
print("  - 日志中出现 'tool_call_history' 相关记录")
print("  - 连续2次调用同一查询工具时触发警告")
print("  - 迭代次数 ≤ 3 次")
print("  - 不会出现 'max iterations' 警告")
print()

try:
    result = agent.run(test_query)
    
    print("\n3. 执行结果:")
    print("-" * 70)
    
    if "error" in result:
        print(f"  ✗ 执行失败: {result['error']}")
    else:
        # 检查迭代次数
        tools_used = result.get("tools_used", [])
        iterations = result.get("iterations", 0)
        
        print(f"  · 迭代次数: {iterations}")
        print(f"  · 工具调用次数: {len(tools_used)}")
        print(f"  · 工具列表: {tools_used}")
        
        # 检查是否有重复
        from collections import Counter
        tool_counts = Counter(tools_used)
        has_repeats = any(count >= 2 for count in tool_counts.values())
        
        if has_repeats:
            print("\n  ⚠️  检测到重复调用:")
            for tool, count in tool_counts.items():
                if count >= 2:
                    print(f"    - {tool}: {count}次")
        
        # 判断测试结果
        print("\n4. 测试结论:")
        print("-" * 70)
        
        if iterations <= 3 and not has_repeats:
            print("  ✅ 重复检测生效! 迭代次数正常且无重复调用")
        elif iterations <= 3 and has_repeats:
            print("  ⚠️  有重复调用但被及时终止 (可接受)")
        else:
            print("  ✗ 重复检测可能未生效 (迭代次数过多或重复调用未被阻止)")
        
        # 显示最终回答
        answer = result.get("output", "无输出")
        print(f"\n  最终回答: {answer[:200]}...")

except Exception as e:
    print(f"\n  ✗ 执行异常: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("提示: 请检查日志文件确认是否有重复检测相关日志")
print("  tail -50 src/agent/logs/agent.log | grep 'tool_call_history\\|repeated_tool'")
print("=" * 70)
