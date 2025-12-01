#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证运行中的代码是否包含重复检测逻辑
"""

import sys
sys.path.insert(0, 'src')

# 导入 react_agent 模块
from agent import react_agent
import inspect

print("=" * 70)
print("验证 react_agent.py 是否包含重复检测代码")
print("=" * 70)

# 获取 run_agent 函数的源码
source = inspect.getsource(react_agent.run_agent)

# 检查关键代码片段
checks = [
    ("tool_call_history 初始化", "tool_call_history"),
    ("工具签名生成", "tool_signature"),
    ("查询工具列表", "query_tools"),
    ("重复阈值设置", "repeat_threshold"),
    ("重复检测逻辑", "if len(set(recent_calls)) == 1"),
]

print("\n检查结果:")
print("-" * 70)

all_present = True
for desc, pattern in checks:
    if pattern in source:
        print(f"  ✓ {desc}")
    else:
        print(f"  ✗ {desc} - 未找到!")
        all_present = False

print()
if all_present:
    print("✅ 运行中的代码包含所有重复检测逻辑")
else:
    print("❌ 运行中的代码缺少部分重复检测逻辑")
    print("   可能原因: Python缓存了旧版本模块")
    print("   解决方案: 重新导入或重启进程")

print("\n" + "=" * 70)
