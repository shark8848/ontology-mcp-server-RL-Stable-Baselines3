#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析"还有哪几个订单没有支付"为何迭代到上限的问题
"""

import json
import sys
sys.path.insert(0, 'src')

from agent.react_agent import ReactAgent
from agent.llm_deepseek import DeepseekLLM
from ontology_mcp_server.db_service import DatabaseService

def analyze_unpaid_orders_issue():
    """分析未支付订单查询的迭代问题"""
    
    print("=" * 70)
    print("分析: '还有哪几个订单没有支付' 迭代上限问题")
    print("=" * 70)
    
    # 1. 查询实际的未支付订单数据
    print("\n1. 查询数据库中的未支付订单...")
    db = DatabaseService(db_path="data/ecommerce.db")
    
    with db.get_session() as session:
        from ontology_mcp_server.models import Order
        unpaid_orders = session.query(Order).filter(
            Order.payment_status == 'unpaid'
        ).all()
        
        print(f"   ✓ 找到 {len(unpaid_orders)} 个未支付订单")
        for order in unpaid_orders[:5]:  # 只显示前5个
            print(f"     - Order {order.order_id}: {order.order_no}, "
                  f"金额={order.final_amount}, 状态={order.payment_status}")
    
    # 2. 分析问题根源
    print("\n2. 问题根源分析:")
    print("   根据日志和代码分析，问题可能在于:")
    print()
    print("   ❌ LLM 持续调用工具而不返回最终答案")
    print("      - 可能原因1: LLM 认为需要多次查询来获取完整信息")
    print("      - 可能原因2: 工具返回的数据格式导致 LLM 理解困难")
    print("      - 可能原因3: Prompt 中缺少明确的'何时停止'指示")
    print()
    print("   循环终止条件 (react_agent.py:688-702):")
    print("   ```python")
    print("   if not tool_calls:")
    print("       final_answer = assistant_content")
    print("       break")
    print("   ```")
    print()
    print("   ✓ 只有当 LLM 不再调用工具时才会 break")
    print("   ✗ 如果 LLM 一直调用工具，将迭代到 max_iterations=10")
    
    # 3. 检查工具返回的数据
    print("\n3. 检查 commerce_get_user_orders 工具返回:")
    from ontology_mcp_server.commerce_service import CommerceService
    service = CommerceService(db_path="data/ecommerce.db")
    
    # 模拟工具调用
    orders = service.orders.list_orders(limit=100)
    user_1_orders = [o for o in orders if o.user_id == 1]
    unpaid = [o for o in user_1_orders if o.payment_status == 'unpaid']
    
    print(f"   用户1的订单总数: {len(user_1_orders)}")
    print(f"   未支付订单数: {len(unpaid)}")
    
    if unpaid:
        print(f"\n   未支付订单详情:")
        for order in unpaid[:3]:
            print(f"   - 订单号: {order.order_no}")
            print(f"     订单ID: {order.order_id}")
            print(f"     金额: {order.final_amount}")
            print(f"     支付状态: {order.payment_status}")
            print(f"     创建时间: {order.created_at}")
    
    # 4. 分析可能的解决方案
    print("\n4. 可能的解决方案:")
    print()
    print("   方案1: 优化 System Prompt")
    print("   ├─ 添加明确指示: '一次工具调用后立即总结并回答用户'")
    print("   ├─ 强调: '避免重复调用同一工具'")
    print("   └─ 示例: '获取数据后，直接用自然语言总结，不要再次查询'")
    print()
    print("   方案2: 添加重复工具调用检测")
    print("   ├─ 在 react_agent.py 中记录已调用的工具")
    print("   ├─ 如果同一工具被连续调用2次，强制返回答案")
    print("   └─ 代码位置: react_agent.py:704-810 (工具调用循环)")
    print()
    print("   方案3: 改进工具返回格式")
    print("   ├─ 在返回数据中添加明确的总结信息")
    print("   ├─ 添加 'summary' 字段告诉 LLM 数据已完整")
    print("   └─ 示例: {'orders': [...], 'summary': '找到3个未支付订单，已全部返回'}")
    print()
    print("   方案4: 设置更合理的 max_iterations")
    print("   ├─ 当前默认值: 10")
    print("   ├─ 对于简单查询，5次迭代应该足够")
    print("   └─ 可以根据任务类型动态调整")
    
    # 5. 检查 Prompt
    print("\n5. 检查当前的 System Prompt:")
    from agent.prompts import REACT_SYSTEM_PROMPT
    
    # 查找关键指示
    if "一次调用" in REACT_SYSTEM_PROMPT or "立即回答" in REACT_SYSTEM_PROMPT:
        print("   ✓ Prompt 中包含明确的单次调用指示")
    else:
        print("   ⚠️  Prompt 中缺少明确的单次调用指示")
    
    if "重复" in REACT_SYSTEM_PROMPT:
        print("   ✓ Prompt 中提到避免重复调用")
    else:
        print("   ⚠️  Prompt 中未提醒避免重复调用")
    
    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)
    
    return {
        "unpaid_orders_count": len(unpaid) if 'unpaid' in locals() else 0,
        "issue": "LLM持续调用工具导致迭代上限",
        "root_cause": "缺少明确的停止信号或重复调用检测",
        "recommended_solution": "方案1+方案2组合: 优化Prompt + 添加重复检测"
    }

if __name__ == "__main__":
    try:
        result = analyze_unpaid_orders_issue()
        print(f"\n📊 分析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"\n✗ 分析失败: {e}")
        import traceback
        traceback.print_exc()
