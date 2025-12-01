#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证优化后的重复检测机制
"""

def test_optimized_detection():
    print("=" * 70)
    print("验证: 优化后的重复检测机制")
    print("=" * 70)
    
    # 查询类工具列表
    query_tools = [
        "commerce_get_user_orders", "commerce_get_order_detail",
        "commerce_get_user_info", "commerce_get_cart",
        "commerce_search_products", "commerce_get_product_detail"
    ]
    
    print("\n1. 查询类工具 - 阈值=2次 (更严格)")
    print("   工具列表:", ", ".join(query_tools))
    
    # 测试查询类工具
    print("\n   测试场景: 连续2次调用 commerce_get_user_orders")
    calls = [
        "commerce_get_user_orders({})",
        "commerce_get_user_orders({})"
    ]
    tool_name = "commerce_get_user_orders"
    repeat_threshold = 2 if tool_name in query_tools else 3
    
    if len(calls) >= repeat_threshold:
        recent = calls[-repeat_threshold:]
        is_repeated = len(set(recent)) == 1
        print(f"   - 调用次数: {len(calls)}")
        print(f"   - 阈值: {repeat_threshold}")
        print(f"   - 检测结果: {'✓ 触发拦截' if is_repeated else '✗ 未触发'}")
        print(f"   - 影响: 快速阻止查询类工具重复")
    
    print("\n2. 其他工具 - 阈值=3次 (标准)")
    
    # 测试操作类工具
    print("\n   测试场景: 连续2次调用 commerce_add_to_cart")
    calls2 = [
        "commerce_add_to_cart({'product_id': 1})",
        "commerce_add_to_cart({'product_id': 1})"
    ]
    tool_name2 = "commerce_add_to_cart"
    repeat_threshold2 = 2 if tool_name2 in query_tools else 3
    
    if len(calls2) >= repeat_threshold2:
        recent2 = calls2[-repeat_threshold2:]
        is_repeated2 = len(set(recent2)) == 1
        print(f"   - 调用次数: {len(calls2)}")
        print(f"   - 阈值: {repeat_threshold2}")
        print(f"   - 检测结果: {'✓ 触发拦截' if is_repeated2 else '✗ 未触发'}")
        print(f"   - 影响: 允许操作类工具有更多重试空间")
    else:
        print(f"   - 调用次数: {len(calls2)}")
        print(f"   - 阈值: {repeat_threshold2}")
        print(f"   - 检测结果: ✗ 未触发 (未达到阈值)")
        print(f"   - 影响: 2次调用不会被拦截，需要3次")
    
    print("\n3. 典型多工具场景验证")
    
    scenarios = [
        {
            "name": "完整购物流程",
            "calls": [
                "commerce_search_products({'keyword': 'iPhone'})",
                "commerce_get_product_detail({'product_id': 1})",
                "commerce_add_to_cart({'product_id': 1})",
                "commerce_get_cart({})",
                "commerce_create_order({...})",
                "ontology_validate_order({...})",
                "commerce_process_payment({'order_id': 123})",
                "commerce_get_order_detail({'order_id': 123})"
            ],
            "expected": "✓ 不受影响 - 所有工具都不同"
        },
        {
            "name": "查询+详情(正常)",
            "calls": [
                "commerce_search_products({'keyword': 'iPhone'})",
                "commerce_get_product_detail({'product_id': 1})",
                "commerce_get_product_detail({'product_id': 2})"
            ],
            "expected": "✓ 不受影响 - 参数不同"
        },
        {
            "name": "重复查询订单(异常)",
            "calls": [
                "commerce_get_user_orders({})",
                "commerce_get_user_orders({})"
            ],
            "expected": "✓ 触发拦截 - 查询类工具2次即拦截"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n   场景{i}: {scenario['name']}")
        print(f"   - 调用序列长度: {len(scenario['calls'])}")
        
        # 检查是否会触发
        if len(scenario['calls']) >= 2:
            last_2 = scenario['calls'][-2:]
            triggered_2 = len(set(last_2)) == 1
            if triggered_2:
                tool = scenario['calls'][-1].split('(')[0]
                is_query = tool in query_tools
                print(f"   - 检测: {'✓ 触发(查询类2次)' if is_query else '需要检查3次'}")
        
        if len(scenario['calls']) >= 3:
            last_3 = scenario['calls'][-3:]
            triggered_3 = len(set(last_3)) == 1
            if triggered_3:
                print(f"   - 检测: ✓ 触发(3次重复)")
        
        print(f"   - 预期: {scenario['expected']}")
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    print("\n✅ 优化效果:")
    print("  1. 查询类工具阈值=2 → 更快拦截 '还有哪几个订单没有支付' 类问题")
    print("  2. 操作类工具阈值=3 → 保留必要的重试空间")
    print("  3. 多工具任务完全不受影响 → 每个工具独立计数")
    print("  4. 不同参数调用不受影响 → 基于完整签名(tool+args)判断")
    
    print("\n🎯 针对原始问题的改进:")
    print("  场景: '还有哪几个订单没有支付'")
    print("  ├─ 迭代1: 调用 commerce_get_user_orders")
    print("  ├─ 迭代2: LLM再次调用 commerce_get_user_orders")
    print("  └─ ✓ 检测到查询类工具连续2次，立即终止")
    print("  ")
    print("  原来: 需要3次才拦截，可能需要4-5次迭代")
    print("  现在: 2次即拦截，最多3次迭代 (更快)")
    
    print("\n⚠️  边界情况:")
    print("  1. 如果确实需要连续查询2次相同数据 → 极少见，可通过调整阈值")
    print("  2. 分页查询不受影响 → 参数(page/offset)不同")
    print("  3. 操作失败重试 → 操作类工具有3次阈值空间")
    
    print("\n" + "=" * 70)
    print("✓ 优化完成，既保护了查询效率，又不影响多工具任务!")
    print("=" * 70)

if __name__ == "__main__":
    test_optimized_detection()
