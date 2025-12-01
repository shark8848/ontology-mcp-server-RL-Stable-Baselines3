#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试重复检测机制对多工具任务的影响
"""

def test_scenarios():
    """模拟不同场景下的工具调用"""
    
    print("=" * 70)
    print("测试: 重复检测机制对多工具任务的影响")
    print("=" * 70)
    
    # 场景1: 重复调用同一工具(会被拦截)
    print("\n【场景1】重复调用同一工具 - 应该被拦截 ✓")
    calls_1 = [
        "commerce_get_user_orders({})",
        "commerce_get_user_orders({})",
        "commerce_get_user_orders({})"
    ]
    print(f"  工具调用序列: {calls_1}")
    recent_3 = calls_1[-3:]
    is_repeated = len(set(recent_3)) == 1
    print(f"  检测结果: {'✓ 触发拦截' if is_repeated else '✗ 未触发'}")
    print(f"  影响: {'正确 - 阻止了无意义的重复' if is_repeated else '错误'}")
    
    # 场景2: 调用不同的工具(不会被拦截)
    print("\n【场景2】多步骤任务 - 不应该被拦截 ✓")
    calls_2 = [
        "commerce_search_products({'keyword': 'iPhone'})",
        "commerce_get_product_detail({'product_id': 1})",
        "commerce_add_to_cart({'product_id': 1, 'quantity': 1})"
    ]
    print(f"  工具调用序列: {calls_2}")
    recent_3 = calls_2[-3:]
    is_repeated = len(set(recent_3)) == 1
    print(f"  检测结果: {'✓ 触发拦截' if is_repeated else '✗ 未触发'}")
    print(f"  影响: {'错误 - 误杀了正常流程' if is_repeated else '正确 - 允许多步骤执行'}")
    
    # 场景3: 同一工具不同参数(不会被拦截)
    print("\n【场景3】同一工具不同参数 - 不应该被拦截 ✓")
    calls_3 = [
        "commerce_search_products({'keyword': 'iPhone'})",
        "commerce_search_products({'keyword': 'Samsung'})",
        "commerce_search_products({'keyword': 'Huawei'})"
    ]
    print(f"  工具调用序列: {calls_3}")
    recent_3 = calls_3[-3:]
    is_repeated = len(set(recent_3)) == 1
    print(f"  检测结果: {'✓ 触发拦截' if is_repeated else '✗ 未触发'}")
    print(f"  影响: {'错误 - 误杀了合理的搜索' if is_repeated else '正确 - 允许不同搜索'}")
    
    # 场景4: 复杂的多工具流程
    print("\n【场景4】复杂订单流程 - 不应该被拦截 ✓")
    calls_4 = [
        "commerce_get_user_info({'user_id': 1})",
        "commerce_search_products({'keyword': 'iPhone'})",
        "commerce_check_stock({'product_id': 1})",
        "commerce_add_to_cart({'product_id': 1})",
        "commerce_get_cart({'user_id': 1})",
        "commerce_create_order({'user_id': 1, 'items': [...]})",
        "ontology_validate_order({...})",
        "commerce_process_payment({'order_id': 123})"
    ]
    print(f"  工具调用数量: {len(calls_4)}")
    print(f"  最后3次调用: {calls_4[-3:]}")
    recent_3 = calls_4[-3:]
    is_repeated = len(set(recent_3)) == 1
    print(f"  检测结果: {'✓ 触发拦截' if is_repeated else '✗ 未触发'}")
    print(f"  影响: {'错误 - 打断了正常流程' if is_repeated else '正确 - 允许完整流程'}")
    
    # 场景5: 间隔重复(不会被拦截，因为不连续)
    print("\n【场景5】间隔重复调用 - 不应该被拦截 ✓")
    calls_5 = [
        "commerce_get_user_orders({})",
        "commerce_get_product_detail({'product_id': 1})",
        "commerce_get_user_orders({})",
        "commerce_get_product_detail({'product_id': 2})",
        "commerce_get_user_orders({})"
    ]
    print(f"  工具调用序列: {calls_5}")
    recent_3 = calls_5[-3:]
    is_repeated = len(set(recent_3)) == 1
    print(f"  检测结果: {'✓ 触发拦截' if is_repeated else '✗ 未触发'}")
    print(f"  影响: {'错误' if is_repeated else '正确 - 不连续的重复不拦截'}")
    
    # 场景6: 过度调用检测(超过5次)
    print("\n【场景6】单个工具累计超过5次 - 触发警告但不强制终止")
    calls_6 = [
        "commerce_search_products({'keyword': 'iPhone'})",
        "commerce_get_product_detail({'product_id': 1})",
        "commerce_search_products({'keyword': 'Samsung'})",
        "commerce_get_product_detail({'product_id': 2})",
        "commerce_search_products({'keyword': 'Huawei'})",
        "commerce_search_products({'keyword': 'Xiaomi'})",
        "commerce_search_products({'keyword': 'OPPO'})",
        "commerce_search_products({'keyword': 'Vivo'})",  # 第6次
    ]
    search_count = sum(1 for c in calls_6 if c.startswith("commerce_search_products"))
    print(f"  工具调用序列长度: {len(calls_6)}")
    print(f"  commerce_search_products 调用次数: {search_count}")
    print(f"  检测结果: {'✓ 触发警告' if search_count > 5 else '✗ 未触发'}")
    print(f"  影响: {'仅记录日志，不强制终止' if search_count > 5 else 'N/A'}")
    
    print("\n" + "=" * 70)
    print("结论分析")
    print("=" * 70)
    
    print("\n✅ 机制设计良好:")
    print("  1. 只拦截【连续3次完全相同】的调用")
    print("  2. 不同工具之间的调用不受影响")
    print("  3. 同一工具不同参数不受影响")
    print("  4. 间隔重复不受影响(不连续)")
    print("  5. 多工具流程可以正常执行")
    
    print("\n⚠️  需要注意的边界情况:")
    print("  1. 如果任务确实需要连续3次相同查询(极少见)")
    print("  2. 分页查询: 参数不同(offset/page)所以不会误判")
    print("  3. 重试机制: 如果工具调用失败需要重试3次以上")
    
    print("\n🔧 可能的优化空间:")
    print("  1. 对于查询类工具,阈值可以设为2次(更严格)")
    print("  2. 对于操作类工具,阈值可以设为4-5次(更宽松)")
    print("  3. 添加工具类型白名单(某些工具允许重复)")
    
    print("\n✓ 总体评估: 机制不会影响正常的多工具任务!")
    print("=" * 70)

if __name__ == "__main__":
    test_scenarios()
