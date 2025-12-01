#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物流自动创建工作流测试
测试从订单创建到物流查询的完整流程
"""

import sys
sys.path.insert(0, 'src')

from ontology_mcp_server.commerce_service import CommerceService

def test_shipment_workflow():
    """测试完整的物流工作流"""
    print("=" * 60)
    print("测试物流自动创建工作流")
    print("=" * 60)
    
    service = CommerceService(db_path="data/ecommerce.db")
    
    # 步骤1: 创建订单
    print("\n步骤1: 创建订单...")
    result = service.create_order(
        user_id=1,
        items=[{
            "product_id": 1,
            "quantity": 1
        }],
        shipping_address="北京市朝阳区测试路123号",
        contact_phone="13800138000"
    )
    
    order_id = result["order"]["order_id"]
    print(f"✓ 订单创建成功: order_id={order_id}")
    print(f"  - 订单号: {result['order']['order_no']}")
    print(f"  - 最终金额: {result['order']['final_amount']}")
    
    # 步骤2: 查询物流状态
    print(f"\n步骤2: 查询物流状态 (order_id={order_id})...")
    shipment = service.get_shipment_status(order_id)
    
    if not shipment:
        print("✗ 未找到物流信息")
        return False
    
    print("✓ 物流信息查询成功:")
    print(f"  - 运单号: {shipment['tracking_no']}")
    print(f"  - 承运商: {shipment['carrier']}")
    print(f"  - 当前状态: {shipment['current_status']}")
    print(f"  - 预计送达: {shipment['estimated_delivery']}")
    print(f"  - 物流轨迹数: {len(shipment['tracks'])}")
    
    # 步骤3: 验证关键字段
    print("\n步骤3: 验证关键字段...")
    checks = [
        (shipment['tracking_no'].startswith('SF'), "运单号格式检查"),
        (shipment['carrier'] == "顺丰速运", "承运商检查"),
        (shipment['current_status'] == "待揽收", "初始状态检查"),
        (shipment['estimated_delivery'] is not None, "预计送达时间检查"),
    ]
    
    all_passed = True
    for check, desc in checks:
        if check:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc}")
            all_passed = False
    
    # 步骤4: 测试使用运单号查询
    print("\n步骤4: 使用运单号查询物流...")
    tracking_no = shipment['tracking_no']
    shipment2 = service.track_shipment(tracking_no)
    
    if not shipment2:
        print(f"✗ 使用运单号 {tracking_no} 查询失败")
        return False
    
    print(f"✓ 运单号查询成功:")
    print(f"  - 订单ID: {shipment2['order_id']}")
    print(f"  - 运单号: {shipment2['tracking_no']}")
    
    if shipment2['order_id'] != order_id:
        print(f"✗ 订单ID不匹配: {shipment2['order_id']} != {order_id}")
        return False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    try:
        success = test_shipment_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
