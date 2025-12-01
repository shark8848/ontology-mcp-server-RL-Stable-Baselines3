#!/usr/bin/env python3
"""测试订单创建后自动生成物流信息"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from decimal import Decimal
from src.ontology_mcp_server.commerce_service import CommerceService


def test_auto_shipment_creation():
    """验证创建订单后自动生成物流记录"""
    service = CommerceService(db_path="data/ecommerce.db")
    
    # 创建测试订单
    user_id = 1
    items = [
        {"product_id": 1, "quantity": 1, "unit_price": 5999.00}
    ]
    shipping_address = "北京市朝阳区测试街道123号"
    contact_phone = "13800138000"
    
    print("正在创建订单...")
    result = service.create_order(
        user_id=user_id,
        items=items,
        shipping_address=shipping_address,
        contact_phone=contact_phone
    )
    
    order_id = result["order"]["order_id"]
    print(f"✓ 订单创建成功: order_id={order_id}")
    
    # 验证物流信息是否自动生成
    print("检查物流信息...")
    try:
        shipment = service.get_shipment_status(order_id)
        print(f"✓ 物流信息已自动生成:")
        print(f"  - 运单号: {shipment['tracking_no']}")
        print(f"  - 承运商: {shipment['carrier']}")
        print(f"  - 当前状态: {shipment['current_status']}")
        print(f"  - 预计送达: {shipment['estimated_delivery']}")
        return True
    except ValueError as e:
        print(f"✗ 物流信息未生成: {e}")
        return False


if __name__ == "__main__":
    success = test_auto_shipment_creation()
    sys.exit(0 if success else 1)
