"""
Copyright (c) 2025 shark8848
MIT License

Ontology MCP Server - 电商 AI 助手系统
Author: shark8848
Repository: https://github.com/shark8848/ontology-mcp-server
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from ontology_mcp_server.commerce_service import CommerceService
from ontology_mcp_server.models import Order


@pytest.fixture()
def commerce_service(tmp_path):
    db_path = tmp_path / "ecommerce_test.db"
    service = CommerceService(db_path=str(db_path))
    service.init_database()
    user = service.users.create_user("alice", email="alice@example.com")
    product = service.products.create_product(
        product_name="iPhone 15 Pro",
        category="手机",
        brand="Apple",
        model="A3100",
        price=Decimal("6999"),
        stock_quantity=10,
        description="旗舰手机",
    )
    return service, user, product


def test_search_and_order_flow(commerce_service):
    service, user, product = commerce_service

    search_result = service.search_products(keyword="iPhone")
    assert search_result["total"] >= 1

    order_payload = {
        "user_id": user.user_id,
        "items": [
            {"product_id": product.product_id, "quantity": 1, "unit_price": float(product.price)}
        ],
        "shipping_address": "上海市徐汇区漕溪北路",
        "contact_phone": "13800008888",
    }

    create_result = service.create_order(**order_payload)
    order_data = create_result["order"]
    assert order_data["order_id"] is not None
    assert create_result["inference"]["discount_inference"]["discount_type"] != "无折扣"

    # 库存应减少 1
    stock_info = service.check_stock(product.product_id, 1)
    assert stock_info["stock_quantity"] == product.stock_quantity - 1

    profile = service.get_user_profile(user.user_id)
    assert profile["inferred_level"] in {"VIP", "SVIP"}

    return_result = service.process_return(
        order_id=order_data["order_id"],
        user_id=user.user_id,
        product_category="手机",
        is_activated=False,
    )
    assert return_result["policy"]["returnable"] is True
    assert return_result["return_created"] is True


def test_get_order_detail_accepts_order_number(commerce_service):
    service, user, product = commerce_service

    order_payload = {
        "user_id": user.user_id,
        "items": [
            {"product_id": product.product_id, "quantity": 1, "unit_price": float(product.price)}
        ],
        "shipping_address": "深圳市福田区深南大道",
        "contact_phone": "13900007777",
    }

    create_result = service.create_order(**order_payload)
    order_data = create_result["order"]
    order_no = order_data["order_no"]

    detail_by_no = service.get_order_detail(order_no)
    assert detail_by_no["order"]["order_no"] == order_no

    digits_only = order_no.replace("ORD", "")
    detail_by_digits = service.get_order_detail(digits_only)
    assert detail_by_digits["order"]["order_no"] == order_no


def test_cancel_order_pending_within_window(commerce_service):
    service, user, product = commerce_service

    order_payload = {
        "user_id": user.user_id,
        "items": [
            {"product_id": product.product_id, "quantity": 1, "unit_price": float(product.price)}
        ],
        "shipping_address": "杭州市西湖区文三路",
        "contact_phone": "13600006666",
    }

    create_result = service.create_order(**order_payload)
    order_id = create_result["order"]["order_id"]

    cancel_result = service.cancel_order(order_id)

    assert cancel_result["cancelled"] is True
    assert cancel_result["policy"]["rule"] == "Pending24hCancellationRule"


def test_cancel_order_pending_after_window_denied(commerce_service):
    service, user, product = commerce_service

    order_payload = {
        "user_id": user.user_id,
        "items": [
            {"product_id": product.product_id, "quantity": 1, "unit_price": float(product.price)}
        ],
        "shipping_address": "北京市朝阳区建国路",
        "contact_phone": "13500005555",
    }

    create_result = service.create_order(**order_payload)
    order_id = create_result["order"]["order_id"]

    with service.database.get_session() as session:
        db_order = session.query(Order).filter(Order.order_id == order_id).first()
        db_order.created_at = db_order.created_at - timedelta(hours=30)
        session.commit()

    cancel_result = service.cancel_order(order_id)

    assert cancel_result["cancelled"] is False
    assert cancel_result["policy"]["allowed"] is False
    assert "24" in cancel_result["policy"]["reason"]


def test_cancel_order_paid_within_window(commerce_service):
    service, user, product = commerce_service

    order_payload = {
        "user_id": user.user_id,
        "items": [
            {"product_id": product.product_id, "quantity": 1, "unit_price": float(product.price)}
        ],
        "shipping_address": "深圳市南山区科技园",
        "contact_phone": "13400004444",
    }

    create_result = service.create_order(**order_payload)
    order_id = create_result["order"]["order_id"]

    with service.database.get_session() as session:
        db_order = session.query(Order).filter(Order.order_id == order_id).first()
        db_order.order_status = "paid"
        db_order.created_at = datetime.now() - timedelta(hours=2)
        session.commit()

    cancel_result = service.cancel_order(order_id)

    assert cancel_result["cancelled"] is True
    assert cancel_result["policy"]["rule"] == "Paid12hCancellationRule"
