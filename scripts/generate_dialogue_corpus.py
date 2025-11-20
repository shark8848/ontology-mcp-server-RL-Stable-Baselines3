#!/usr/bin/env python3
"""Generate 200 dialogue scenarios backed by real catalog, users, and orders."""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.environ.get("ONTOLOGY_DATA_DIR", PROJECT_ROOT / "data")
DB_PATH = Path(DATA_DIR) / "ecommerce.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training_scenarios" / "sample_dialogues.json"

TOTAL_SCENARIOS = 200
CATEGORY_PLAN = [
    ("transaction_success", 80),
    ("consultation", 40),
    ("issue", 30),
    ("customer_service", 30),
    ("return", 20),
]
RNG = random.Random(2025)

NEEDS = [
    "今天锁定库存",
    "附带企业发票",
    "加上延保",
    "安排同城加急",
    "同步采购合同",
    "追加官方配件",
    "升级到旗舰配置",
    "加上刻字标识",
    "套用教育折扣",
]
SUPPORT_NOTES = [
    "门卫代收",
    "加包装防撞",
    "备注勿敲门",
    "保密送达",
    "货到前电话联系",
]
RETURNS_REASONS = [
    "型号不符",
    "包装破损",
    "希望升级配置",
    "键盘缺失",
    "系统语言设置错误",
]


@dataclass
class UserRow:
    user_id: Optional[int]
    username: str
    phone: str
    user_level: str
    is_real: bool = True


@dataclass
class ProductRow:
    product_id: int
    product_name: str
    brand: str
    category: str
    price: float


@dataclass
class OrderItemRow:
    order_id: int
    product: ProductRow
    quantity: int
    unit_price: float
    subtotal: float


@dataclass
class OrderRow:
    order_id: int
    order_no: str
    user: UserRow
    total_amount: float
    discount_amount: float
    final_amount: float
    order_status: str
    payment_status: str
    shipping_address: str
    contact_phone: str
    created_at: Optional[str]
    paid_at: Optional[str]
    shipped_at: Optional[str]
    delivered_at: Optional[str]
    items: List[OrderItemRow] = field(default_factory=list)


def load_users(conn: sqlite3.Connection) -> List[UserRow]:
    cur = conn.execute(
        "SELECT user_id, username, IFNULL(phone,''), IFNULL(user_level,'Regular') FROM users ORDER BY user_id"
    )
    users: List[UserRow] = []
    for row in cur.fetchall():
        phone = row[2] or f"139{row[0]:08d}"
        users.append(UserRow(user_id=row[0], username=row[1], phone=phone, user_level=row[3], is_real=True))
    if not users:
        raise RuntimeError("用户表为空，无法生成真实语料")
    return users


def load_products(conn: sqlite3.Connection) -> Tuple[Dict[str, List[ProductRow]], Dict[int, ProductRow]]:
    cur = conn.execute(
        "SELECT product_id, product_name, brand, category, price FROM products WHERE is_available = 1"
    )
    buckets: Dict[str, List[ProductRow]] = {}
    by_id: Dict[int, ProductRow] = {}
    for pid, name, brand, category, price in cur.fetchall():
        product = ProductRow(
            product_id=pid,
            product_name=name,
            brand=brand or "",
            category=category or "其他",
            price=float(price or 0),
        )
        buckets.setdefault(product.category, []).append(product)
        by_id[pid] = product
    if not buckets:
        raise RuntimeError("商品表为空，无法生成语料")
    return buckets, by_id


def load_orders(
    conn: sqlite3.Connection,
    user_lookup: Dict[int, UserRow],
    product_lookup: Dict[int, ProductRow],
) -> List[OrderRow]:
    cur = conn.execute(
        """
        SELECT order_id, order_no, user_id, total_amount, discount_amount, final_amount,
               IFNULL(order_status,''), IFNULL(payment_status,''),
               IFNULL(shipping_address,''), IFNULL(contact_phone,''),
               created_at, paid_at, shipped_at, delivered_at
        FROM orders
        ORDER BY created_at DESC
        """
    )
    orders: Dict[int, OrderRow] = {}
    for row in cur.fetchall():
        user = user_lookup.get(row[2])
        if not user:
            continue
        orders[row[0]] = OrderRow(
            order_id=row[0],
            order_no=row[1],
            user=user,
            total_amount=float(row[3] or 0),
            discount_amount=float(row[4] or 0),
            final_amount=float(row[5] or 0),
            order_status=row[6],
            payment_status=row[7],
            shipping_address=row[8],
            contact_phone=row[9] or user.phone,
            created_at=row[10],
            paid_at=row[11],
            shipped_at=row[12],
            delivered_at=row[13],
        )

    if not orders:
        raise RuntimeError("订单表为空，无法生成真实语料")

    item_cur = conn.execute(
        """
        SELECT oi.order_id, oi.product_id, oi.product_name,
               COALESCE(p.brand, ''), COALESCE(p.category, '其他'),
               COALESCE(p.price, oi.unit_price) AS price,
               oi.quantity, oi.unit_price, oi.subtotal
        FROM order_items oi
        LEFT JOIN products p ON oi.product_id = p.product_id
        """
    )
    for row in item_cur.fetchall():
        order = orders.get(row[0])
        if not order:
            continue
        product = product_lookup.get(row[1])
        if not product:
            product = ProductRow(
                product_id=row[1] or -1,
                product_name=row[2],
                brand=row[3],
                category=row[4],
                price=float(row[5] or row[7] or 0),
            )
        order.items.append(
            OrderItemRow(
                order_id=row[0],
                product=product,
                quantity=int(row[6] or 1),
                unit_price=float(row[7] or product.price),
                subtotal=float(row[8] or 0),
            )
        )

    enriched_orders = [order for order in orders.values() if order.items]
    if not enriched_orders:
        raise RuntimeError("订单缺少商品明细，无法生成完整场景")
    return enriched_orders


def random_product(products: Dict[str, List[ProductRow]]) -> ProductRow:
    category = RNG.choice(list(products.keys()))
    return RNG.choice(products[category])


def derive_tracking_no(order: OrderRow, scenario_idx: int) -> str:
    digits = "".join(ch for ch in order.order_no if ch.isdigit()) or f"{order.order_id:08d}"
    seed = (int(digits[-8:]) + scenario_idx * 13) % 100000000
    return f"SF{seed:08d}"


def derive_ticket_no(order: OrderRow, scenario_idx: int) -> str:
    return f"TCK{order.order_id:04d}{scenario_idx:03d}"


def derive_return_no(order: OrderRow, scenario_idx: int) -> str:
    return f"RTN{order.order_id:04d}{scenario_idx:03d}"


def format_amount(amount: float) -> str:
    return f"¥{amount:,.2f}"


def build_persona(user: UserRow, product: object, category: str) -> str:
    if isinstance(product, dict):
        brand = product.get("brand", "")
        name = product.get("product_name", "")
    else:
        brand = getattr(product, "brand", "")
        name = getattr(product, "product_name", "")
    return f"{user.username}（{user.user_level}）关注 {brand}{name} 的 {category} 场景"


def transaction_steps(order: OrderRow, item: OrderItemRow, scenario_idx: int) -> List[Dict[str, str]]:
    user = order.user
    need = RNG.choice(NEEDS)
    accessory_note = RNG.choice(["官方延保", "定制贴膜", "智能配件套装"])
    tracking = derive_tracking_no(order, scenario_idx)
    discount = format_amount(order.discount_amount)
    final_amount_value = order.final_amount or order.total_amount
    final_amount = format_amount(final_amount_value)
    date_label = order.created_at.split(" ")[0] if order.created_at else datetime.now().strftime("%Y-%m-%d")
    address = order.shipping_address or "客户默认地址"
    return [
        {
            "role": "user",
            "content": (
                f"我是{user.username}（{user.user_level}），订单 {order.order_no} 想确认 {item.quantity} 台"
                f" {item.product.brand}{item.product.product_name} 是否已经按 {need} 备注。"
            ),
        },
        {
            "role": "agent",
            "content": (
                f"采购单已审批，通过 {discount} 折扣后实付 {final_amount}，"
                f"我也同步了 {need} 和增值税专票抬头。"
            ),
        },
        {
            "role": "agent",
            "content": (
                f"另外把 {accessory_note} 一并加进订单，物流生成 {tracking}，"
                f"送货地址按 {address} 执行。"
            ),
        },
        {
            "role": "user",
            "content": "明白了，付款通知我已经收到，麻烦保持库存。",
        },
        {
            "role": "agent",
            "content": f"好的，系统显示 {date_label} 内会推送装箱照片给您确认。",
        },
    ]


def consultation_steps(user: UserRow, product_a: ProductRow, product_b: ProductRow) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                f"我叫{user.username}，正在比较 {product_a.brand}{product_a.product_name} 和"
                f" {product_b.brand}{product_b.product_name}，主要看续航和售后。"
            ),
        },
        {
            "role": "agent",
            "content": (
                f"{product_a.product_name} 续航约 20 小时，{product_b.product_name} 支持 120W 充电，"
                f"我整理了对比表并发到您的手机 {user.phone}，同时锁定企业折扣名额。"
            ),
        },
        {
            "role": "user",
            "content": "如果后续要批量采购，能否预留测试机？",
        },
        {
            "role": "agent",
            "content": "可以，我预约了旗舰店试用并保留 48 小时采购优惠。",
        },
    ]


def issue_steps(order: OrderRow, item: OrderItemRow, scenario_idx: int) -> List[Dict[str, str]]:
    user = order.user
    ticket = derive_ticket_no(order, scenario_idx)
    phone = order.contact_phone or user.phone
    return [
        {
            "role": "user",
            "content": (
                f"订单 {order.order_no} 付款提示失败，我是 {user.username}，电话 {phone}，"
                f"想确认 {item.product.product_name} 的库存是否还能保留。"
            ),
        },
        {
            "role": "agent",
            "content": (
                f"检测到支付状态为 {order.payment_status or '未付款'}，"
                f"我已重新发送验证码并延长锁定时间 30 分钟。"
            ),
        },
        {
            "role": "user",
            "content": "如果再次失败，可以改走对公转账或分期吗？",
        },
        {
            "role": "agent",
            "content": (
                f"可以，我为您创建了客服工单 {ticket}，"
                "附上对公账户和分次支付流程，短信稍后推送。"
            ),
        },
    ]


def customer_service_steps(order: OrderRow, item: OrderItemRow, scenario_idx: int) -> List[Dict[str, str]]:
    note = RNG.choice(SUPPORT_NOTES)
    tracking = derive_tracking_no(order, scenario_idx)
    eta = order.shipped_at or order.created_at or "今日"
    return [
        {
            "role": "user",
            "content": (
                f"订单 {order.order_no} 的 {item.product.product_name} 是否已经出库？"
                f"我想在物流备注 {note}。"
            ),
        },
        {
            "role": "agent",
            "content": (
                f"系统显示已安排出库，物流单号 {tracking}，预计 {eta[:16]} 装车，"
                f"我已写入“{note}”。"
            ),
        },
        {
            "role": "user",
            "content": "配件可以和主机一起发货并保价吗？",
        },
        {
            "role": "agent",
            "content": "已合单并开启全额保价，费用将在账单中自动更新。",
        },
    ]


def return_steps(order: OrderRow, item: OrderItemRow, scenario_idx: int) -> List[Dict[str, str]]:
    user = order.user
    reason = RNG.choice(RETURNS_REASONS)
    return_id = derive_return_no(order, scenario_idx)
    return [
        {
            "role": "user",
            "content": (
                f"我是 {user.username}，订单 {order.order_no} 的 {item.product.product_name} 收到后发现 {reason}，"
                "想申请退货或升级版本。"
            ),
        },
        {
            "role": "agent",
            "content": (
                f"已创建退货单 {return_id}，顺丰将在 24 小时内上门，检测通过后可直接换成旗舰配置。"
            ),
        },
        {
            "role": "user",
            "content": "补差价走原支付方式即可，麻烦写到工单里。",
        },
        {
            "role": "agent",
            "content": "好的，财务会在 3 个工作日内完成差额结算，并邮件抄送给您。",
        },
    ]


CATEGORY_BUILDERS = {
    "transaction_success": transaction_steps,
    "consultation": consultation_steps,
    "issue": issue_steps,
    "customer_service": customer_service_steps,
    "return": return_steps,
}


def build_scenario(
    category: str,
    idx: int,
    users: List[UserRow],
    products: Dict[str, List[ProductRow]],
    orders: List[OrderRow],
) -> Dict[str, object]:
    if category == "consultation":
        user = RNG.choice(users)
        prod_a = random_product(products)
        prod_b = random_product(products)
        while prod_b.product_id == prod_a.product_id:
            prod_b = random_product(products)
        steps = consultation_steps(user, prod_a, prod_b)
        product_meta: object = [dict(prod_a.__dict__), dict(prod_b.__dict__)]
        persona_source = product_meta[0]
        order_meta = None
    else:
        order = RNG.choice(orders)
        item = RNG.choice(order.items)
        builder = CATEGORY_BUILDERS[category]
        steps = builder(order, item, idx)
        product_meta = {
            **item.product.__dict__,
            "order_quantity": item.quantity,
            "order_unit_price": item.unit_price,
        }
        user = order.user
        persona_source = product_meta
        order_meta = {
            "order_id": order.order_id,
            "order_no": order.order_no,
            "total_amount": order.total_amount,
            "discount_amount": order.discount_amount,
            "final_amount": order.final_amount,
            "order_status": order.order_status,
            "payment_status": order.payment_status,
            "shipping_address": order.shipping_address,
            "contact_phone": order.contact_phone,
            "created_at": order.created_at,
            "paid_at": order.paid_at,
            "shipped_at": order.shipped_at,
            "delivered_at": order.delivered_at,
            "items": [
                {
                    "product_id": it.product.product_id,
                    "product_name": it.product.product_name,
                    "brand": it.product.brand,
                    "category": it.product.category,
                    "quantity": it.quantity,
                    "unit_price": it.unit_price,
                    "subtotal": it.subtotal,
                }
                for it in order.items
            ],
        }

    scenario = {
        "name": f"{category}_{idx:04d}",
        "category": category,
        "persona": build_persona(user, persona_source, category),
        "is_real_data": True,
        "metadata": {
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "phone": user.phone,
                "user_level": user.user_level,
            },
            "order": order_meta,
            "products": product_meta,
        },
        "steps": steps,
    }
    return scenario


def generate_corpus(output: Path) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        user_rows = load_users(conn)
        product_catalog, product_lookup = load_products(conn)
        user_lookup = {user.user_id: user for user in user_rows if user.user_id is not None}
        order_rows = load_orders(conn, user_lookup, product_lookup)
    finally:
        conn.close()

    scenarios: List[Dict[str, object]] = []

    for category, count in CATEGORY_PLAN:
        for _ in range(count):
            scenario = build_scenario(
                category,
                len(scenarios) + 1,
                user_rows,
                product_catalog,
                order_rows,
            )
            scenarios.append(scenario)

    header = {
        "version": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "description": "基于真实用户/订单/商品生成的 200 条电商完整服务语料。",
        "summary": {
            "total": len(scenarios),
            "categories": Counter([s["category"] for s in scenarios]),
        },
        "scenarios": scenarios,
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False, indent=2)

    print(f"生成语料 {len(scenarios)} 组 -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成电商对话语料")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSON 文件路径")
    args = parser.parse_args()
    generate_corpus(args.output)


if __name__ == "__main__":
    main()
