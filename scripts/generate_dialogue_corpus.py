#!/usr/bin/env python3
"""Generate >=200 dialogue scenarios referencing real catalog & users."""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.environ.get("ONTOLOGY_DATA_DIR", PROJECT_ROOT / "data")
DB_PATH = Path(DATA_DIR) / "ecommerce.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training_scenarios" / "sample_dialogues.json"

TOTAL_SCENARIOS = 220
REAL_RATIO = 0.65
CATEGORY_PLAN = [
    ("transaction_success", 120),
    ("consultation", 30),
    ("issue", 25),
    ("customer_service", 25),
    ("return", 20),
]
RNG = random.Random(2025)

NEEDS = [
    "今天锁定库存", "附带企业发票", "加上延保", "安排同城加急", "同步采购合同",
    "追加官方配件", "升级到旗舰配置", "加上刻字标识", "套用教育折扣",
]
SUPPORT_NOTES = [
    "门卫代收", "加包装防撞", "备注勿敲门", "保密送达", "货到前电话联系",
]
RETURNS_REASONS = [
    "型号不符", "包装破损", "希望升级配置", "键盘缺失", "系统语言设置错误",
]
SYNTHETIC_NAMES = [
    "林栀遥", "白知越", "宋以墨", "容安雅", "岑曜西", "霍清野", "程野舟",
    "安鹿黎", "许聞珩", "陆见深", "苏砚笙", "顾意洲", "叶知砚", "唐知砾",
]
USER_LEVELS = ["Regular", "VIP", "SVIP", "Enterprise"]


@dataclass
class UserRow:
    user_id: Optional[int]
    username: str
    phone: str
    user_level: str
    is_real: bool


@dataclass
class ProductRow:
    product_id: int
    product_name: str
    brand: str
    category: str


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


def load_products(conn: sqlite3.Connection) -> Dict[str, List[ProductRow]]:
    cur = conn.execute(
        "SELECT product_id, product_name, brand, category FROM products WHERE is_available = 1"
    )
    buckets: Dict[str, List[ProductRow]] = {}
    for pid, name, brand, category in cur.fetchall():
        buckets.setdefault(category or "其他", []).append(
            ProductRow(product_id=pid, product_name=name, brand=brand or "", category=category or "其他")
        )
    if not buckets:
        raise RuntimeError("商品表为空，无法生成语料")
    return buckets


def synthetic_user(counter: int) -> UserRow:
    name = RNG.choice(SYNTHETIC_NAMES)
    phone = f"199{counter:08d}"[-11:]
    level = RNG.choice(USER_LEVELS)
    return UserRow(user_id=None, username=name, phone=phone, user_level=level, is_real=False)


def random_product(products: Dict[str, List[ProductRow]]) -> ProductRow:
    category = RNG.choice(list(products.keys()))
    return RNG.choice(products[category])


def pick_product_by_category(products: Dict[str, List[ProductRow]], desired: Sequence[str]) -> ProductRow:
    for cat in desired:
        if cat in products and products[cat]:
            return RNG.choice(products[cat])
    return random_product(products)


def random_order_no(user_id: Optional[int], idx: int) -> str:
    base = datetime.now() - timedelta(days=RNG.randint(0, 45))
    tag = user_id if user_id is not None else RNG.randint(1000, 9999)
    return f"ORD{base.strftime('%Y%m%d%H%M%S')}{tag:04d}{idx:03d}"


def random_tracking_no() -> str:
    return f"SF{RNG.randint(2025100000, 2025999999)}"


def random_ticket_no(idx: int) -> str:
    return f"TCK{datetime.now().strftime('%m%d')}{idx:04d}"


def build_persona(user: UserRow, product: object, category: str) -> str:
    if isinstance(product, dict):
        brand = product.get("brand", "")
        name = product.get("product_name", "")
    else:
        brand = getattr(product, "brand", "")
        name = getattr(product, "product_name", "")
    return f"{user.username}（{user.user_level}）关注 {brand}{name} 的 {category} 场景"


def transaction_steps(user: UserRow, product: ProductRow, order_no: str) -> List[Dict[str, str]]:
    quantity = RNG.randint(5, 30)
    need = RNG.choice(NEEDS)
    accessory_note = RNG.choice(["官方保护壳", "氮化镓充电器", "智能手写笔", "企业延保"])
    return [
        {
            "role": "user",
            "content": f"我是{user.username}，电话{user.phone}，想一次性采购{quantity}台{product.brand}{product.product_name}，需要{need}。",
        },
        {
            "role": "agent",
            "content": f"已为您锁定{quantity}台库存，并核算出{user.user_level}折扣，预计两小时内完成审批。",
        },
        {
            "role": "agent",
            "content": f"额外赠送{accessory_note}，并同步发票抬头。请留意支付链接。",
        },
        {
            "role": "user",
            "content": "折扣和附件都没问题，麻烦一并加上。",
        },
        {
            "role": "agent",
            "content": f"订单 {order_no} 已生成，稍后短信推送物流单号 {random_tracking_no()}。",
        },
    ]


def consultation_steps(user: UserRow, product_a: ProductRow, product_b: ProductRow) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": f"我叫{user.username}，正在比较 {product_a.brand}{product_a.product_name} 和 {product_b.brand}{product_b.product_name}，主要看续航和售后。",
        },
        {
            "role": "agent",
            "content": f"{product_a.product_name} 续航 20 小时，{product_b.product_name} 支持 120W 充电，我整理了对比表发到您手机 {user.phone}。",
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


def issue_steps(user: UserRow, product: ProductRow, order_no: str) -> List[Dict[str, str]]:
    ticket = random_ticket_no(RNG.randint(1, 9999))
    return [
        {
            "role": "user",
            "content": f"订单 {order_no} 付款提示失败，我是 {user.username}，手机 {user.phone}，可以查一下吗？",
        },
        {
            "role": "agent",
            "content": f"检测到银行卡 OTP 超时，我已重新推送验证码，并把 {product.product_name} 库存延长 30 分钟。",
        },
        {
            "role": "user",
            "content": "如果再次失败，可以改走对公转账吗？",
        },
        {
            "role": "agent",
            "content": f"完全可以。我创建了客服工单 {ticket} 并准备备用收款方式，稍后短信会同步。",
        },
    ]


def customer_service_steps(user: UserRow, product: ProductRow, order_no: str) -> List[Dict[str, str]]:
    note = RNG.choice(SUPPORT_NOTES)
    tracking = random_tracking_no()
    return [
        {
            "role": "user",
            "content": f"订单 {order_no} 的 {product.product_name} 已经出库吗？我想备注 {note}。",
        },
        {
            "role": "agent",
            "content": f"已在系统中写明“{note}”，并更新物流单号 {tracking}，预计明晚送达。",
        },
        {
            "role": "user",
            "content": "能顺便把配件一起打包吗？",
        },
        {
            "role": "agent",
            "content": "没问题，已经合单并返还多余运费，稍后账单会更新。",
        },
    ]


def return_steps(user: UserRow, product: ProductRow, order_no: str) -> List[Dict[str, str]]:
    reason = RNG.choice(RETURNS_REASONS)
    return_id = f"RTN{datetime.now().strftime('%Y%m%d')}{RNG.randint(100,999)}"
    return [
        {
            "role": "user",
            "content": f"我是 {user.username}，订单 {order_no} 收到后发现 {reason}，想申请退货换货。",
        },
        {
            "role": "agent",
            "content": f"已创建退货单 {return_id}，顺丰将在 24 小时内上门，检测通过后可换同款升级版。",
        },
        {
            "role": "user",
            "content": "补差价走原支付方式即可。",
        },
        {
            "role": "agent",
            "content": "好的，财务会在 3 个工作日内完成差额结算，并邮件告知。",
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
    user: UserRow,
    products: Dict[str, List[ProductRow]],
    real: bool,
) -> Dict[str, object]:
    if category == "consultation":
        prod_a = random_product(products)
        prod_b = random_product(products)
        while prod_b.product_id == prod_a.product_id:
            prod_b = random_product(products)
        steps = consultation_steps(user, prod_a, prod_b)
        product_meta: object = [prod_a.__dict__, prod_b.__dict__]
        order_no = None
    else:
        prod = random_product(products)
        order_no = random_order_no(user.user_id, idx)
        builder = CATEGORY_BUILDERS[category]
        steps = builder(user, prod, order_no)  # type: ignore[arg-type]
        product_meta = prod.__dict__

    persona_source = product_meta[0] if isinstance(product_meta, list) else product_meta

    scenario = {
        "name": f"{category}_{idx:04d}",
        "category": category,
        "persona": build_persona(user, persona_source, category),
        "is_real_data": real,
        "metadata": {
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "phone": user.phone,
                "user_level": user.user_level,
            },
            "order_no": order_no,
            "products": product_meta,
        },
        "steps": steps,
    }
    return scenario


def generate_corpus(output: Path) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        user_rows = load_users(conn)
        product_rows = load_products(conn)
    finally:
        conn.close()

    real_target = int(TOTAL_SCENARIOS * REAL_RATIO + 0.5)
    scenarios: List[Dict[str, object]] = []
    real_assigned = 0
    synthetic_counter = 0

    for category, count in CATEGORY_PLAN:
        for _ in range(count):
            remaining = TOTAL_SCENARIOS - len(scenarios)
            need_real = real_target - real_assigned
            real_flag = need_real > 0 and (need_real >= remaining or RNG.random() < need_real / remaining)

            user = RNG.choice(user_rows) if real_flag else synthetic_user(700000 + synthetic_counter)
            if not real_flag:
                synthetic_counter += 1
            else:
                real_assigned += 1

            scenario = build_scenario(category, len(scenarios) + 1, user, product_rows, real_flag)
            scenarios.append(scenario)

    header = {
        "version": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "description": "自动生成的训练语料，含 200+ 场景，65% 来源于真实用户/电话/订单号。",
        "summary": {
            "total": len(scenarios),
            "real_ratio": real_assigned / len(scenarios),
            "categories": Counter([s["category"] for s in scenarios]),
        },
        "scenarios": scenarios,
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False, indent=2)

    print(f"生成语料 {len(scenarios)} 组 -> {output}")
    print(f"真实数据占比: {real_assigned / len(scenarios):.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成电商对话语料")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSON 文件路径")
    args = parser.parse_args()
    generate_corpus(args.output)


if __name__ == "__main__":
    main()
