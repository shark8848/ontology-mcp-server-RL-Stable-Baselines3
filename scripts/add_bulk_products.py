#!/usr/bin/env python3
"""批量向电商数据库插入 1000 条设备/配件商品数据."""

import os
import sys
from decimal import Decimal
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from ontology_mcp_server.db_service import EcommerceService
from ontology_mcp_server.models import Product  # type: ignore

BRANDS = ["Apple", "Samsung", "Lenovo", "Xiaomi", "Huawei"]

PRODUCT_TEMPLATES = [
    {
        "code": "phone",
        "category": "手机",
        "display": "旗舰手机",
        "base_price": 3499,
        "price_step": 85,
        "desc": "旗舰级影像与 5G 体验",
    },
    {
        "code": "pc",
        "category": "电脑",
        "display": "创作本",
        "base_price": 5999,
        "price_step": 120,
        "desc": "高性能个人电脑，适合商务与创作",
    },
    {
        "code": "tablet",
        "category": "平板",
        "display": "Pro 平板",
        "base_price": 2999,
        "price_step": 70,
        "desc": "轻薄大屏，支持手写与分屏多任务",
    },
    {
        "code": "accessory",
        "category": "配件",
        "display": "智能配件",
        "base_price": 199,
        "price_step": 15,
        "desc": "配套充电/保护/音频配件，完善设备体验",
    },
]

VARIANTS_PER_TEMPLATE = 50  # 5 品牌 * 4 模板 * 50 = 1000

COLORS = [
    "曜石黑",
    "冰川蓝",
    "星空银",
    "落日金",
    "云雾白",
    "墨玉绿",
]
RAM_OPTIONS = ["6GB", "8GB", "12GB", "16GB"]
STORAGE_OPTIONS = ["128GB", "256GB", "512GB", "1TB"]
ACCESSORY_TYPES = ["MagSafe 充电器", "氟硅保护壳", "TWS 耳机", "65W 氮化镓充电器", "智能手写笔"]
PC_CPU = ["Intel i7", "Intel i9", "AMD R7", "ARM M 系列"]


def _specs_for(template_code: str, idx: int) -> Dict[str, str]:
    color = COLORS[idx % len(COLORS)]
    ram = RAM_OPTIONS[idx % len(RAM_OPTIONS)]
    storage = STORAGE_OPTIONS[idx % len(STORAGE_OPTIONS)]

    if template_code == "phone":
        return {
            "color": color,
            "memory": ram,
            "storage": storage,
            "network": "5G",
        }
    if template_code == "pc":
        return {
            "cpu": PC_CPU[idx % len(PC_CPU)],
            "ram": ram.replace("GB", "") + "GB",
            "storage": f"{storage} SSD",
            "screen": "16英寸 Mini-LED",
        }
    if template_code == "tablet":
        return {
            "color": color,
            "memory": ram,
            "storage": storage,
            "display": "11英寸 120Hz",
        }
    return {
        "type": ACCESSORY_TYPES[idx % len(ACCESSORY_TYPES)],
        "color": color,
        "compatible": "多型号通用",
    }


def _image_url(brand: str, template_code: str, idx: int) -> str:
    return f"https://cdn.example.com/{brand.lower()}/{template_code}/{idx:03d}.jpg"


def generate_product_payloads() -> List[Dict[str, object]]:
    payloads: List[Dict[str, object]] = []
    for brand in BRANDS:
        for template in PRODUCT_TEMPLATES:
            for idx in range(1, VARIANTS_PER_TEMPLATE + 1):
                model_code = f"{brand[:2].upper()}{template['code'][0].upper()}{idx:03d}"
                product_name = f"{brand} {template['display']} {idx:02d}"
                price_value = template["base_price"] + template["price_step"] * idx
                payloads.append(
                    {
                        "product_name": product_name,
                        "category": template["category"],
                        "brand": brand,
                        "model": model_code,
                        "price": Decimal(str(price_value)),
                        "stock_quantity": 80 + (idx * 3) % 120,
                        "description": f"{brand}{template['display']}系列第{idx}款，{template['desc']}",
                        "specs": _specs_for(template["code"], idx),
                        "image_url": _image_url(brand, template["code"], idx),
                    }
                )
    assert len(payloads) == 1000, "生成的商品数量必须为 1000"
    return payloads


def product_exists(service: EcommerceService, name: str) -> bool:
    with service.db.get_session() as session:
        return session.query(Product.product_id).filter(Product.product_name == name).first() is not None


def main():
    data_dir = os.environ.get("ONTOLOGY_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
    db_path = os.path.join(data_dir, "ecommerce.db")
    print(f"📦 向 {db_path} 批量插入商品...")

    service = EcommerceService(db_path=db_path)
    payloads = generate_product_payloads()

    inserted = 0
    skipped = 0
    for payload in payloads:
        if product_exists(service, payload["product_name"]):
            skipped += 1
            continue
        service.products.create_product(**payload)
        inserted += 1
        if inserted % 100 == 0:
            print(f"  ✓ 已插入 {inserted} 条")

    print("\n" + "=" * 60)
    print(f"✅ 新增商品: {inserted} 条")
    print(f"↩️ 已存在跳过: {skipped} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
