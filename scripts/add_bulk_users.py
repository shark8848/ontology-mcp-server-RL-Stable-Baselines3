#!/usr/bin/env python3
"""批量向电商数据库插入 200 名用户用于测试/训练场景。"""

import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from ontology_mcp_server.db_service import EcommerceService
from ontology_mcp_server.models import User  # type: ignore

# 20 个常见姓氏 + 对应拼音，结合 10 组名字即可覆盖 200 个组合
SURNAMES: List[Tuple[str, str]] = [
    ("赵", "zhao"),
    ("钱", "qian"),
    ("孙", "sun"),
    ("李", "li"),
    ("周", "zhou"),
    ("吴", "wu"),
    ("郑", "zheng"),
    ("王", "wang"),
    ("冯", "feng"),
    ("陈", "chen"),
    ("褚", "chu"),
    ("卫", "wei"),
    ("蒋", "jiang"),
    ("沈", "shen"),
    ("韩", "han"),
    ("杨", "yang"),
    ("朱", "zhu"),
    ("秦", "qin"),
    ("尤", "you"),
    ("许", "xu"),
]

GIVEN_NAMES: List[Tuple[str, str]] = [
    ("佳怡", "jiayi"),
    ("子墨", "zimo"),
    ("梓萱", "zixuan"),
    ("晨曦", "chenxi"),
    ("思远", "siyuan"),
    ("雅静", "yajing"),
    ("浩然", "haoran"),
    ("思辰", "sichen"),
    ("俊杰", "junjie"),
    ("雨桐", "yutong"),
]

EMAIL_DOMAINS = ["eshopper.cn", "retailhub.com", "vipmail.cn", "smartmall.ai"]
USER_LEVEL_SEQUENCE = [
    "Regular",
    "VIP",
    "Regular",
    "SVIP",
    "Enterprise",
    "VIP",
]
LEVEL_RULES: Dict[str, Dict[str, int]] = {
    "Regular": {"base": 180, "step": 90, "credit": 620},
    "VIP": {"base": 3200, "step": 140, "credit": 720},
    "SVIP": {"base": 7600, "step": 220, "credit": 780},
    "Enterprise": {"base": 15000, "step": 320, "credit": 820},
}


def _compose_name(idx: int) -> Tuple[str, str]:
    surname, surname_slug = SURNAMES[idx % len(SURNAMES)]
    given, given_slug = GIVEN_NAMES[(idx // len(SURNAMES)) % len(GIVEN_NAMES)]
    return f"{surname}{given}", f"{surname_slug}{given_slug}"


def generate_user_payloads(total: int = 200) -> List[Dict[str, object]]:
    """生成结构化的用户参数集合。"""

    payloads: List[Dict[str, object]] = []
    base_date = datetime.now() - timedelta(days=540)

    for idx in range(total):
        username, slug = _compose_name(idx)
        level = USER_LEVEL_SEQUENCE[idx % len(USER_LEVEL_SEQUENCE)]
        level_rule = LEVEL_RULES[level]
        email_domain = EMAIL_DOMAINS[idx % len(EMAIL_DOMAINS)]

        payloads.append(
            {
                "username": username,
                "email": f"{slug}{idx + 1:03d}@{email_domain}",
                "phone": str(13888000000 + idx),
                "user_level": level,
                "total_spent": Decimal(str(level_rule["base"] + (idx % 20) * level_rule["step"])),
                "credit_score": level_rule["credit"] + (idx % 5) * 3,
                "registration_date": base_date + timedelta(days=idx % 540),
            }
        )

    assert len(payloads) == total, "用户生成数量与期望不一致"
    return payloads


def insert_users(service: EcommerceService, payloads: List[Dict[str, object]]) -> Tuple[int, int]:
    """批量写入用户，返回 (新增数量, 跳过数量)。"""

    inserted = 0
    skipped = 0
    with service.db.get_session() as session:
        for payload in payloads:
            exists = (
                session.query(User.user_id)
                .filter(User.username == payload["username"])
                .first()
            )
            if exists:
                skipped += 1
                continue

            user = User(**payload)  # type: ignore[arg-type]
            session.add(user)
            inserted += 1
    return inserted, skipped


def main():
    data_dir = os.environ.get("ONTOLOGY_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
    db_path = os.path.join(data_dir, "ecommerce.db")
    print(f"👥 向 {db_path} 批量插入用户...")

    service = EcommerceService(db_path=db_path)
    payloads = generate_user_payloads()
    inserted, skipped = insert_users(service, payloads)

    print("\n" + "=" * 60)
    print(f"✅ 新增用户: {inserted} 名")
    print(f"↩️ 已存在跳过: {skipped} 名")
    print("=" * 60)


if __name__ == "__main__":
    main()
