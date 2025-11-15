#!/usr/bin/env python3
"""随机更新示例用户（ID 1-5）的姓名，使其更贴近真实场景。"""

import argparse
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from ontology_mcp_server.db_service import EcommerceService
from ontology_mcp_server.models import User  # type: ignore

TARGET_USER_IDS: Sequence[int] = (1, 2, 3, 4, 5)
CANDIDATE_NAMES: List[str] = [
    "顾昕怡",
    "程亦航",
    "陆芷若",
    "许嘉木",
    "宋栩宁",
    "姜思祺",
    "萧景澄",
    "唐姝言",
    "黎承睿",
    "温以沫",
    "季知意",
    "傅明昊",
    "邵意淇",
    "霍允辰",
    "褚沐晴",
]


def load_existing_usernames(service: EcommerceService) -> Tuple[Dict[int, str], set]:
    """返回 (目标用户当前姓名映射, 全量用户名集合)。"""
    with service.db.get_session() as session:
        current = (
            session.query(User.user_id, User.username)
            .filter(User.user_id.in_(TARGET_USER_IDS))
            .order_by(User.user_id)
            .all()
        )
        username_map = {user_id: username for user_id, username in current}
        all_names = {name for (name,) in session.query(User.username).all()}
    return username_map, all_names


def pick_new_names(existing: set, current_targets: Dict[int, str], seed: int | None) -> List[str]:
    """从候选池随机挑选与现有姓名不重合的新名字。"""
    rng = random.Random(seed)
    pool = CANDIDATE_NAMES.copy()
    rng.shuffle(pool)

    forbidden = existing.difference(current_targets.values())
    chosen: List[str] = []
    for candidate in pool:
        if candidate in forbidden or candidate in chosen:
            continue
        chosen.append(candidate)
        if len(chosen) == len(TARGET_USER_IDS):
            break
    if len(chosen) < len(TARGET_USER_IDS):
        raise RuntimeError("候选姓名不足以生成唯一组合，请扩充 CANDIDATE_NAMES。")
    return chosen


def apply_updates(service: EcommerceService, updates: Dict[int, Tuple[str, str]]) -> None:
    """执行数据库更新。"""
    with service.db.get_session() as session:
        for user_id, (_, new_name) in updates.items():
            user = session.query(User).filter(User.user_id == user_id).first()
            if not user:
                raise RuntimeError(f"未找到用户 ID={user_id}")
            user.username = new_name
        # session context manager会自动提交


def main() -> None:
    parser = argparse.ArgumentParser(description="随机更新示例用户姓名")
    parser.add_argument("--seed", type=int, default=None, help="可选的随机种子，便于复现")
    args = parser.parse_args()

    data_dir = os.environ.get("ONTOLOGY_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
    db_path = os.path.join(data_dir, "ecommerce.db")
    print(f"👥 更新示例用户姓名 (数据库: {db_path})")

    service = EcommerceService(db_path=db_path)
    current_map, all_names = load_existing_usernames(service)

    if len(current_map) != len(TARGET_USER_IDS):
        missing = set(TARGET_USER_IDS) - set(current_map.keys())
        raise RuntimeError(f"缺少需要更新的用户: {sorted(missing)}")

    new_names = pick_new_names(all_names, current_map, args.seed)
    updates = {uid: (current_map[uid], new_name) for uid, new_name in zip(TARGET_USER_IDS, new_names)}

    apply_updates(service, updates)

    print("\n更新结果:")
    for uid in TARGET_USER_IDS:
        old_name, new_name = updates[uid]
        print(f"  - user_id={uid}: {old_name} -> {new_name}")
    print("\n✅ 已完成示例用户姓名替换")


if __name__ == "__main__":
    main()
