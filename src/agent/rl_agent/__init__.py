"""
Copyright (c) 2025 shark8848
MIT License

强化学习优化模块

使用 Stable Baselines3 对电商 AI 助手进行持续自我优化：
- 状态提取：从对话上下文提取特征向量
- 动作空间：工具选择和参数生成
- 奖励函数：基于任务完成度、效率、满意度的多目标优化
- 在线学习：从真实用户交互中持续改进
"""

from .state_extractor import StateExtractor
from .reward_calculator import RewardCalculator, RewardComponents
from .gym_env import EcommerceGymEnv

__all__ = [
    "StateExtractor",
    "RewardCalculator",
    "RewardComponents",
    "EcommerceGymEnv",
]
