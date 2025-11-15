#!/usr/bin/env python3
"""场景脚本管理与环境 wrapper."""

import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import gymnasium as gym

from agent.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ScenarioScript:
    """结构化的对话脚本，仅保留用户话术，以驱动环境多轮输入."""

    name: str
    persona: str
    user_turns: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["ScenarioScript"]:
        user_turns = [
            (step.get("content") or "").strip()
            for step in data.get("steps", [])
            if step.get("role") == "user" and (step.get("content") or "").strip()
        ]
        if not user_turns:
            return None
        return cls(
            name=data.get("name") or "scenario",
            persona=data.get("persona") or "",
            user_turns=user_turns,
        )


def load_scenario_scripts(file_path: Optional[str]) -> List[ScenarioScript]:
    """加载完整对话脚本，提取多轮用户话术."""
    if not file_path:
        return []
    resolved_path = os.path.abspath(file_path)
    if not os.path.exists(resolved_path):
        LOGGER.warning("未找到场景脚本文件: %s", resolved_path)
        return []
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        LOGGER.error("读取场景脚本失败: %s", exc)
        return []

    scripts: List[ScenarioScript] = []
    for raw in data.get("scenarios", []):
        script = ScenarioScript.from_dict(raw)
        if script:
            scripts.append(script)

    if not scripts:
        LOGGER.warning("场景脚本中未解析到可用的用户话术: %s", resolved_path)
    else:
        LOGGER.info("已加载 %d 个对话脚本: %s", len(scripts), resolved_path)
    return scripts


class ScenarioConversationWrapper(gym.Wrapper):
    """根据脚本在 episode 内逐步注入用户话术，模拟真实对话流程."""

    def __init__(
        self,
        env: gym.Env,
        scripts: List[ScenarioScript],
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self._scripts = scripts
        self._rng = random.Random(seed)
        self._active_script: Optional[ScenarioScript] = None
        self._user_idx: int = 0
        self._fallback_utterance = "谢谢，暂时就这些。"

    def _choose_script(self) -> Optional[ScenarioScript]:
        if not self._scripts:
            return None
        return self._rng.choice(self._scripts)

    def _next_user_utterance(self) -> str:
        if not self._active_script:
            return self._fallback_utterance
        self._user_idx += 1
        if self._user_idx < len(self._active_script.user_turns):
            return self._active_script.user_turns[self._user_idx]
        return self._fallback_utterance

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        options = dict(options or {})
        self._active_script = self._choose_script()
        self._user_idx = 0

        if self._active_script and not options.get("user_input"):
            options["user_input"] = self._active_script.user_turns[0]

        obs, info = self.env.reset(seed=seed, options=options)
        info = info or {}
        if self._active_script:
            info = dict(info)
            info["scenario"] = self._active_script.name
            info["scenario_step"] = 1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not (terminated or truncated) and self._active_script:
            next_utterance = self._next_user_utterance()
            setattr(self.env, "current_user_input", next_utterance)

        info = info or {}
        if self._active_script:
            info = dict(info)
            info.setdefault("scenario", self._active_script.name)
            info["scenario_step"] = min(self._user_idx + 1, len(self._active_script.user_turns))
        return obs, reward, terminated, truncated, info
