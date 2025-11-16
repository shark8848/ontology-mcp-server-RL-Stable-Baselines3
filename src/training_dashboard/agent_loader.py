"""Utilities to apply trained models to the production agent."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

from agent.logger import get_logger

from .config import TrainingDashboardConfig
from .model_registry import ModelEntry

LOGGER = get_logger(__name__)


class AgentLoader:
    def __init__(self, config: TrainingDashboardConfig):
        self.config = config
        self.active_dir = Path(self.config.model_registry.active_model_dir)
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.touch_file = (
            Path(self.config.agent_loader.touch_file)
            if self.config.agent_loader.touch_file
            else self.active_dir / "last_update.txt"
        )

    def apply_model(self, entry: ModelEntry, variant: str = "best") -> Path:
        source: Optional[Path]
        if variant == "final":
            source = entry.final_model or entry.best_model
        else:
            source = entry.best_model or entry.final_model
        if not source or not source.exists():
            raise FileNotFoundError("所选模型文件不存在")

        target = self.active_dir / source.name
        shutil.copy2(source, target)
        meta = {
            "model_id": entry.model_id,
            "variant": variant,
            "source": str(source),
            "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (self.active_dir / "active_model.json").write_text(
            json_dumps(meta), encoding="utf-8"
        )
        self.touch_file.parent.mkdir(parents=True, exist_ok=True)
        self.touch_file.write_text(meta["applied_at"], encoding="utf-8")
        LOGGER.info("模型已同步到 Agent 目录: %s", target)
        return target


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)
