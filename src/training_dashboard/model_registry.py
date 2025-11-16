"""Model registry helper for managing RL checkpoints."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from agent.logger import get_logger

from .config import TrainingDashboardConfig

LOGGER = get_logger(__name__)


@dataclass
class ModelEntry:
    model_id: str
    created_at: float
    metadata_path: Path
    best_model: Optional[Path]
    final_model: Optional[Path]

    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "created_at": self.created_at,
            "metadata_path": str(self.metadata_path),
            "best_model": str(self.best_model) if self.best_model else None,
            "final_model": str(self.final_model) if self.final_model else None,
        }


class ModelRegistry:
    def __init__(self, config: TrainingDashboardConfig):
        self.config = config
        self.root = Path(self.config.model_registry.registry_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "index.json"

    def register_run(self, session_info: Dict, output_dir: Path) -> Optional[str]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_dir = self.root / timestamp
        model_dir.mkdir(parents=True, exist_ok=True)

        best_source = output_dir / "best_model" / "best_model.zip"
        final_source = output_dir / "models" / "ppo_ecommerce_final.zip"

        best_target = model_dir / "best_model.zip" if best_source.exists() else None
        final_target = model_dir / "final_model.zip" if final_source.exists() else None

        if best_source.exists():
            shutil.copy2(best_source, best_target)
        if final_source.exists():
            shutil.copy2(final_source, final_target)

        metadata = {
            "created_at": timestamp,
            "session": session_info,
            "best_model": str(best_target) if best_target else None,
            "final_model": str(final_target) if final_target else None,
        }
        metadata_path = model_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        model_id = timestamp
        entry = ModelEntry(
            model_id=model_id,
            created_at=time.time(),
            metadata_path=metadata_path,
            best_model=best_target,
            final_model=final_target,
        )
        self._persist_entry(entry)
        LOGGER.info("模型已登记: %s", model_id)
        return model_id

    def list_models(self) -> List[ModelEntry]:
        entries = []
        if not self.index_file.exists():
            return entries
        try:
            data = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception:
            return entries
        for row in data:
            entries.append(
                ModelEntry(
                    model_id=row["model_id"],
                    created_at=row["created_at"],
                    metadata_path=Path(row["metadata_path"]),
                    best_model=Path(row["best_model"]) if row.get("best_model") else None,
                    final_model=Path(row["final_model"]) if row.get("final_model") else None,
                )
            )
        return entries

    def get_entry(self, model_id: str) -> Optional[ModelEntry]:
        for entry in self.list_models():
            if entry.model_id == model_id:
                return entry
        return None

    def _persist_entry(self, entry: ModelEntry) -> None:
        entries = [row for row in self.list_models() if row.model_id != entry.model_id]
        entries.append(entry)
        entries.sort(key=lambda e: e.created_at, reverse=True)
        payload = [row.to_dict() for row in entries]
        self.index_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
