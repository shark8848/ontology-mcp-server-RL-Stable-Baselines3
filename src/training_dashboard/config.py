"""Configuration loader for the RL training dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CONFIG_PATH = Path("config/training_dashboard.yaml")
EXAMPLE_CONFIG_PATH = Path("config/training_dashboard.example.yaml")


@dataclass
class CorpusConfig:
    static_paths: List[str] = field(default_factory=list)
    log_source_path: str = "src/ontology_mcp_server/logs/server.log"
    log_corpus_dir: str = "data/training_dashboard/log_corpus"
    ingest_interval_minutes: int = 60
    min_dialogues_per_batch: int = 5


@dataclass
class TrainingParams:
    timesteps: int = 50000
    eval_freq: int = 1000
    checkpoint_freq: int = 5000
    max_steps_per_episode: int = 10
    use_text_embedding: bool = False
    output_dir: str = "data/rl_training"


@dataclass
class ModelRegistryConfig:
    registry_root: str = "data/training_dashboard/models"
    active_model_dir: str = "data/rl_training/active_model"


@dataclass
class AgentLoaderConfig:
    strategy: str = "file_copy"
    touch_file: Optional[str] = None


@dataclass
class TrainingDashboardConfig:
    data_root: str = "data/training_dashboard"
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    training: TrainingParams = field(default_factory=TrainingParams)
    model_registry: ModelRegistryConfig = field(default_factory=ModelRegistryConfig)
    agent_loader: AgentLoaderConfig = field(default_factory=AgentLoaderConfig)

    @property
    def data_root_path(self) -> Path:
        return Path(self.data_root)


def _merge(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_dataclass(data: Dict[str, Any]) -> TrainingDashboardConfig:
    corpus_cfg = CorpusConfig(**data.get("corpus", {}))
    training_cfg = TrainingParams(**data.get("training", {}))
    model_cfg = ModelRegistryConfig(**data.get("model_registry", {}))
    loader_cfg = AgentLoaderConfig(**data.get("agent_loader", {}))
    return TrainingDashboardConfig(
        data_root=data.get("data_root", "data/training_dashboard"),
        corpus=corpus_cfg,
        training=training_cfg,
        model_registry=model_cfg,
        agent_loader=loader_cfg,
    )


def load_config(config_path: Optional[Path] = None) -> TrainingDashboardConfig:
    """Load dashboard config with fallback to example file."""

    path = config_path or DEFAULT_CONFIG_PATH
    base: Dict[str, Any] = _load_yaml(EXAMPLE_CONFIG_PATH)
    override: Dict[str, Any] = _load_yaml(path)
    merged = _merge(base, override)
    config = _build_dataclass(merged)

    # ensure directories exist
    Path(config.corpus.log_corpus_dir).mkdir(parents=True, exist_ok=True)
    Path(config.model_registry.registry_root).mkdir(parents=True, exist_ok=True)
    Path(config.model_registry.active_model_dir).mkdir(parents=True, exist_ok=True)
    config.data_root_path.mkdir(parents=True, exist_ok=True)
    return config
