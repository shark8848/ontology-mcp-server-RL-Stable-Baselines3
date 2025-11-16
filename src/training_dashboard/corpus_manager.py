"""Corpus aggregation and log ingestion utilities."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.logger import get_logger

from .config import TrainingDashboardConfig

LOGGER = get_logger(__name__)


@dataclass
class CorpusSummary:
    path: Path
    total: int
    description: str


class CorpusManager:
    """Manage static corpus files and log-derived corpora."""

    def __init__(self, config: TrainingDashboardConfig):
        self.config = config
        self.log_offset_file = Path(self.config.corpus.log_corpus_dir) / ".ingest.offset"
        self.log_schedule_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Static corpus helpers
    # ------------------------------------------------------------------
    def list_static_corpus(self) -> List[CorpusSummary]:
        summaries: List[CorpusSummary] = []
        for path_str in self.config.corpus.static_paths:
            path = Path(path_str)
            if not path.exists():
                LOGGER.warning("静态语料不存在: %s", path)
                continue
            total = self._count_scenarios(path)
            summaries.append(CorpusSummary(path=path, total=total, description="静态配置"))
        return summaries

    def list_log_corpus(self) -> List[CorpusSummary]:
        log_dir = Path(self.config.corpus.log_corpus_dir)
        entries: List[CorpusSummary] = []
        for file in sorted(log_dir.glob("*.json")):
            total = self._count_scenarios(file)
            entries.append(CorpusSummary(path=file, total=total, description="日志提炼"))
        return entries

    # ------------------------------------------------------------------
    # Log ingestion
    # ------------------------------------------------------------------
    def start_scheduler(self) -> None:
        if self.log_schedule_thread and self.log_schedule_thread.is_alive():
            return
        self._stop_event.clear()
        self.log_schedule_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.log_schedule_thread.start()
        LOGGER.info("日志语料调度器已启动，周期 %s 分钟", self.config.corpus.ingest_interval_minutes)

    def stop_scheduler(self) -> None:
        self._stop_event.set()
        if self.log_schedule_thread and self.log_schedule_thread.is_alive():
            self.log_schedule_thread.join(timeout=2)
            LOGGER.info("日志语料调度器已停止")

    def ingest_logs_once(self) -> Optional[Path]:
        log_path = Path(self.config.corpus.log_source_path)
        if not log_path.exists():
            LOGGER.warning("日志文件不存在: %s", log_path)
            return None
        new_lines = self._read_new_log_lines(log_path)
        if not new_lines:
            LOGGER.info("无新增日志可提炼")
            return None
        scenarios = self._lines_to_scenarios(new_lines)
        if len(scenarios) < self.config.corpus.min_dialogues_per_batch:
            LOGGER.info(
                "本次提炼场景不足最小阈值(%s/%s)",
                len(scenarios),
                self.config.corpus.min_dialogues_per_batch,
            )
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.config.corpus.log_corpus_dir) / f"log_batch_{timestamp}.json"
        payload = {
            "version": timestamp,
            "source": "server_log",
            "scenarios": scenarios,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        LOGGER.info("已生成日志语料: %s (%s 场景)", output_path, len(scenarios))
        return output_path

    # ------------------------------------------------------------------
    # Corpus merge
    # ------------------------------------------------------------------
    def build_training_corpus(self, use_static: bool, use_logs: bool) -> Tuple[Optional[Path], int]:
        scenarios: List[Dict] = []
        if use_static:
            for summary in self.list_static_corpus():
                scenarios.extend(self._load_scenarios(summary.path))
        if use_logs:
            for summary in self.list_log_corpus():
                scenarios.extend(self._load_scenarios(summary.path))
        if not scenarios:
            return None, 0
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        combined_path = Path(self.config.data_root) / f"combined_{timestamp}.json"
        payload = {
            "version": timestamp,
            "source": "dashboard",
            "scenarios": scenarios,
        }
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with combined_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        LOGGER.info("已生成训练语料: %s (共 %s 场景)", combined_path, len(scenarios))
        return combined_path, len(scenarios)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run_scheduler(self) -> None:
        interval = max(1, int(self.config.corpus.ingest_interval_minutes)) * 60
        while not self._stop_event.is_set():
            try:
                self.ingest_logs_once()
            except Exception as exc:  # pragma: no cover - best effort logging
                LOGGER.exception("日志提炼失败: %s", exc)
            self._stop_event.wait(interval)

    def _count_scenarios(self, path: Path) -> int:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return len(data.get("scenarios", []))
        except Exception:
            return 0

    def _load_scenarios(self, path: Path) -> List[Dict]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return list(data.get("scenarios", []))
        except Exception as exc:
            LOGGER.error("读取语料失败 %s: %s", path, exc)
            return []

    def _read_new_log_lines(self, log_path: Path) -> List[str]:
        offset = 0
        if self.log_offset_file.exists():
            try:
                offset = int(self.log_offset_file.read_text().strip() or 0)
            except ValueError:
                offset = 0
        lines: List[str] = []
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(offset)
            for line in handle:
                lines.append(line.rstrip())
            new_offset = handle.tell()
        self.log_offset_file.write_text(str(new_offset))
        return lines

    def _lines_to_scenarios(self, lines: List[str]) -> List[Dict]:
        scenarios: List[Dict] = []
        current: Optional[Dict] = None
        for line in lines:
            if "/invoke 请求: tool=" in line:
                tool = self._extract_tool_name(line)
                current = self._new_scenario(tool, len(scenarios))
            elif "工具执行" in line and current:
                current["steps"].append({
                    "role": "agent",
                    "content": line.strip(),
                })
                scenarios.append(current)
                current = None
        return scenarios

    def _extract_tool_name(self, line: str) -> str:
        token = line.split("tool=")[-1]
        return token.split()[0]

    def _new_scenario(self, tool: str, index: int) -> Dict:
        timestamp = time.strftime("%Y%m%d")
        return {
            "name": f"log_{timestamp}_{index:04d}",
            "category": "log_replay",
            "persona": "自动提炼日志", 
            "is_real_data": False,
            "metadata": {
                "source_log": self.config.corpus.log_source_path,
                "tool": tool,
            },
            "steps": [
                {
                    "role": "user",
                    "content": f"请执行工具 {tool}，我需要相关帮助。",
                }
            ],
        }
