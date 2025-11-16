"""Orchestrates training sessions by wrapping train_rl_agent.py."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional

from agent.logger import get_logger

from .config import TrainingDashboardConfig

LOGGER = get_logger(__name__)


@dataclass
class TrainingRequest:
    timesteps: int
    eval_freq: int
    checkpoint_freq: int
    max_steps_per_episode: int
    use_text_embedding: bool
    output_dir: str
    scenario_file: Optional[str] = None


@dataclass
class TrainingStatus:
    state: str = "idle"
    message: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    params: Optional[TrainingRequest] = None


class TrainingRunner:
    """Launches training subprocesses and streams logs/metrics."""

    def __init__(self, config: TrainingDashboardConfig):
        self.config = config
        self._process: Optional[subprocess.Popen[str]] = None
        self._log_queue: "queue.Queue[str]" = queue.Queue(maxsize=1000)
        self._log_history: Deque[str] = deque(maxlen=2000)
        self._status = TrainingStatus()
        self._lock = threading.Lock()
        self._log_thread: Optional[threading.Thread] = None
        self._metrics_cache: Dict[str, float] = {}
        self._current_output_dir: Optional[Path] = None
        self._metrics_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    def start(self, request: TrainingRequest) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                raise RuntimeError("已有训练任务正在运行")
            cmd = self._build_command(request)
            LOGGER.info("启动训练命令: %s", " ".join(cmd))
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._status = TrainingStatus(
                state="running",
                message="训练进行中",
                started_at=time.time(),
                params=request,
            )
            self._current_output_dir = Path(request.output_dir)
            self._log_thread = threading.Thread(target=self._consume_logs, daemon=True)
            self._log_thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._process:
                return
            if self._process.poll() is None:
                self._process.terminate()
                LOGGER.info("已发送训练终止信号")
            self._process = None
            self._status.state = "stopped"
            self._status.finished_at = time.time()

    def get_status(self) -> TrainingStatus:
        self._refresh_state()
        return self._status

    def read_logs(self, max_lines: int = 200) -> str:
        lines = []
        while len(lines) < max_lines:
            try:
                lines.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        return "\n".join(lines)

    def read_metrics(self) -> Dict[str, float]:
        self._refresh_metrics()
        return dict(self._metrics_cache)

    def read_log_history(self, max_lines: int = 200) -> str:
        """Return recent log lines without draining the live queue."""

        if not self._log_history:
            return ""
        tail: List[str] = list(self._log_history)[-max_lines:]
        return "\n".join(tail)

    def read_metric_series(self, limit: int = 200) -> List[Dict[str, float]]:
        """Return the cached metric history for richer visualizations."""

        self._refresh_metrics()
        if limit and limit > 0:
            return self._metrics_history[-limit:]
        return list(self._metrics_history)

    # ------------------------------------------------------------------
    def _build_command(self, request: TrainingRequest) -> list[str]:
        cmd = [sys.executable, "train_rl_agent.py"]
        cmd += ["--timesteps", str(request.timesteps)]
        cmd += ["--eval-freq", str(request.eval_freq)]
        cmd += ["--checkpoint-freq", str(request.checkpoint_freq)]
        cmd += ["--output-dir", request.output_dir]
        cmd += ["--max-steps-per-episode", str(request.max_steps_per_episode)]
        if request.use_text_embedding:
            cmd.append("--use-text-embedding")
        if request.scenario_file:
            cmd += ["--scenario-file", request.scenario_file]
        return cmd

    def _consume_logs(self) -> None:  # pragma: no cover - runtime side effect
        assert self._process and self._process.stdout
        for line in self._process.stdout:
            cleaned = line.rstrip()
            try:
                self._log_queue.put_nowait(cleaned)
            except queue.Full:
                pass
            self._log_history.append(cleaned)
        self._process.wait()
        exit_code = self._process.returncode
        LOGGER.info("训练进程退出: %s", exit_code)
        self._status.finished_at = time.time()
        self._status.state = "succeeded" if exit_code == 0 else "failed"
        self._status.message = f"训练结束，退出码 {exit_code}" if exit_code else "训练完成"
        self._process = None

    def _refresh_state(self) -> None:
        with self._lock:
            if not self._process:
                return
            if self._process.poll() is not None:
                # 线程可能早于此处更新
                return

    def _refresh_metrics(self) -> None:
        if not self._current_output_dir:
            return
        log_file = self._current_output_dir / "logs" / "training_log.json"
        if not log_file.exists():
            return
        try:
            with log_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return
        stats = data.get("episode_stats") or []
        if not stats:
            return

        history: List[Dict[str, float]] = []
        for idx, item in enumerate(stats):
            reward = item.get("mean_reward") or item.get("reward") or 0
            length = item.get("mean_length") or item.get("length") or 0
            history.append({
                "step": idx,
                "mean_reward": float(reward),
                "mean_length": float(length),
            })

        latest = history[-1]
        self._metrics_history = history
        self._metrics_cache = {
            "timesteps": data.get("num_timesteps", 0),
            "mean_reward": latest["mean_reward"],
            "mean_length": latest["mean_length"],
        }

    def export_request(self) -> Dict:
        if not self._status.params:
            return {}
        return asdict(self._status.params)
