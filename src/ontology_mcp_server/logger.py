from __future__ import annotations
# Copyright (c) 2025 shark8848
# MIT License
#
# Ontology MCP Server - 电商 AI 助手系统
# 本体推理 + 电商业务逻辑 + 对话记忆 + 可视化 UI
#
# Author: shark8848
# Repository: https://github.com/shark8848/ontology-mcp-server
"""日志初始化与获取封装。"""

import logging
import os
import re
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

_initialized: bool = False
_LOGGER_NAME = "ontology_mcp_server"


class DailyTimestampedRotatingHandler(TimedRotatingFileHandler):
    """按天分割日志，历史文件自动追加时间戳。"""

    _TIMESTAMP_FORMAT = "%Y%m%d"

    def __init__(self, filename: str, *, backup_count: int = 14) -> None:
        super().__init__(
            filename,
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        self.suffix = "%Y%m%d"

    def rotate(self, source: str, dest: str) -> None:  # type: ignore[override]
        src_path = Path(source)
        # dest 形如 /path/server.log.20251130
        timestamp = Path(dest).name.split(".")[-1]
        if not timestamp:
            timestamp = datetime.now().strftime(self._TIMESTAMP_FORMAT)
        rotated_name = f"{src_path.stem}_{timestamp}{src_path.suffix}"
        rotated_path = src_path.with_name(rotated_name)
        if rotated_path.exists():
            rotated_path.unlink()
        super().rotate(source, str(rotated_path))

    def getFilesToDelete(self) -> list[str]:  # type: ignore[override]
        if self.backupCount <= 0:
            return []

        base_path = Path(self.baseFilename)
        pattern = f"{base_path.stem}_*{base_path.suffix}"
        candidates = sorted(
            (
                path
                for path in base_path.parent.glob(pattern)
                if self._is_timestamped_rotation(path)
            ),
            key=lambda p: p.stat().st_mtime,
        )

        if len(candidates) <= self.backupCount:
            return []
        return [str(p) for p in candidates[: len(candidates) - self.backupCount]]

    @staticmethod
    def _is_timestamped_rotation(path: Path) -> bool:
        stem_parts = path.stem.rsplit("_", 1)
        if len(stem_parts) != 2:
            return False
        timestamp = stem_parts[1]
        return bool(re.fullmatch(r"\d{8}", timestamp))


def _log_dir() -> Path:
    env_dir = os.getenv("ONTOLOGY_SERVER_LOG_DIR") or os.getenv("ONTOLOGY_MCP_LOG_DIR")

    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))

    repo_logs = Path(__file__).resolve().parents[2] / "logs"
    candidates.append(repo_logs)
    candidates.append(Path.cwd() / "logs")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            continue

    fallback = Path.cwd()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _base_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def init_logging(level_name: Optional[str] = None) -> None:
    """初始化 MCP Server 的专用 logger。

    输出到控制台以及 ontology_mcp_server/logs/server.log（或自定义目录），支持 ONTOLOGY_MCP_LOG_LEVEL。"""
    global _initialized
    if _initialized:
        return

    log_dir = _log_dir()

    level_str = (level_name or os.getenv("ONTOLOGY_MCP_LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    logger = _base_logger()
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    try:
        file_path = log_dir / "server.log"
        backup_count = int(os.getenv("ONTOLOGY_LOG_BACKUP_COUNT", "14"))
        file_handler = DailyTimestampedRotatingHandler(
            str(file_path), backup_count=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass

    _initialized = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """返回 MCP Server 作用域下的 logger。"""
    init_logging()
    base = _base_logger()
    if not name or name == _LOGGER_NAME:
        return base
    if name.startswith(f"{_LOGGER_NAME}."):
        suffix = name[len(_LOGGER_NAME) + 1 :]
        return base.getChild(suffix)
    return base.getChild(name)
