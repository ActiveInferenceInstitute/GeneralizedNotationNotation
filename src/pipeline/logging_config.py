#!/usr/bin/env python3
"""
Structured Logging — JSON-formatted log output for pipeline observability.

Provides:
  - configure_logging(): sets up stdlib logging with optional JSON output
  - JSONFormatter: structured log formatter (timestamp, level, step, message)
  - Log rotation: 10 MB per file, 5 backups
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for machine-readable output.

    Each log line is a JSON object with fields:
      timestamp, level, logger, message, step (optional), duration (optional)
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Optional structured fields
        if hasattr(record, "step"):
            entry["step"] = record.step
        if hasattr(record, "duration"):
            entry["duration"] = record.duration
        if hasattr(record, "step_num"):
            entry["step_num"] = record.step_num

        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])

        return json.dumps(entry)


class HumanFormatter(logging.Formatter):
    """
    Human-readable colored formatter for terminal output.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        prefix = f"{color}{record.levelname:7s}{self.RESET}"

        step_tag = ""
        if hasattr(record, "step"):
            step_tag = f" [{record.step}]"

        return f"{prefix}{step_tag} {record.getMessage()}"


def configure_logging(
    level: int = logging.INFO,
    log_format: str = "human",
    log_file: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Configure pipeline logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_format: "human" for colored terminal, "json" for structured JSON.
        log_file: Optional file path for log rotation.
        max_bytes: Maximum size per log file (default 10 MB).
        backup_count: Number of rotated backups to keep.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    root.handlers.clear()

    # Choose formatter
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = HumanFormatter()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler with rotation
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        # Always use JSON for file logs
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)

    logging.getLogger(__name__).debug(
        f"Logging configured: format={log_format}, level={logging.getLevelName(level)}"
    )


def step_logger(step_name: str, step_num: int = -1) -> logging.LoggerAdapter:
    """
    Create a logger adapter with step context fields.

    Usage:
        log = step_logger("gnn_parse", step_num=3)
        log.info("Parsing complete", extra={"duration": 1.5})
    """
    logger = logging.getLogger(f"gnn.step.{step_name}")
    return logging.LoggerAdapter(logger, {"step": step_name, "step_num": step_num})
