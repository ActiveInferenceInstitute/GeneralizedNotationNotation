"""Real behavioral tests for pipeline log/format configuration contracts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline import logging_config


def test_json_formatter_emits_parseable_entry() -> None:
    record = logging.LogRecord(
        name="pipeline.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    line = logging_config.JSONFormatter().format(record)
    entry = json.loads(line)
    assert entry["level"] == "INFO"
    assert entry["logger"] == "pipeline.test"
    assert entry["message"] == "hello world"
    assert "timestamp" in entry


def test_json_formatter_includes_structured_extra_and_exc() -> None:
    record = logging.LogRecord(
        name="pipeline.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="boom",
        args=(),
        exc_info=(ValueError, ValueError("bad"), None),
    )
    record.step = "3_gnn"
    record.duration = 1.25
    record.step_num = 3
    entry = json.loads(logging_config.JSONFormatter().format(record))
    assert entry["step"] == "3_gnn"
    assert entry["duration"] == 1.25
    assert entry["step_num"] == 3
    assert entry["exception"] == "bad"


def test_human_formatter_colors_level_and_messages() -> None:
    record = logging.LogRecord(
        name="pipeline.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="careful",
        args=(),
        exc_info=None,
    )
    out = logging_config.HumanFormatter().format(record)
    assert "WARNING" in out
    assert "careful" in out
    # ANSI color codes are present for the warning level.
    assert "\033[33m" in out


def test_human_formatter_appends_step_tag() -> None:
    record = logging.LogRecord(
        name="pipeline.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="step done",
        args=(),
        exc_info=None,
    )
    record.step = "11_render"
    out = logging_config.HumanFormatter().format(record)
    assert "[11_render]" in out


def test_configure_logging_sets_root_level_and_console_handler(
    monkeypatch: object,
) -> None:
    # Force a clean root logger state after the call.
    root = logging.getLogger()
    prior_handlers = list(root.handlers)
    prior_level = root.level
    try:
        logging_config.configure_logging(level=logging.WARNING, log_format="json")
        assert root.level == logging.WARNING
        assert len(root.handlers) >= 1
        # A JSON formatter is attached to the console handler.
        json_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert any(
            isinstance(h.formatter, logging_config.JSONFormatter)
            for h in json_handlers
        )
    finally:
        root.handlers = prior_handlers
        root.setLevel(prior_level)


def test_configure_logging_with_rotation_file(tmp_path: Path) -> None:
    root = logging.getLogger()
    prior = list(root.handlers)
    prior_level = root.level
    log_file = tmp_path / "logs" / "pipeline.log"
    try:
        logging_config.configure_logging(log_format="json", log_file=log_file)
        assert log_file.exists()
        # RotatingFileHandler is configured with rotation semantics.
        rot = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(rot) == 1
        assert rot[0].maxBytes == 10 * 1024 * 1024
        assert rot[0].backupCount == 5
        assert isinstance(rot[0].formatter, logging_config.JSONFormatter)
    finally:
        root.handlers = prior
        root.setLevel(prior_level)


def test_step_logger_propagates_step_context() -> None:
    adapter = logging_config.step_logger("3_gnn", step_num=3)
    assert adapter.extra["step"] == "3_gnn"
    assert adapter.extra["step_num"] == 3
    assert adapter.logger.name == "gnn.step.3_gnn"