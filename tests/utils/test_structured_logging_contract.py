"""Pins for ``utils/structured_logging`` (previously 41% coverage)."""

from __future__ import annotations

import logging

from gnn.utils.structured_logging import (
    LogContext,
    LogLevel,
    PerformanceMetrics,
    StructuredLogger,
    get_pipeline_logger,
)


def test_structured_logger_generates_correlation_id() -> None:
    logger = StructuredLogger("w2probe")

    assert logger.context.correlation_id
    assert len(logger.context.correlation_id) >= 8


def test_structured_logger_accepts_explicit_correlation_id() -> None:
    logger = StructuredLogger("w2probe", correlation_id="fixed-id-42")

    assert logger.context.correlation_id == "fixed-id-42"


def test_log_emits_message_at_requested_level(caplog: object) -> None:
    logger = StructuredLogger("w2probe-log")
    logger.set_context(step_name="3_gnn")

    with caplog.at_level(logging.DEBUG, logger="w2probe-log"):  # type: ignore[attr-defined]
        logger.info("hello structured", extra_key="extra-value")
        logger.error("bad thing", code=7)

    messages = [r.getMessage() for r in caplog.records]  # type: ignore[attr-defined]
    assert any("hello structured" in m for m in messages)
    assert any("bad thing" in m for m in messages)


def test_operation_context_restores_operation_and_tracks_duration(caplog: object) -> None:
    logger = StructuredLogger("w2probe-op")

    with caplog.at_level(logging.DEBUG, logger="w2probe-op"):  # type: ignore[attr-defined]
        with logger.operation_context("render_step"):
            assert logger.context.operation == "render_step"

    assert logger.context.operation is None
    # Performance tracking start/end pairs are logged.
    assert any("render_step" in r.getMessage() for r in caplog.records)  # type: ignore[attr-defined]


def test_performance_metrics_duration() -> None:
    metrics = PerformanceMetrics(start_time=1.0, end_time=3.5)

    assert metrics.duration_seconds == 2.5


def test_log_context_defaults_come_from_environment(monkeypatch: object) -> None:
    monkeypatch.setenv("ENVIRONMENT", "testing-env")  # type: ignore[attr-defined]

    context = LogContext(correlation_id="abc")

    assert context.environment == "testing-env"
    assert context.version == "1.1.3"


def test_get_pipeline_logger_returns_configured_logger() -> None:
    pipeline_logger = get_pipeline_logger("w2probe-pipeline")

    assert isinstance(pipeline_logger, StructuredLogger)



def test_all_log_levels_available() -> None:
    logger = StructuredLogger("w2probe-levels")

    for level in LogLevel:
        assert callable(getattr(logger, level.name.lower()))
