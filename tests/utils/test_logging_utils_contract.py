"""Pins for ``utils/logging/logging_utils`` (previously 34% coverage)."""

from __future__ import annotations

import logging

from gnn.utils.logging.logging_utils import (
    BasicPipelineLogger,
    CorrelationFormatter,
    PipelineLogger,
    setup_step_logging,
)


def _record() -> logging.LogRecord:
    return logging.LogRecord(
        name="w2probe",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="with correlation",
        args=(),
        exc_info=None,
    )


def test_setup_step_logging_returns_named_logger() -> None:
    logger = setup_step_logging("3_gnn", verbose=False)

    assert isinstance(logger, logging.Logger)
    assert logger.name == "3_gnn"


def test_correlation_formatter_stamps_record_with_context_defaults() -> None:
    BasicPipelineLogger.clear_correlation_context()
    record = _record()

    CorrelationFormatter("%(message)s").format(record)

    # The formatter stamps the record so format strings can reference the
    # correlation context ("MAIN"/"pipeline" defaults).
    assert record.correlation_id == "MAIN"  # type: ignore[attr-defined]
    assert record.step_name == "pipeline"  # type: ignore[attr-defined]


def test_correlation_formatter_renders_context_when_referenced() -> None:
    BasicPipelineLogger.set_correlation_context("pipeline", correlation_id="corr-w2")
    try:
        formatted = CorrelationFormatter(
            "[%(correlation_id)s][%(step_name)s] %(message)s"
        ).format(_record())
        assert formatted == "[corr-w2][pipeline] with correlation"
    finally:
        BasicPipelineLogger.clear_correlation_context()


def test_pipeline_logger_get_logger_is_idempotent() -> None:
    first = PipelineLogger.get_logger("w2probe-pipeline")
    second = PipelineLogger.get_logger("w2probe-pipeline")

    assert first is second
    assert isinstance(first, logging.Logger)


def test_basic_pipeline_logger_initializes_once() -> None:
    BasicPipelineLogger.initialize()

    logger = BasicPipelineLogger.get_logger("w2probe-basic")
    assert isinstance(logger, logging.Logger)


def test_correlation_context_clear_restores_default() -> None:
    BasicPipelineLogger.set_correlation_context("corr-temp")
    BasicPipelineLogger.clear_correlation_context()
    record = _record()

    CorrelationFormatter("%(message)s").format(record)

    assert record.correlation_id == "MAIN"  # type: ignore[attr-defined]
