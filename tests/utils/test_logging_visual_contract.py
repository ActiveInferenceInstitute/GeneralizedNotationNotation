"""Pins for the logging subpackage's visual surface (batch 10).

``VisualLoggingEnhancer`` formatting tiers, the in-memory
``PipelineProgressTracker`` lifecycle, standalone logging setup, the custom
TRACE/STEP levels, and the module-level correlation-context helpers.
"""

from __future__ import annotations

import logging

from gnn.utils.logging import logging_utils
from gnn.utils.logging.logging_utils import (
    PipelineLogger,
    PipelineProgressTracker,
    VisualLoggingEnhancer,
    log_section_header,
    set_correlation_context,
    setup_correlation_context,
    setup_standalone_logging,
    silence_noisy_modules_in_console,
)

# --- VisualLoggingEnhancer ----------------------------------------------------


def test_format_duration_tiers() -> None:
    assert "ms" in VisualLoggingEnhancer.format_duration(0.5)
    assert "s" in VisualLoggingEnhancer.format_duration(5.0)
    minutes = VisualLoggingEnhancer.format_duration(90.0)
    assert "1m" in minutes and "30s" in minutes
    hours = VisualLoggingEnhancer.format_duration(5400.0)
    assert "1h" in hours


def test_format_memory_usage_tiers() -> None:
    assert "MB" in VisualLoggingEnhancer.format_memory_usage(50.0)
    assert "GB" in VisualLoggingEnhancer.format_memory_usage(2048.0)


def test_format_step_header_structure() -> None:
    header = VisualLoggingEnhancer.format_step_header(2, 5, "3_gnn.py")

    assert "Step 2/5" in header
    assert "3_gnn.py" in header
    assert "40%" in header


def test_colorize_passthrough_without_tty() -> None:
    # pytest captures stdout: not a tty, so colorize must return the text.
    assert VisualLoggingEnhancer.colorize("plain", "GREEN") == "plain"


# --- PipelineProgressTracker ---------------------------------------------------


def test_progress_tracker_lifecycle() -> None:
    tracker = PipelineProgressTracker(total_steps=3)

    header = tracker.start_step(1, "3_gnn.py")
    assert "Step 1/3" in header
    assert tracker.step_status[1] == "RUNNING"

    summary = tracker.complete_step(1, "SUCCESS", duration=2.5)
    assert "SUCCESS" in summary
    assert "1/3" in summary
    assert tracker.step_durations[1] == 2.5


def test_progress_tracker_counts_statuses() -> None:
    tracker = PipelineProgressTracker(total_steps=4)
    tracker.start_step(1, "a")
    tracker.complete_step(1, "SUCCESS")
    tracker.start_step(2, "b")
    tracker.complete_step(2, "SUCCESS_WITH_WARNINGS")
    tracker.start_step(3, "c")
    tracker.complete_step(3, "FAILED")

    overall = tracker.get_overall_progress()

    assert "Success: 2" in overall
    assert "Warnings: 1" in overall
    assert "Failed: 1" in overall


# --- standalone setup + custom levels + context helpers -------------------------


def test_setup_standalone_logging_returns_named_logger(tmp_path: object) -> None:
    logger = setup_standalone_logging(
        output_dir=tmp_path,
        logger_name="W2_Standalone",  # type: ignore[attr-defined]
    )

    assert isinstance(logger, logging.Logger)
    assert logger.name == "W2_Standalone"


def test_silence_noisy_modules_caps_levels() -> None:
    silence_noisy_modules_in_console()

    for module in ("PIL", "matplotlib", "urllib3", "requests"):
        assert logging.getLogger(module).level == logging.WARNING


def test_custom_trace_and_step_levels_registered() -> None:
    assert logging.getLevelName("TRACE") == 5
    assert logging.getLevelName("STEP") == 25


def test_module_correlation_context_helpers() -> None:
    # setup_correlation_context(step_name, correlation_id=None) — the step
    correlation_id = setup_correlation_context("W2_Probe_Step", "corr-w2b")
    assert correlation_id == "corr-w2b"

    set_correlation_context("W2_Probe_Step", "corr-w2c")

    # The module helpers delegate to the thread-local context the formatter
    # stamps from.
    assert logging_utils._correlation_context.correlation_id == "corr-w2c"  # type: ignore[attr-defined]
    assert logging_utils._correlation_context.step_name == "W2_Probe_Step"  # type: ignore[attr-defined]

    PipelineLogger.clear_correlation_context()


def test_log_section_header_emits_border(caplog: object) -> None:
    probe = logging.getLogger("W2_SectionProbe")
    probe.setLevel(logging.INFO)

    with caplog.at_level(logging.INFO, logger="W2_SectionProbe"):  # type: ignore[attr-defined]
        log_section_header(probe, "Section Title", char="-", length=40)

    rendered = " ".join(r.getMessage() for r in caplog.records)  # type: ignore[attr-defined]
    assert "Section Title" in rendered
    assert "-" * 40 in rendered
