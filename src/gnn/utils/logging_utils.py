#!/usr/bin/env python3
"""Logging utilities for the GNN processing pipeline.

This is the single public entry point for pipeline logging. The
implementation lives in ``gnn.utils.logging.logging_utils`` (internal);
import everything from this module.
"""

import logging
from typing import Any, Dict, Optional, cast

from gnn.utils.observability.performance_tracking import (
    PerformanceTracker,
    performance_tracker,
)

# Import the structured logging implementation.
from .logging.logging_utils import (
    PipelineLogger as PipelineLogger,
)
from .logging.logging_utils import (
    PipelineProgressTracker as PipelineProgressTracker,
)
from .logging.logging_utils import (
    log_pipeline_summary as log_pipeline_summary,
)
from .logging.logging_utils import (
    log_section_header as log_section_header,
)
from .logging.logging_utils import (
    log_step_error as new_log_step_error,
)
from .logging.logging_utils import (
    log_step_start as log_step_start,
)
from .logging.logging_utils import (
    log_step_success as log_step_success,
)
from .logging.logging_utils import (
    log_step_warning as log_step_warning,
)
from .logging.logging_utils import (
    reset_progress_tracker as reset_progress_tracker,
)
from .logging.logging_utils import (
    rotate_logs as rotate_logs,
)
from .logging.logging_utils import (
    set_correlation_context as set_correlation_context,
)
from .logging.logging_utils import (
    set_global_progress_tracker as set_global_progress_tracker,
)
from .logging.logging_utils import (
    setup_correlation_context as new_setup_correlation_context,
)
from .logging.logging_utils import (
    setup_main_logging as setup_main_logging,
)
from .logging.logging_utils import (
    setup_step_logging as setup_step_logging,
)


def log_step_error(
    logger: logging.Logger,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    **metadata: Any,
) -> None:
    """Log an error during step execution (merges ``context`` into metadata)."""
    details = dict(metadata)
    if context:
        details.update(context)
    new_log_step_error(logger, message, **details)


def get_performance_summary() -> Dict[str, Any]:
    """Get a summary of performance metrics."""
    return cast("dict[str, Any]", performance_tracker.get_summary())


def setup_correlation_context(
    correlation_id: Optional[str] = None, step_name: Optional[str] = None
) -> None:
    """Set up correlation context for logging."""
    new_setup_correlation_context(step_name or "unknown", correlation_id)


# Export all public symbols for compatibility
__all__: list[Any] = [
    "PipelineLogger",
    "PipelineProgressTracker",
    "setup_step_logging",
    "setup_main_logging",
    "log_step_start",
    "log_step_success",
    "log_step_warning",
    "log_step_error",
    "log_section_header",
    "log_pipeline_summary",
    "get_performance_summary",
    "PerformanceTracker",
    "performance_tracker",
    "reset_progress_tracker",
    "rotate_logs",
    "set_correlation_context",
    "set_global_progress_tracker",
    "setup_correlation_context",
]
