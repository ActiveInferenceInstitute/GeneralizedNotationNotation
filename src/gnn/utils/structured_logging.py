"""Earlier name; implementation moved to
``gnn/utils/observability/structured_logging.py`` (S2-33 Step 5).

The module-level ``LogAggregator`` singleton and correlation context
threadlocal moved with the file, so identity and state are preserved (R4)."""

import warnings

warnings.warn(
    "gnn.utils.structured_logging is the earlier name; import gnn.utils.observability.structured_logging instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.observability.structured_logging import (  # noqa: E402,F401
    LogAggregator,
    LogContext,
    LogFormat,
    LogLevel,
    PerformanceMetrics,
    StructuredFormatter,
    StructuredLogger,
    get_pipeline_logger,
    get_system_info,
    log_pipeline_complete,
    log_pipeline_start,
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
    set_correlation_context,
)

__all__ = [
    "LogAggregator",
    "LogContext",
    "LogFormat",
    "LogLevel",
    "PerformanceMetrics",
    "StructuredFormatter",
    "StructuredLogger",
    "get_pipeline_logger",
    "get_system_info",
    "log_pipeline_complete",
    "log_pipeline_start",
    "log_step_error",
    "log_step_start",
    "log_step_success",
    "log_step_warning",
    "set_correlation_context",
]
