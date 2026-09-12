"""Earlier name; implementation moved to
``gnn/utils/errors/error_handling.py`` (S2-33 Step 7, family 3/3)."""

import warnings

warnings.warn(
    "gnn.utils.error_handling is the earlier name; import gnn.utils.errors.error_handling instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.errors.error_handling import (  # noqa: E402,F401
    CRITICAL_STEP_NUMBERS,
    ErrorCategory,
    ExitCode,
    PipelineError,
    PipelineErrorHandler,
    PipelineErrorSeverity,
    RecoveryStrategy,
    RetryConfig,
    coerce_step_exit_code,
    generate_correlation_id,
    handle_file_system_error,
    handle_network_error,
    handle_timeout_error,
    is_critical_pipeline_step,
    pipeline_exit_code,
    status_from_exit_code,
)

__all__ = [
    "CRITICAL_STEP_NUMBERS",
    "ErrorCategory",
    "ExitCode",
    "PipelineError",
    "PipelineErrorHandler",
    "PipelineErrorSeverity",
    "RecoveryStrategy",
    "RetryConfig",
    "coerce_step_exit_code",
    "generate_correlation_id",
    "handle_file_system_error",
    "handle_network_error",
    "handle_timeout_error",
    "is_critical_pipeline_step",
    "pipeline_exit_code",
    "status_from_exit_code",
]
