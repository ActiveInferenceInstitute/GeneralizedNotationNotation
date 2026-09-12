"""Earlier name; implementation moved to
``gnn/utils/errors/error_recovery.py`` (S2-33 Step 7, family 3/3)."""

import warnings

warnings.warn(
    "gnn.utils.error_recovery is the earlier name; import gnn.utils.errors.error_recovery instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.errors.error_recovery import (  # noqa: E402,F401
    ErrorCodeRegistry,
    ErrorContext,
    ErrorRecord,
    ErrorRecoveryManager,
    ErrorReporter,
    ErrorSeverity,
    format_and_log_error,
    format_error_message,
    get_recovery_manager,
)

__all__ = [
    "ErrorCodeRegistry",
    "ErrorContext",
    "ErrorRecord",
    "ErrorRecoveryManager",
    "ErrorReporter",
    "ErrorSeverity",
    "format_and_log_error",
    "format_error_message",
    "get_recovery_manager",
]
