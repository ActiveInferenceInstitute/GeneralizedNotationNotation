"""Error-handling concern package (S2-33 Step 7, family 3/3): canonical home
of the structured error-handling/recovery/reporting framework and the error
message + recovery-strategy layer — code that used to live in the
``gnn/utils`` top-level grab-bag (design §3.8: the two modules are tightly
coupled — correlation IDs, recovery strategies — and are re-exported as one
family).

Eager re-export note (design §4.3.1): this family's ``__init__`` re-exports
the public names as real objects, not lazily. Both leaves are stdlib-only
(no module-scope psutil/matplotlib/logging-config side effects), so the
eager variant carries none of the import-weight risk that forced
``runtime_safety``/``observability`` onto PEP 562 resolution. Importing this
package eagerly imports both leaves — the same cost the old
``import gnn.utils.error_handling`` paid. Importing ``gnn.utils`` never
touches this package (the top-level facade stays lazy through its PEP 562
map, guarded by tests/tests/test_light_import.py).

Leaf inventory:
- error_handling: ``PipelineErrorHandler``/``PipelineError`` + severity,
  category, and recovery-strategy enums, the ``ExitCode`` contract, and the
  step exit-code helpers (``coerce_step_exit_code``, ``status_from_exit_code``,
  ``pipeline_exit_code``, ...)
- error_recovery: ``ErrorRecoveryManager``/``ErrorReporter``, the
  ``ErrorContext``/``ErrorRecord`` structures, ``ErrorCodeRegistry``, and the
  formatting helpers (``format_error_message``, ``format_and_log_error``,
  ``get_recovery_manager``)

Cross-family imports go through leaf modules, never through any facade (I5).
The old top-level paths (``gnn/utils/error_handling.py`` and
``gnn/utils/error_recovery.py``) are deprecation facades over this package.
"""

from gnn.utils.errors.error_handling import (
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
from gnn.utils.errors.error_recovery import (
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

__all__: list[str] = [
    "CRITICAL_STEP_NUMBERS",
    "ErrorCategory",
    "ErrorCodeRegistry",
    "ErrorContext",
    "ErrorRecord",
    "ErrorRecoveryManager",
    "ErrorReporter",
    "ErrorSeverity",
    "ExitCode",
    "PipelineError",
    "PipelineErrorHandler",
    "PipelineErrorSeverity",
    "RecoveryStrategy",
    "RetryConfig",
    "coerce_step_exit_code",
    "format_and_log_error",
    "format_error_message",
    "generate_correlation_id",
    "get_recovery_manager",
    "handle_file_system_error",
    "handle_network_error",
    "handle_timeout_error",
    "is_critical_pipeline_step",
    "pipeline_exit_code",
    "status_from_exit_code",
]
