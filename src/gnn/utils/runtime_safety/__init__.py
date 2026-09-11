"""Runtime-safety concern package (S2-33 Step 4): canonical home of the
"is it safe to run / did it finish / how much did it cost" concerns —
dependency availability + installation, operation timeouts, memory/resource
probes, bounded ``ast.literal_eval`` for untrusted parameter strings, schema
validation, and framework-availability probing (single source of truth used
by Steps 11/12) — code that used to live in the ``gnn/utils`` top-level
grab-bag.

Lazy re-export note (PEP 562): this family's ``__init__`` resolves names
lazily through ``__getattr__``. The original eager variant pulled
``resource_manager``'s module-scope ``psutil`` into EVERY leaf import of
this package (e.g. ``from gnn.utils.runtime_safety.safe_eval import ...``),
which broke psutil-blocked isolation tests and raised the package's import
weight; lazy resolution keeps every public name importable while leaves load
on first attribute access (same contract as the top-level ``gnn.utils``
facade; fixed 2026-09-11).

Cross-family imports go through leaf modules, never through any facade (I5).
The old top-level paths (``gnn/utils/dependency_validator.py`` etc.) are
deprecation facades over this package.
"""

from importlib import import_module
from typing import Any

_LEAF_BY_NAME: dict[str, str] = {
    "DEFAULT_MAX_DEPTH": "safe_eval",
    "DEFAULT_MAX_LEN": "safe_eval",
    "DependencySpec": "dependency_validator",
    "DependencyValidator": "dependency_validator",
    "FRAMEWORK_IMPORT_CHECK": "framework_availability",
    "FRAMEWORK_PRESETS": "validation_schemas",
    "FRAMEWORK_PROBE_STATEMENT": "framework_availability",
    "FrameworkStatus": "framework_availability",
    "KNOWN_FRAMEWORKS": "validation_schemas",
    "LLMTimeoutManager": "timeout_manager",
    "MATRIX_MAX_LEN": "safe_eval",
    "ProcessTimeoutManager": "timeout_manager",
    "ResourceTracker": "resource_manager",
    "TimeoutConfig": "timeout_manager",
    "TimeoutManager": "timeout_manager",
    "TimeoutResult": "timeout_manager",
    "TimeoutStrategy": "timeout_manager",
    "check_disk_space": "resource_manager",
    "check_framework": "framework_availability",
    "check_optional_dependencies": "dependency_validator",
    "estimate_resources": "resource_manager",
    "get_current_memory_usage": "resource_manager",
    "get_dependency_status": "dependency_validator",
    "get_llm_timeout_manager": "timeout_manager",
    "get_memory_usage": "resource_manager",
    "get_process_timeout_manager": "timeout_manager",
    "get_system_info": "resource_manager",
    "get_timeout_manager": "timeout_manager",
    "install_missing_dependencies": "dependency_validator",
    "is_framework_available": "framework_availability",
    "jax_pymdp_stack_ok": "jax_stack_validation",
    "log_resource_usage": "resource_manager",
    "normalize_pomdp_columns": "validation_schemas",
    "performance_tracker": "resource_manager",
    "run_jax_stack_probe_subprocess": "jax_stack_validation",
    "safe_literal_eval": "safe_eval",
    "track_peak_memory": "resource_manager",
    "validate_frameworks_arg": "validation_schemas",
    "validate_model_data": "validation_schemas",
    "validate_pipeline_dependencies": "dependency_validator",
    "validate_pipeline_dependencies_if_available": "dependency_validator",
    "validate_target_dir": "validation_schemas",
    "verify_jax_pymdp_stack": "jax_stack_validation",
    "with_async_timeout": "timeout_manager",
    "with_resource_limits": "resource_manager",
    "with_timeout": "timeout_manager",
}


def __getattr__(name: str) -> Any:
    """Resolve one public name from its leaf module (lazy; PEP 562)."""
    module_name = _LEAF_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"gnn.utils.runtime_safety.{module_name}"), name)


def __dir__() -> list[str]:
    return sorted(_LEAF_BY_NAME)


__all__: list[str] = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_LEN",
    "DependencySpec",
    "DependencyValidator",
    "FRAMEWORK_IMPORT_CHECK",
    "FRAMEWORK_PRESETS",
    "FRAMEWORK_PROBE_STATEMENT",
    "FrameworkStatus",
    "KNOWN_FRAMEWORKS",
    "LLMTimeoutManager",
    "MATRIX_MAX_LEN",
    "ProcessTimeoutManager",
    "ResourceTracker",
    "TimeoutConfig",
    "TimeoutManager",
    "TimeoutResult",
    "TimeoutStrategy",
    "check_disk_space",
    "check_framework",
    "check_optional_dependencies",
    "estimate_resources",
    "get_current_memory_usage",
    "get_dependency_status",
    "get_llm_timeout_manager",
    "get_memory_usage",
    "get_process_timeout_manager",
    "get_system_info",
    "get_timeout_manager",
    "install_missing_dependencies",
    "is_framework_available",
    "jax_pymdp_stack_ok",
    "log_resource_usage",
    "normalize_pomdp_columns",
    "performance_tracker",
    "run_jax_stack_probe_subprocess",
    "safe_literal_eval",
    "track_peak_memory",
    "validate_frameworks_arg",
    "validate_model_data",
    "validate_pipeline_dependencies",
    "validate_pipeline_dependencies_if_available",
    "validate_target_dir",
    "verify_jax_pymdp_stack",
    "with_async_timeout",
    "with_resource_limits",
    "with_timeout",
]
