# This file makes utils a package

"""
GNN Pipeline Utilities Package

Lazy PEP 562 re-export surface: 113 exported names aggregated from 17 source
modules. All pipeline steps import from this package for consistency. The
surface area is intentionally wide; it is being split by concern into
sub-packages — see docs/development/utils_split_design.md (S2-33/SC-38) for
the package map and migration plan.

Importing ``utils`` is intentionally LIGHT: no submodule executes at import
time, so heavy module-scope dependencies (psutil via structured_logging /
resource_manager, matplotlib via simulation_utils) are only paid when an
exported name is actually resolved through ``__getattr__``.

Source modules:
- logging_utils: Re-export facade over utils/logging/logging_utils.py (centralized, correlation-aware logging system)
- argument_utils: Streamlined argument parsing and validation
- resource_manager: Memory and resource usage tracking
- error_recovery: Error context, severity, and recovery management (S2-33 Step 7
  concern package errors/; the top-level error_*.py paths remain as deprecation
  facades)
- pipeline_monitor: Pipeline health reporting
- pipeline_validator: Pre-execution prerequisite checker (step output validation)
- dependency_validator: Comprehensive dependency validation
- config_loader: YAML configuration loading and validation (active config system)
  (S2-33 Step 7 concern package config_io/, together with io_utils,
  code_metrics, and path_utils; the four top-level paths remain as deprecation
  facades)
- performance_tracking: Operation timing and performance metrics
- base_processor: Abstract base class for standardized step processors
- venv_utils: Virtual environment path helpers
- system_utils: System information gathering
- testing: Test runner, categories, stages, and coverage targets (S2-33 concern
  package; testing_utils.py remains as its deprecation facade)
- pipeline: Pipeline utility exports
- error_handling: Structured error handler, categories, and recovery strategies
  (S2-33 Step 7 concern package errors/; the top-level path remains as a
  deprecation facade)
- structured_logging: Structured log emission with correlation context
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Static re-export surface for type checkers: mirrors the pre-lazy eager
    # imports so mypy resolves names without executing submodules at runtime.
    from .arguments.arg_parsing import (
        ArgumentParser,
        build_step_command_args,
        get_pipeline_step_info,
        parse_arguments,
    )
    from .arguments.path_conversion import (
        validate_and_convert_paths,
        validate_pipeline_configuration,
    )
    from .arguments.pipeline_arguments import PipelineArguments
    from .arguments.step_config import StepConfiguration
    from .config_io.config_loader import (
        GNNPipelineConfig,
        LLMConfig,
        ModelConfig,
        OntologyConfig,
        PipelineConfig,
        SAPFConfig,
        SetupConfig,
        TypeCheckerConfig,
        WebsiteConfig,
        get_config_value,
        load_config,
        save_config,
        set_config_value,
        validate_config,
    )
    from .errors.error_handling import (
        ErrorCategory,
        ExitCode,
        PipelineErrorHandler,
        PipelineErrorSeverity,
        RecoveryStrategy,
        generate_correlation_id,
        handle_file_system_error,
        handle_network_error,
        handle_timeout_error,
    )
    from .errors.error_recovery import (
        ErrorCodeRegistry,
        ErrorContext,
        ErrorRecoveryManager,
        ErrorSeverity,
        format_and_log_error,
        format_error_message,
        get_recovery_manager,
    )
    from .logging_utils import (
        PipelineLogger,
        get_performance_summary,
        log_section_header,
        setup_correlation_context,
        setup_main_logging,
        setup_step_logging,
    )
    from .observability.performance_tracking import (
        PerformanceTracker,
        performance_tracker,
        track_operation_standalone,
    )
    from .observability.structured_logging import (
        StructuredLogger,
        get_pipeline_logger,
        log_pipeline_complete,
        log_pipeline_start,
        log_step_error,
        log_step_start,
        log_step_success,
        log_step_warning,
        set_correlation_context,
    )
    from .pipeline import (
        RecoveryArgumentParser,
        execute_pipeline_step_template,
        get_output_dir_for_script,
        get_pipeline_utilities,
        validate_output_directory,
    )
    from .pipeline_orchestration.base_processor import (
        BaseProcessor,
        ProcessingResult,
        create_processor,
    )
    from .pipeline_orchestration.pipeline_monitor import (
        generate_pipeline_health_report,
    )
    from .pipeline_orchestration.pipeline_template import (
        create_standardized_pipeline_script,
    )
    from .pipeline_orchestration.pipeline_validator import (
        validate_pipeline_step_sequence,
        validate_step_prerequisites,
    )
    from .runtime_safety.dependency_validator import (
        DependencySpec,
        DependencyValidator,
        check_optional_dependencies,
        get_dependency_status,
        install_missing_dependencies,
        validate_pipeline_dependencies,
        validate_pipeline_dependencies_if_available,
    )
    from .runtime_safety.resource_manager import get_current_memory_usage
    from .system_utils import get_system_info
    from .testing.constants import (
        COVERAGE_TARGETS,
        TEST_CATEGORIES,
        TEST_CONFIG,
        TEST_STAGES,
    )
    from .testing.environment import (
        cleanup_test_environment,
        get_test_configuration,
        get_test_coverage,
        get_test_dependencies,
        get_test_environment,
        install_test_dependencies,
        setup_test_environment,
        validate_coverage_targets,
        validate_test_configuration,
        validate_test_dependencies,
        validate_test_environment,
    )
    from .testing.reports import (
        generate_test_report,
        get_test_artifacts,
        get_test_duration,
        get_test_logs,
        get_test_metadata,
        get_test_performance,
        get_test_progress,
        get_test_results,
        get_test_statistics,
        get_test_status,
        get_test_summary,
        get_test_timestamps,
    )
    from .testing.runner import (
        CoverageTarget,
        TestCategory,
        TestResult,
        TestRunner,
        TestStage,
        run_test_category,
        run_test_stage,
        run_tests,
    )
    from .venv_utils import get_venv_python

from gnn import __version__

FEATURES: dict[str, Any] = {
    "argument_parsing": True,
    "logging": True,
    "error_handling": True,
    "pipeline_monitoring": True,
    "dependency_management": True,
    "path_utilities": True,
}

# Flag to indicate utils are available (used by pipeline modules)
UTILS_AVAILABLE = True

# Explicit name -> source submodule map for every re-export. Resolving a name
# imports only that one submodule (``from importlib import import_module``
# stays inside the function so importing ``utils`` itself stays light).
_EXPORT_MAP: dict[str, str] = {
    # arguments (S2-33 Step 2: moved from the top-level argument modules into
    # the arguments/ concern package; keys unchanged, values repointed)
    "ArgumentParser": "arguments.arg_parsing",
    "PipelineArguments": "arguments.pipeline_arguments",
    "StepConfiguration": "arguments.step_config",
    "build_step_command_args": "arguments.arg_parsing",
    "get_pipeline_step_info": "arguments.arg_parsing",
    "parse_arguments": "arguments.arg_parsing",
    "validate_and_convert_paths": "arguments.path_conversion",
    "validate_pipeline_configuration": "arguments.path_conversion",
    # pipeline_orchestration (S2-33 Step 3: moved from the top-level pipeline
    # modules; keys unchanged, values repointed)
    "BaseProcessor": "pipeline_orchestration.base_processor",
    "ProcessingResult": "pipeline_orchestration.base_processor",
    "create_processor": "pipeline_orchestration.base_processor",
    # config_io (S2-33 Step 7, family 2/3: moved from the top-level config/IO
    # modules; keys unchanged, values repointed)
    "GNNPipelineConfig": "config_io.config_loader",
    "LLMConfig": "config_io.config_loader",
    "ModelConfig": "config_io.config_loader",
    "OntologyConfig": "config_io.config_loader",
    "PipelineConfig": "config_io.config_loader",
    "SAPFConfig": "config_io.config_loader",
    "SetupConfig": "config_io.config_loader",
    "TypeCheckerConfig": "config_io.config_loader",
    "WebsiteConfig": "config_io.config_loader",
    "get_config_value": "config_io.config_loader",
    "load_config": "config_io.config_loader",
    "save_config": "config_io.config_loader",
    "set_config_value": "config_io.config_loader",
    "validate_config": "config_io.config_loader",
    # runtime_safety (S2-33 Step 4: moved from the top-level safety modules;
    # keys unchanged, values repointed)
    "DependencySpec": "runtime_safety.dependency_validator",
    "DependencyValidator": "runtime_safety.dependency_validator",
    "check_optional_dependencies": "runtime_safety.dependency_validator",
    "get_dependency_status": "runtime_safety.dependency_validator",
    "install_missing_dependencies": "runtime_safety.dependency_validator",
    "validate_pipeline_dependencies": "runtime_safety.dependency_validator",
    "validate_pipeline_dependencies_if_available": "runtime_safety.dependency_validator",
    # errors (S2-33 Step 7, family 3/3: moved from the top-level error modules
    # into the errors/ concern package; keys unchanged, values repointed)
    "ErrorCategory": "errors.error_handling",
    "ExitCode": "errors.error_handling",
    "PipelineErrorHandler": "errors.error_handling",
    "PipelineErrorSeverity": "errors.error_handling",
    "RecoveryStrategy": "errors.error_handling",
    "generate_correlation_id": "errors.error_handling",
    "handle_file_system_error": "errors.error_handling",
    "handle_network_error": "errors.error_handling",
    "handle_timeout_error": "errors.error_handling",
    "ErrorCodeRegistry": "errors.error_recovery",
    "ErrorContext": "errors.error_recovery",
    "ErrorRecoveryManager": "errors.error_recovery",
    "ErrorSeverity": "errors.error_recovery",
    "format_and_log_error": "errors.error_recovery",
    "format_error_message": "errors.error_recovery",
    "get_recovery_manager": "errors.error_recovery",
    # logging_utils
    "PipelineLogger": "logging.logging_utils",
    "get_performance_summary": "logging.logging_utils",
    "log_section_header": "logging.logging_utils",
    "setup_correlation_context": "logging.logging_utils",
    "setup_main_logging": "logging.logging_utils",
    "setup_step_logging": "logging.logging_utils",
    # performance_tracking (renamed from performance_tracker.py: the exported
    # object must not share its module's name, or any prior
    # 'import utils.performance_tracker' shadows the re-export with the module)
    "PerformanceTracker": "observability.performance_tracking",
    "performance_tracker": "observability.performance_tracking",
    "track_operation_standalone": "observability.performance_tracking",
    # pipeline
    "RecoveryArgumentParser": "pipeline",
    "execute_pipeline_step_template": "pipeline",
    "get_output_dir_for_script": "pipeline",
    "get_pipeline_utilities": "pipeline",
    "validate_output_directory": "pipeline",
    # pipeline_monitor
    "generate_pipeline_health_report": "pipeline_orchestration.pipeline_monitor",
    # pipeline_template
    "create_standardized_pipeline_script": "pipeline_orchestration.pipeline_template",
    # pipeline_validator
    "validate_pipeline_step_sequence": "pipeline_orchestration.pipeline_validator",
    "validate_step_prerequisites": "pipeline_orchestration.pipeline_validator",
    # resource_manager
    "get_current_memory_usage": "runtime_safety.resource_manager",
    # observability (S2-33 Step 5: moved from the top-level
    # structured_logging/performance_tracking modules; keys unchanged,
    # values repointed)
    "StructuredLogger": "observability.structured_logging",
    "get_pipeline_logger": "observability.structured_logging",
    "log_pipeline_complete": "observability.structured_logging",
    "log_pipeline_start": "observability.structured_logging",
    "log_step_error": "observability.structured_logging",
    "log_step_start": "observability.structured_logging",
    "log_step_success": "observability.structured_logging",
    "log_step_warning": "observability.structured_logging",
    "set_correlation_context": "observability.structured_logging",
    # system_utils
    "get_system_info": "system_utils",
    # testing (S2-33 Step 1: moved from testing_utils.py into the testing/
    # concern package; keys unchanged, values repointed to the new leaves)
    "COVERAGE_TARGETS": "testing.constants",
    "TEST_CATEGORIES": "testing.constants",
    "TEST_CONFIG": "testing.constants",
    "TEST_STAGES": "testing.constants",
    "CoverageTarget": "testing.runner",
    "TestCategory": "testing.runner",
    "TestResult": "testing.runner",
    "TestRunner": "testing.runner",
    "TestStage": "testing.runner",
    "cleanup_test_environment": "testing.environment",
    "generate_test_report": "testing.reports",
    "get_test_artifacts": "testing.reports",
    "get_test_configuration": "testing.environment",
    "get_test_coverage": "testing.environment",
    "get_test_dependencies": "testing.environment",
    "get_test_duration": "testing.reports",
    "get_test_environment": "testing.environment",
    "get_test_logs": "testing.reports",
    "get_test_metadata": "testing.reports",
    "get_test_performance": "testing.reports",
    "get_test_progress": "testing.reports",
    "get_test_results": "testing.reports",
    "get_test_statistics": "testing.reports",
    "get_test_status": "testing.reports",
    "get_test_summary": "testing.reports",
    "get_test_timestamps": "testing.reports",
    "install_test_dependencies": "testing.environment",
    "run_test_category": "testing.runner",
    "run_test_stage": "testing.runner",
    "run_tests": "testing.runner",
    "setup_test_environment": "testing.environment",
    "validate_coverage_targets": "testing.environment",
    "validate_test_configuration": "testing.environment",
    "validate_test_dependencies": "testing.environment",
    "validate_test_environment": "testing.environment",
    # venv_utils
    "get_venv_python": "venv_utils",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve a re-exported name (PEP 562).

    The submodule owning ``name`` is imported on first access and the value is
    cached in the module globals. Any ImportError raised while importing the
    owning submodule propagates unchanged: utils/ submodules are in-tree, so a
    failure is a real bug that must surface, never be silently fallen back
    from.
    """
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), name)
    # Cache so subsequent lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include re-exported names in ``dir(utils)`` alongside module globals."""
    return sorted(set(globals()) | set(_EXPORT_MAP))


# Export all utilities
__all__: list[Any] = [
    # Utils availability flag
    "UTILS_AVAILABLE",
    # Logging utilities
    "PipelineLogger",
    "setup_step_logging",
    "setup_main_logging",
    "log_step_start",
    "log_step_success",
    "log_step_warning",
    "log_step_error",
    "log_section_header",
    "get_performance_summary",
    "setup_correlation_context",
    # Argument utilities
    "ArgumentParser",
    "PipelineArguments",
    "build_step_command_args",
    # Resource management
    "get_current_memory_usage",
    # Error recovery
    "ErrorRecoveryManager",
    "ErrorContext",
    "ErrorSeverity",
    "ErrorCodeRegistry",
    "format_error_message",
    "get_recovery_manager",
    "format_and_log_error",
    # Pipeline monitoring
    "generate_pipeline_health_report",
    # Pipeline validation
    "validate_step_prerequisites",
    "validate_pipeline_step_sequence",
    "StepConfiguration",
    "get_pipeline_step_info",
    "validate_pipeline_configuration",
    "parse_arguments",
    "validate_and_convert_paths",
    # Dependency utilities
    "DependencyValidator",
    "validate_pipeline_dependencies",
    "validate_pipeline_dependencies_if_available",
    "DependencySpec",
    "check_optional_dependencies",
    "get_dependency_status",
    "install_missing_dependencies",
    # Configuration utilities
    "GNNPipelineConfig",
    "PipelineConfig",
    "TypeCheckerConfig",
    "OntologyConfig",
    "LLMConfig",
    "WebsiteConfig",
    "SetupConfig",
    "SAPFConfig",
    "ModelConfig",
    "load_config",
    "save_config",
    "validate_config",
    "get_config_value",
    "set_config_value",
    # Performance utilities
    "PerformanceTracker",
    "performance_tracker",
    "track_operation_standalone",
    # Environment utilities
    "get_venv_python",
    "get_system_info",
    # Pipeline utilities
    "RecoveryArgumentParser",
    "get_pipeline_utilities",
    "validate_output_directory",
    "get_output_dir_for_script",
    "execute_pipeline_step_template",
    # Error handling utilities
    "PipelineErrorHandler",
    "PipelineErrorSeverity",
    "ErrorCategory",
    "RecoveryStrategy",
    "ExitCode",
    "generate_correlation_id",
    "handle_file_system_error",
    "handle_network_error",
    "handle_timeout_error",
    # Structured logging utilities
    "StructuredLogger",
    "get_pipeline_logger",
    "set_correlation_context",
    "log_pipeline_start",
    "log_pipeline_complete",
    "log_step_start",
    "log_step_success",
    "log_step_error",
    "log_step_warning",
    # Test utilities
    "TEST_CATEGORIES",
    "TEST_STAGES",
    "COVERAGE_TARGETS",
    "TEST_CONFIG",
    "TestRunner",
    "TestResult",
    "TestCategory",
    "TestStage",
    "CoverageTarget",
    "run_tests",
    "run_test_category",
    "run_test_stage",
    "get_test_results",
    "generate_test_report",
    "validate_test_environment",
    "setup_test_environment",
    "cleanup_test_environment",
    "get_test_coverage",
    "validate_coverage_targets",
    "get_test_summary",
    "get_test_statistics",
    "get_test_performance",
    "get_test_dependencies",
    "validate_test_dependencies",
    "install_test_dependencies",
    "get_test_configuration",
    "validate_test_configuration",
    "get_test_environment",
    "get_test_logs",
    "get_test_artifacts",
    "get_test_metadata",
    "get_test_timestamps",
    "get_test_duration",
    "get_test_status",
    "get_test_progress",
    # Base processor utilities
    "BaseProcessor",
    "ProcessingResult",
    "create_processor",
    # Pipeline template utilities (most-imported submodule, exposed here to avoid bypass)
    "create_standardized_pipeline_script",
]


def get_module_info() -> dict[str, Any]:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "utils",
        "version": __version__,
        "description": "Shared utilities, logging, and helper functions",
        "features": FEATURES,
    }
