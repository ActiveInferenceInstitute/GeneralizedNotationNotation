# This file makes utils a package

"""
GNN Pipeline Utilities Package

Lazy PEP 562 re-export surface: 118 exported names aggregated from 19 source
modules. All pipeline steps import from this package for consistency. The
surface area is intentionally wide; a future pass should split it by concern
into sub-packages.

Importing ``utils`` is intentionally LIGHT: no submodule executes at import
time, so heavy module-scope dependencies (psutil via structured_logging /
resource_manager, matplotlib via simulation_utils) are only paid when an
exported name is actually resolved through ``__getattr__``.

Source modules:
- logging_utils: Centralized, correlation-aware logging system
- argument_utils: Streamlined argument parsing and validation
- resource_manager: Memory and resource usage tracking
- error_recovery: Error context, severity, and recovery management
- pipeline_monitor: Pipeline health reporting
- pipeline_validator: Pre-execution prerequisite checker (step output validation)
- pipeline_planner: Execution plan generation
- dependency_validator: Comprehensive dependency validation
- config_loader: YAML configuration loading and validation (active config system)
- performance_tracking: Operation timing and performance metrics
- step_logging: Minimal, always-importable logging functions (no external deps)
- base_processor: Abstract base class for standardized step processors
- venv_utils: Virtual environment path helpers
- system_utils: System information gathering
- test_utils: Test runner, categories, stages, and coverage targets
- pipeline: Pipeline utility exports
- error_handling: Structured error handler, categories, and recovery strategies
- structured_logging: Structured log emission with correlation context
- dependency_audit: Dependency auditing and optimization utilities
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Static re-export surface for type checkers: mirrors the pre-lazy eager
    # imports so mypy resolves names without executing submodules at runtime.
    from .argument_utils import (
        ArgumentParser,
        PipelineArguments,
        StepConfiguration,
        build_step_command_args,
        get_pipeline_step_info,
        parse_arguments,
        validate_and_convert_paths,
        validate_pipeline_configuration,
    )
    from .base_processor import BaseProcessor, ProcessingResult, create_processor
    from .config_loader import (
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
    from .dependency_audit import (
        AuditResult,
        DependencyAuditor,
        DependencyInfo,
        DependencyOptimizer,
        audit_project_dependencies,
        optimize_project_dependencies,
    )
    from .dependency_validator import (
        DependencySpec,
        DependencyValidator,
        check_optional_dependencies,
        get_dependency_status,
        install_missing_dependencies,
        validate_pipeline_dependencies,
        validate_pipeline_dependencies_if_available,
    )
    from .error_handling import (
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
    from .error_recovery import (
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
    from .performance_tracking import (
        PerformanceTracker,
        performance_tracker,
        track_operation_standalone,
    )
    from .pipeline import (
        RecoveryArgumentParser,
        execute_pipeline_step_template,
        get_output_dir_for_script,
        get_pipeline_utilities,
        validate_output_directory,
    )
    from .pipeline_monitor import generate_pipeline_health_report
    from .pipeline_planner import generate_execution_plan
    from .pipeline_template import create_standardized_pipeline_script
    from .pipeline_validator import (
        validate_pipeline_step_sequence,
        validate_step_prerequisites,
    )
    from .resource_manager import get_current_memory_usage
    from .structured_logging import (
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
    from .system_utils import get_system_info
    from .test_utils import (
        COVERAGE_TARGETS,
        TEST_CATEGORIES,
        TEST_CONFIG,
        TEST_STAGES,
        CoverageTarget,
        TestCategory,
        TestResult,
        TestRunner,
        TestStage,
        cleanup_test_environment,
        generate_test_report,
        get_test_artifacts,
        get_test_configuration,
        get_test_coverage,
        get_test_dependencies,
        get_test_duration,
        get_test_environment,
        get_test_logs,
        get_test_metadata,
        get_test_performance,
        get_test_progress,
        get_test_results,
        get_test_statistics,
        get_test_status,
        get_test_summary,
        get_test_timestamps,
        install_test_dependencies,
        run_test_category,
        run_test_stage,
        run_tests,
        setup_test_environment,
        validate_coverage_targets,
        validate_test_configuration,
        validate_test_dependencies,
        validate_test_environment,
    )
    from .venv_utils import get_venv_python

__version__ = "1.6.0"

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
    # argument_utils
    "ArgumentParser": "argument_utils",
    "PipelineArguments": "argument_utils",
    "StepConfiguration": "argument_utils",
    "build_step_command_args": "argument_utils",
    "get_pipeline_step_info": "argument_utils",
    "parse_arguments": "argument_utils",
    "validate_and_convert_paths": "argument_utils",
    "validate_pipeline_configuration": "argument_utils",
    # base_processor
    "BaseProcessor": "base_processor",
    "ProcessingResult": "base_processor",
    "create_processor": "base_processor",
    # config_loader
    "GNNPipelineConfig": "config_loader",
    "LLMConfig": "config_loader",
    "ModelConfig": "config_loader",
    "OntologyConfig": "config_loader",
    "PipelineConfig": "config_loader",
    "SAPFConfig": "config_loader",
    "SetupConfig": "config_loader",
    "TypeCheckerConfig": "config_loader",
    "WebsiteConfig": "config_loader",
    "get_config_value": "config_loader",
    "load_config": "config_loader",
    "save_config": "config_loader",
    "set_config_value": "config_loader",
    "validate_config": "config_loader",
    # dependency_audit
    "AuditResult": "dependency_audit",
    "DependencyAuditor": "dependency_audit",
    "DependencyInfo": "dependency_audit",
    "DependencyOptimizer": "dependency_audit",
    "audit_project_dependencies": "dependency_audit",
    "optimize_project_dependencies": "dependency_audit",
    # dependency_validator
    "DependencySpec": "dependency_validator",
    "DependencyValidator": "dependency_validator",
    "check_optional_dependencies": "dependency_validator",
    "get_dependency_status": "dependency_validator",
    "install_missing_dependencies": "dependency_validator",
    "validate_pipeline_dependencies": "dependency_validator",
    "validate_pipeline_dependencies_if_available": "dependency_validator",
    # error_handling
    "ErrorCategory": "error_handling",
    "ExitCode": "error_handling",
    "PipelineErrorHandler": "error_handling",
    "PipelineErrorSeverity": "error_handling",
    "RecoveryStrategy": "error_handling",
    "generate_correlation_id": "error_handling",
    "handle_file_system_error": "error_handling",
    "handle_network_error": "error_handling",
    "handle_timeout_error": "error_handling",
    # error_recovery
    "ErrorCodeRegistry": "error_recovery",
    "ErrorContext": "error_recovery",
    "ErrorRecoveryManager": "error_recovery",
    "ErrorSeverity": "error_recovery",
    "format_and_log_error": "error_recovery",
    "format_error_message": "error_recovery",
    "get_recovery_manager": "error_recovery",
    # logging_utils
    "PipelineLogger": "logging_utils",
    "get_performance_summary": "logging_utils",
    "log_section_header": "logging_utils",
    "setup_correlation_context": "logging_utils",
    "setup_main_logging": "logging_utils",
    "setup_step_logging": "logging_utils",
    # performance_tracking (renamed from performance_tracker.py: the exported
    # object must not share its module's name, or any prior
    # 'import utils.performance_tracker' shadows the re-export with the module)
    "PerformanceTracker": "performance_tracking",
    "performance_tracker": "performance_tracking",
    "track_operation_standalone": "performance_tracking",
    # pipeline
    "RecoveryArgumentParser": "pipeline",
    "execute_pipeline_step_template": "pipeline",
    "get_output_dir_for_script": "pipeline",
    "get_pipeline_utilities": "pipeline",
    "validate_output_directory": "pipeline",
    # pipeline_monitor
    "generate_pipeline_health_report": "pipeline_monitor",
    # pipeline_planner
    "generate_execution_plan": "pipeline_planner",
    # pipeline_template
    "create_standardized_pipeline_script": "pipeline_template",
    # pipeline_validator
    "validate_pipeline_step_sequence": "pipeline_validator",
    "validate_step_prerequisites": "pipeline_validator",
    # resource_manager
    "get_current_memory_usage": "resource_manager",
    # structured_logging
    "StructuredLogger": "structured_logging",
    "get_pipeline_logger": "structured_logging",
    "log_pipeline_complete": "structured_logging",
    "log_pipeline_start": "structured_logging",
    "log_step_error": "structured_logging",
    "log_step_start": "structured_logging",
    "log_step_success": "structured_logging",
    "log_step_warning": "structured_logging",
    "set_correlation_context": "structured_logging",
    # system_utils
    "get_system_info": "system_utils",
    # test_utils
    "COVERAGE_TARGETS": "test_utils",
    "TEST_CATEGORIES": "test_utils",
    "TEST_CONFIG": "test_utils",
    "TEST_STAGES": "test_utils",
    "CoverageTarget": "test_utils",
    "TestCategory": "test_utils",
    "TestResult": "test_utils",
    "TestRunner": "test_utils",
    "TestStage": "test_utils",
    "cleanup_test_environment": "test_utils",
    "generate_test_report": "test_utils",
    "get_test_artifacts": "test_utils",
    "get_test_configuration": "test_utils",
    "get_test_coverage": "test_utils",
    "get_test_dependencies": "test_utils",
    "get_test_duration": "test_utils",
    "get_test_environment": "test_utils",
    "get_test_logs": "test_utils",
    "get_test_metadata": "test_utils",
    "get_test_performance": "test_utils",
    "get_test_progress": "test_utils",
    "get_test_results": "test_utils",
    "get_test_statistics": "test_utils",
    "get_test_status": "test_utils",
    "get_test_summary": "test_utils",
    "get_test_timestamps": "test_utils",
    "install_test_dependencies": "test_utils",
    "run_test_category": "test_utils",
    "run_test_stage": "test_utils",
    "run_tests": "test_utils",
    "setup_test_environment": "test_utils",
    "validate_coverage_targets": "test_utils",
    "validate_test_configuration": "test_utils",
    "validate_test_dependencies": "test_utils",
    "validate_test_environment": "test_utils",
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
    # Pipeline planning
    "generate_execution_plan",
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
    # Dependency audit utilities
    "DependencyAuditor",
    "DependencyOptimizer",
    "DependencyInfo",
    "AuditResult",
    "audit_project_dependencies",
    "optimize_project_dependencies",
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


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "utils",
        "version": __version__,
        "description": "Shared utilities, logging, and helper functions",
        "features": FEATURES,
    }
