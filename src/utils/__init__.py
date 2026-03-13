# This file marks the directory as a Python package.
# It can also be used to define package-level exports or initialization code.

# This file makes utils a package

"""
GNN Pipeline Utilities Package

184 exported names aggregated from 20 source modules. All pipeline steps import
from this package for consistency. The surface area is intentionally wide; a
future pass should split it by concern into sub-packages.

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
- performance_tracker: Operation timing and performance metrics
- step_logging: Minimal, always-importable logging functions (no external deps)
- base_processor: Abstract base class for standardized step processors
- venv_utils: Virtual environment path helpers
- system_utils: System information gathering
- test_utils: Test runner, categories, stages, and coverage targets
- pipeline: Previous utilities re-exported for backwards compatibility
- error_handling: Structured error handler, categories, and recovery strategies
- structured_logging: Structured log emission with correlation context
- dependency_audit: Dependency auditing and optimization utilities
"""

import logging
from pathlib import Path
from typing import Optional, Any, Tuple, Dict, List

# Flag to indicate utils are available (used by pipeline modules)
UTILS_AVAILABLE = True

# Import and expose the main classes and functions for easy access
try:
    from .logging_utils import (
        PipelineLogger,
        setup_step_logging,
        setup_main_logging,
        log_section_header,
        get_performance_summary,
        PerformanceTracker,
        performance_tracker,
        setup_correlation_context
    )
    # log_step_{start,success,warning,error} come from structured_logging below

    from .argument_utils import (
        ArgumentParser,
        PipelineArguments,
        build_step_command_args,
        get_step_output_dir,
        StepConfiguration,
        get_pipeline_step_info,
        validate_pipeline_configuration,
        parse_arguments,
        validate_and_convert_paths
    )

    from .resource_manager import (
        get_current_memory_usage
    )

    from .error_recovery import (
        ErrorRecoveryManager,
        ErrorContext,
        ErrorSeverity,
        ErrorCodeRegistry,
        format_error_message,
        get_recovery_manager,
        format_and_log_error
    )

    from .pipeline_monitor import (
        generate_pipeline_health_report
    )

    from .pipeline_validator import (
        validate_step_prerequisites,
        validate_pipeline_step_sequence
    )

    from .pipeline_planner import (
        generate_execution_plan
    )

    from .dependency_validator import (
        DependencyValidator,
        validate_pipeline_dependencies,
        validate_pipeline_dependencies_if_available,
        DependencySpec,
        check_optional_dependencies,
        get_dependency_status,
        install_missing_dependencies
    )

    from .config_loader import (
        GNNPipelineConfig,
        PipelineConfig,
        TypeCheckerConfig,
        OntologyConfig,
        LLMConfig,
        WebsiteConfig,
        SetupConfig,
        SAPFConfig,
        ModelConfig,
        load_config,
        save_config,
        validate_config,
        get_config_value,
        set_config_value
    )

    from .performance_tracker import (
        PerformanceTracker,
        performance_tracker,
        track_operation_standalone
    )

    # Base processor - abstract base class for standardized processing
    from .base_processor import (
        BaseProcessor,
        ProcessingResult,
        create_processor
    )

    from .venv_utils import (
        get_venv_python
    )

    from .system_utils import (
        get_system_info
    )

    from .test_utils import (
        TEST_CATEGORIES,
        TEST_STAGES,
        COVERAGE_TARGETS,
        TEST_CONFIG,
        TestRunner,
        TestResult,
        TestCategory,
        TestStage,
        CoverageTarget,
        run_tests,
        run_test_category,
        run_test_stage,
        get_test_results,
        generate_test_report,
        validate_test_environment,
        setup_test_environment,
        cleanup_test_environment,
        get_test_coverage,
        validate_coverage_targets,
        get_test_summary,
        get_test_statistics,
        get_test_performance,
        get_test_dependencies,
        validate_test_dependencies,
        install_test_dependencies,
        get_test_configuration,
        validate_test_configuration,
        get_test_environment,
        get_test_logs,
        get_test_artifacts,
        get_test_metadata,
        get_test_timestamps,
        get_test_duration,
        get_test_status,
        get_test_progress
    )

    from .pipeline import (
        RecoveryArgumentParser,
        get_pipeline_utilities,
        validate_output_directory,
        execute_pipeline_step_template,
        get_output_dir_for_script
    )

    from .pipeline_template import (
        create_standardized_pipeline_script,
    )

    # Import error handling framework
    from .error_handling import (
        PipelineErrorHandler,
        PipelineErrorSeverity,
        ErrorCategory,
        RecoveryStrategy,
        ExitCode,
        generate_correlation_id,
        handle_file_system_error,
        handle_network_error,
        handle_timeout_error
    )

    # Import structured logging
    from .structured_logging import (
        StructuredLogger,
        get_pipeline_logger,
        set_correlation_context,
        log_pipeline_start,
        log_pipeline_complete,
        log_step_start,
        log_step_success,
        log_step_error,
        log_step_warning
    )

    # Import dependency audit system
    from .dependency_audit import (
        DependencyAuditor,
        DependencyOptimizer,
        DependencyInfo,
        AuditResult,
        audit_project_dependencies,
        optimize_project_dependencies
    )

except ImportError as e:
    # Import recovery functions when modules are not available
    logging.warning(f"Some utils modules not available: {e}")
    from .recovery import (
        RecoveryArgumentParser,
        setup_step_logging
    )

# Export all utilities
__all__ = [
    # Utils availability flag
    'UTILS_AVAILABLE',

    # Logging utilities
    'PipelineLogger',
    'setup_step_logging',
    'setup_main_logging',
    'log_step_start',
    'log_step_success',
    'log_step_warning',
    'log_step_error',
    'log_section_header',
    'get_performance_summary',
    'PerformanceTracker',
    'performance_tracker',
    'setup_correlation_context',

    # Argument utilities
    'ArgumentParser',
    'PipelineArguments',
    'build_step_command_args',

    # Resource management
    'get_current_memory_usage',

    # Error recovery
    'ErrorRecoveryManager',
    'ErrorContext',
    'ErrorSeverity',
    'ErrorCodeRegistry',
    'format_error_message',
    'get_recovery_manager',
    'format_and_log_error',

    # Pipeline monitoring
    'generate_pipeline_health_report',

    # Pipeline validation
    'validate_step_prerequisites',
    'validate_pipeline_step_sequence',

    # Pipeline planning
    'generate_execution_plan',
    'get_step_output_dir',
    'StepConfiguration',
    'get_pipeline_step_info',
    'validate_pipeline_configuration',
    'parse_arguments',
    'validate_and_convert_paths',

    # Dependency utilities
    'DependencyValidator',
    'validate_pipeline_dependencies',
    'validate_pipeline_dependencies_if_available',
    'DependencySpec',
    'check_optional_dependencies',
    'get_dependency_status',
    'install_missing_dependencies',

    # Configuration utilities
    'GNNPipelineConfig',
    'PipelineConfig',
    'TypeCheckerConfig',
    'OntologyConfig',
    'LLMConfig',
    'WebsiteConfig',
    'SetupConfig',
    'SAPFConfig',
    'ModelConfig',
    'load_config',
    'save_config',
    'validate_config',
    'get_config_value',
    'set_config_value',

    # Performance utilities
    'PerformanceTracker',
    'performance_tracker',
    'track_operation_standalone',

    # Environment utilities
    'get_venv_python',
    'get_system_info',

    # Pipeline utilities
    'RecoveryArgumentParser',
    'get_pipeline_utilities',
    'validate_output_directory',
    'get_output_dir_for_script',
    'execute_pipeline_step_template',

    # Error handling utilities
    'PipelineErrorHandler',
    'PipelineErrorSeverity',
    'ErrorCategory',
    'RecoveryStrategy',
    'ExitCode',
    'generate_correlation_id',
    'handle_file_system_error',
    'handle_network_error',
    'handle_timeout_error',

    # Structured logging utilities
    'StructuredLogger',
    'get_pipeline_logger',
    'set_correlation_context',
    'log_pipeline_start',
    'log_pipeline_complete',
    'log_step_start',
    'log_step_success',
    'log_step_error',
    'log_step_warning',

    # Dependency audit utilities
    'DependencyAuditor',
    'DependencyOptimizer',
    'DependencyInfo',
    'AuditResult',
    'audit_project_dependencies',
    'optimize_project_dependencies',

    # Test utilities
    'TEST_CATEGORIES',
    'TEST_STAGES',
    'COVERAGE_TARGETS',
    'TEST_CONFIG',
    'TestRunner',
    'TestResult',
    'TestCategory',
    'TestStage',
    'CoverageTarget',
    'run_tests',
    'run_test_category',
    'run_test_stage',
    'get_test_results',
    'generate_test_report',
    'validate_test_environment',
    'setup_test_environment',
    'cleanup_test_environment',
    'get_test_coverage',
    'validate_coverage_targets',
    'get_test_summary',
    'get_test_statistics',
    'get_test_performance',
    'get_test_dependencies',
    'validate_test_dependencies',
    'install_test_dependencies',
    'get_test_configuration',
    'validate_test_configuration',
    'get_test_environment',
    'get_test_logs',
    'get_test_artifacts',
    'get_test_metadata',
    'get_test_timestamps',
    'get_test_duration',
    'get_test_status',
    'get_test_progress',

    # Base processor utilities
    'BaseProcessor',
    'ProcessingResult',
    'create_processor',

    # Pipeline template utilities (most-imported submodule, exposed here to avoid bypass)
    'create_standardized_pipeline_script',
]
