# This file marks the directory as a Python package.
# It can also be used to define package-level exports or initialization code.

# This file makes utils a package 

"""
GNN Pipeline Utilities Package

This package provides streamlined utilities for the GNN processing pipeline:
- logging_utils: Centralized, correlation-aware logging system
- argument_utils: Streamlined argument parsing and validation
- dependency_validator: Comprehensive dependency validation
- config_loader: YAML configuration loading and validation

All pipeline modules should import from this package for consistency.
"""

import logging
from pathlib import Path
from typing import Optional, Any, Tuple, Dict, List

# Import and expose the main classes and functions for easy access
try:
    from .logging_utils import (
        PipelineLogger,
        EnhancedPipelineLogger, 
        setup_step_logging,
        setup_main_logging,
        setup_enhanced_step_logging,
        log_step_start,
        log_step_success,
        log_step_warning,
        log_step_error,
        log_section_header,
        get_performance_summary,
        PerformanceTracker,
        performance_tracker,
        setup_correlation_context
    )
    
    from .argument_utils import (
        ArgumentParser,
        EnhancedArgumentParser,
        PipelineArguments,
        build_step_command_args,
        build_enhanced_step_command_args,
        get_step_output_dir,
        StepConfiguration,
        get_pipeline_step_info,
        validate_pipeline_configuration,
        parse_arguments,
        validate_and_convert_paths
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
        validate_test_environment,
        setup_test_environment,
        cleanup_test_environment,
        get_test_logs,
        get_test_artifacts,
        get_test_metadata,
        get_test_timestamps,
        get_test_duration,
        get_test_status,
        get_test_progress,
        get_test_summary,
        get_test_statistics,
        get_test_performance,
        get_test_dependencies,
        validate_test_dependencies,
        install_test_dependencies,
        get_test_configuration,
        validate_test_configuration,
        get_test_environment,
        validate_test_environment,
        setup_test_environment,
        cleanup_test_environment,
        get_test_logs,
        get_test_artifacts,
        get_test_metadata,
        get_test_timestamps,
        get_test_duration,
        get_test_status,
        get_test_progress
    )
    
    from .pipeline import (
        setup_step_logging,
        MockArgumentParser,
        get_pipeline_utilities,
        validate_output_directory,
        execute_pipeline_step_template
    )
    
except ImportError as e:
    # Import fallback functions when modules are not available
    logging.warning(f"Some utils modules not available: {e}")
    from .fallback import (
        MockArgumentParser,
        setup_step_logging
    )

# Export all utilities
__all__ = [
    # Logging utilities
    'PipelineLogger',
    'EnhancedPipelineLogger', 
    'setup_step_logging',
    'setup_main_logging',
    'setup_enhanced_step_logging',
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
    'EnhancedArgumentParser',
    'PipelineArguments',
    'build_step_command_args',
    'build_enhanced_step_command_args',
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
    'MockArgumentParser',
    'get_pipeline_utilities',
    'validate_output_directory',
    'execute_pipeline_step_template',
    
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
    'get_test_progress'
] 