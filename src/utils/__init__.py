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
        track_operation
    )
    
    from .venv_utils import (
        get_venv_python
    )
    
    from .system_utils import (
        get_system_info
    )
    
    from .shared_functions import (
        find_gnn_files,
        parse_gnn_sections,
        extract_model_parameters,
        create_processing_report,
        save_processing_report,
        validate_file_paths,
        ensure_output_directory,
        log_processing_summary
    )
    
    from .pipeline_template import (
        standard_module_function,
        create_standard_module_function,
        create_standard_pipeline_script,
        create_standardized_pipeline_script,
        get_standard_function_name,
        validate_module_function_signature,
        STANDARD_MODULE_FUNCTION_NAMES
    )
    
    # Flag to indicate utilities are available
    UTILS_AVAILABLE = True
    
    # Set up logging for the utils package itself
    _utils_logger = setup_step_logging("utils", verbose=False)
    _utils_logger.debug("GNN Pipeline utilities loaded successfully")
    
except ImportError as e:
    # Minimal fallback if utilities can't be imported
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _fallback_logger = logging.getLogger("utils_fallback")
    _fallback_logger.error(f"Critical error: Failed to import utilities: {e}")
    UTILS_AVAILABLE = False
    
    # Minimal compatibility stubs
    def setup_step_logging(step_name: str, verbose: bool = False):
        return logging.getLogger(step_name)
    
    def log_step_start(logger, message): logger.info(f"🚀 {message}")
    def log_step_success(logger, message): logger.info(f"✅ {message}")
    def log_step_warning(logger, message): logger.warning(f"⚠️ {message}")
    def log_step_error(logger, message): logger.error(f"❌ {message}")
    
    # Minimal stubs for other functions
    setup_main_logging = setup_step_logging
    get_performance_summary = lambda: {"error": "utils_unavailable"}
    get_venv_python = lambda x: (None, None)
    get_system_info = lambda: {"error": "utils_unavailable"}
    validate_output_directory = lambda x, y: False
    performance_tracker = None
    
    class MockArgumentParser:
        @staticmethod
        def parse_step_arguments(step_name): 
            import argparse
            return argparse.Namespace(target_dir=Path("input"), output_dir=Path("output"), verbose=False)

# Performance tracker is already imported with canonical names

# Convenience function for pipeline modules to get all they need in one import
def get_pipeline_utilities(step_name: str, verbose: bool = False) -> Tuple[Any, ...]:
    """
    Get all essential utilities for a pipeline step in one call.
    
    Args:
        step_name: Name of the pipeline step
        verbose: Whether to enable verbose logging
        
    Returns:
        Tuple of (logger, log_step_start, log_step_success, log_step_warning, log_step_error)
    """
    logger = setup_step_logging(step_name, verbose)
    return logger, log_step_start, log_step_success, log_step_warning, log_step_error

def validate_output_directory(output_dir: Path, step_name: str) -> bool:
    """
    Validate and create output directory for a pipeline step.
    
    Args:
        output_dir: Base output directory
        step_name: Name of the step (for creating subdirectory)
        
    Returns:
        True if directory is ready, False otherwise
    """
    try:
        step_output_dir = output_dir / f"{step_name}_step" 
        step_output_dir.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

def execute_pipeline_step_template(
    step_name: str,
    step_description: str,
    main_function,
    import_dependencies: Optional[List[str]] = None
):
    """
    Standardized template for executing pipeline steps with consistent error handling.
    
    This function provides a uniform execution pattern for all pipeline steps:
    - Standardized argument parsing with fallback
    - Consistent logging setup
    - Error handling with proper exit codes
    - Performance tracking when available
    - Dependency validation
    
    Args:
        step_name: Name of the step (e.g., "1_setup.py")
        step_description: Description of what the step does
        main_function: Function to execute the step logic
        import_dependencies: List of module names to validate before execution
    """
    # Implementation would go here
    pass

# Export commonly used items at package level
__all__ = [
    'PipelineLogger',
    'EnhancedPipelineLogger',
    'setup_step_logging', 
    'setup_main_logging',
    'log_step_start',
    'log_step_success', 
    'log_step_warning',
    'log_step_error',
    'log_section_header',
    'ArgumentParser',
    'EnhancedArgumentParser',
    'PipelineArguments',
    'StepConfiguration',
    'DependencyValidator',
    'validate_pipeline_dependencies',
    'get_pipeline_utilities',
    'validate_output_directory',
    'get_performance_summary',
    'execute_pipeline_step_template',
    'get_venv_python',
    'get_system_info',
    'parse_arguments',
    'validate_and_convert_paths',
    'validate_pipeline_dependencies_if_available',
    'UTILS_AVAILABLE'
] 