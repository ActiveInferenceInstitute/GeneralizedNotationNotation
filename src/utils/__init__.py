# This file marks the directory as a Python package.
# It can also be used to define package-level exports or initialization code.

# This file makes utils a package 

"""
GNN Pipeline Utilities Package

This package provides streamlined utilities for the GNN processing pipeline:
- logging_utils: Centralized, correlation-aware logging system
- argument_utils: Streamlined argument parsing and validation
- dependency_validator: Comprehensive dependency validation

All pipeline modules should import from this package for consistency.
"""

import logging
from pathlib import Path
from typing import Optional, Any, Tuple, Dict

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
        performance_tracker
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
        validate_pipeline_configuration
    )
    
    from .dependency_validator import (
        DependencyValidator,
        validate_pipeline_dependencies,
        DependencySpec
    )
    
    # Flag to indicate utilities are available
    UTILS_AVAILABLE = True
    
    # Set up logging for the utils package itself
    _utils_logger = setup_step_logging("utils", verbose=False)
    _utils_logger.debug("GNN Pipeline utilities loaded successfully")
    
except ImportError as e:
    # Graceful fallback if utilities can't be imported
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _fallback_logger = logging.getLogger("utils_fallback")
    _fallback_logger.warning(f"Failed to import some utilities: {e}")
    UTILS_AVAILABLE = False
    
    # Provide basic fallbacks that maintain interface compatibility
    class PipelineLogger:
        @staticmethod
        def get_logger(name: str):
            return logging.getLogger(name)
        
        @staticmethod
        def set_correlation_context(step_name: str, correlation_id: Optional[str] = None):
            return correlation_id or "fallback"
    
    def setup_step_logging(step_name: str, verbose: bool = False) -> logging.Logger:
        """Fallback logging setup that maintains interface compatibility."""
        level = logging.DEBUG if verbose else logging.INFO
        logger = logging.getLogger(step_name)
        logger.setLevel(level)
        
        # Ensure handler exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [fallback] %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def log_step_start(logger_or_step_name, message: str = None, **kwargs):
        """Fallback for log_step_start with flexible signature."""
        if isinstance(logger_or_step_name, str):
            logger = logging.getLogger(logger_or_step_name)
            message = message or f"Starting {logger_or_step_name}"
        else:
            logger = logger_or_step_name
            message = message or "Starting step"
        logger.info(f"🚀 {message}")
    
    def log_step_success(logger_or_step_name, message: str = None, **kwargs):
        """Fallback for log_step_success with flexible signature."""
        if isinstance(logger_or_step_name, str):
            logger = logging.getLogger(logger_or_step_name)
            message = message or f"{logger_or_step_name} completed successfully"
        else:
            logger = logger_or_step_name
            message = message or "Step completed successfully"
        logger.info(f"✅ {message}")
    
    def log_step_warning(logger_or_step_name, message: str = None, **kwargs):
        """Fallback for log_step_warning with flexible signature."""
        if isinstance(logger_or_step_name, str):
            logger = logging.getLogger(logger_or_step_name)
            message = message or f"Warning in {logger_or_step_name}"
        else:
            logger = logger_or_step_name
            message = message or "Step warning"
        logger.warning(f"⚠️ {message}")
    
    def log_step_error(logger_or_step_name, message: str = None, **kwargs):
        """Fallback for log_step_error with flexible signature."""
        if isinstance(logger_or_step_name, str):
            logger = logging.getLogger(logger_or_step_name)
            message = message or f"Error in {logger_or_step_name}"
        else:
            logger = logger_or_step_name
            message = message or "Step error"
        logger.error(f"❌ {message}")

    # Minimal fallbacks for other utilities
    def setup_main_logging(log_dir: Optional[Path] = None, verbose: bool = False):
        return setup_step_logging("main", verbose)
    
    def get_performance_summary() -> Dict[str, Any]:
        return {"fallback": True, "performance_tracking": "unavailable"}

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
    'PipelineArguments',
    'StepConfiguration',
    'DependencyValidator',
    'validate_pipeline_dependencies',
    'get_pipeline_utilities',
    'validate_output_directory',
    'get_performance_summary',
    'UTILS_AVAILABLE'
] 