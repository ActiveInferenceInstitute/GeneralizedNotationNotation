#!/usr/bin/env python3
"""
Logging Utilities Compatibility Shim for GNN Processing Pipeline.

This module provides the legacy logging interface but delegates all operations
to the modern, structured logging system in `utils.logging.logging_utils`.
"""

import logging
import warnings
from typing import Any, Dict, Optional
from pathlib import Path

# Import everything from the new system
from .logging.logging_utils import (
    PipelineLogger as NewPipelineLogger,
    setup_step_logging as new_setup_step_logging,
    setup_main_logging as new_setup_main_logging,
    log_step_start as new_log_step_start,
    log_step_success as new_log_step_success,
    log_step_warning as new_log_step_warning,
    log_step_error as new_log_step_error,
    log_section_header as new_log_section_header,
    setup_correlation_context as new_setup_correlation_context,
)

# Re-export performance tracker
try:
    from .performance_tracker import PerformanceTracker, performance_tracker
except ImportError:
    # Minimal stub if not available
    class PerformanceTracker:
        def __init__(self): pass
        def get_summary(self): return {}
    performance_tracker = PerformanceTracker()

class PipelineLogger:
    """Legacy wrapper for PipelineLogger."""
    
    @classmethod
    def get_logger(cls, name: str, level: int = logging.INFO) -> logging.Logger:
        """Get a logger, ensuring the new system is initialized."""
        return NewPipelineLogger.get_logger(name)

    @classmethod
    def setup(cls, log_dir: Path = None, verbose: bool = False):
        """Setup the logging system via the new implementation."""
        NewPipelineLogger.initialize(log_dir=log_dir)
        NewPipelineLogger.set_verbosity(verbose)

def setup_step_logging(step_name: str, verbose: bool = False) -> logging.Logger:
    """Setup logging for a pipeline step."""
    return new_setup_step_logging(step_name, verbose)

def setup_main_logging(verbose: bool = False) -> logging.Logger:
    """Setup main pipeline logging."""
    return new_setup_main_logging(verbose=verbose)

def log_step_start(logger: logging.Logger, message: str):
    """Log the start of a step."""
    new_log_step_start(logger, message)

def log_step_success(logger: logging.Logger, message: str):
    """Log a successful step completion."""
    new_log_step_success(logger, message)

def log_step_warning(logger: logging.Logger, message: str):
    """Log a warning during step execution."""
    new_log_step_warning(logger, message)

def log_step_error(logger: logging.Logger, message: str):
    """Log an error during step execution."""
    new_log_step_error(logger, message)

def log_section_header(logger: logging.Logger, title: str, char: str = "="):
    """Log a section header."""
    new_log_section_header(logger, title, char)

def get_performance_summary() -> Dict[str, Any]:
    """Get a summary of performance metrics."""
    return performance_tracker.get_summary()

def setup_correlation_context(correlation_id: Optional[str] = None, step_name: Optional[str] = None):
    """Set up correlation context for logging."""
    new_setup_correlation_context(step_name or "unknown", correlation_id)

# Export all public symbols for compatibility
__all__ = [
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
]
