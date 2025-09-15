#!/usr/bin/env python3
"""
Standardized Error Handling Module

This module provides standardized error handling utilities for the GNN pipeline.
"""

import logging
import sys
from typing import Dict, Any, Optional, Callable, Type
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class ErrorHandler:
    """Standardized error handler for the GNN pipeline."""
    
    def __init__(self, module_name: str = "unknown", step_name: str = None):
        self.module_name = module_name
        self.step_name = step_name or module_name
        self.error_count = 0
        self.warning_count = 0
        self.correlation_id = f"{module_name}_{id(self)}"
    
    def handle_error(self, error: Exception, context: str = "", 
                    recoverable: bool = True, **kwargs) -> Dict[str, Any]:
        """Handle an error with standardized logging and recovery."""
        self.error_count += 1
        
        error_info = {
            "module": self.module_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "recoverable": recoverable,
            "error_count": self.error_count,
            **kwargs
        }
        
        if recoverable:
            logger.warning(f"Recoverable error in {self.module_name}: {error} (context: {context})")
        else:
            logger.error(f"Critical error in {self.module_name}: {error} (context: {context})")
        
        return error_info
    
    def handle_warning(self, message: str, context: str = "", **kwargs) -> Dict[str, Any]:
        """Handle a warning with standardized logging."""
        self.warning_count += 1
        
        warning_info = {
            "module": self.module_name,
            "warning_message": message,
            "context": context,
            "warning_count": self.warning_count,
            **kwargs
        }
        
        logger.warning(f"Warning in {self.module_name}: {message} (context: {context})")
        return warning_info
    
    def get_stats(self) -> Dict[str, int]:
        """Get error and warning statistics."""
        return {
            "error_count": self.error_count,
            "warning_count": self.warning_count
        }
    
    def error_context(self, context: str):
        """Context manager for error handling."""
        class ErrorContext:
            def __init__(self, handler, context):
                self.handler = handler
                self.context = context
            
            def __enter__(self):
                return self.handler
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    self.handler.handle_error(exc_val, self.context)
                return False  # Don't suppress exceptions
        
        return ErrorContext(self, context)

def create_error_handler(module_name: str, step_name: str = None) -> ErrorHandler:
    """Create a standardized error handler for a module."""
    return ErrorHandler(module_name, step_name)

def safe_execute(func: Callable, *args, **kwargs) -> tuple[bool, Any, Optional[Exception]]:
    """Safely execute a function with error handling."""
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}")
        return False, None, e

def safe_file_operation(operation: Callable, file_path: Path, *args, **kwargs) -> tuple[bool, Any, Optional[Exception]]:
    """Safely perform a file operation with error handling."""
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        result = operation(file_path, *args, **kwargs)
        return True, result, None
    except Exception as e:
        logger.error(f"Error in file operation on {file_path}: {e}")
        return False, None, e

def validate_environment() -> Dict[str, Any]:
    """Validate the execution environment."""
    validation_result = {
        "python_version": sys.version_info,
        "platform": sys.platform,
        "path_separator": Path.sep,
        "current_directory": Path.cwd(),
        "errors": [],
        "warnings": []
    }
    
    # Check Python version
    if sys.version_info < (3, 8):
        validation_result["errors"].append("Python 3.8+ required")
    
    # Check current directory
    if not Path.cwd().exists():
        validation_result["errors"].append("Current directory does not exist")
    
    return validation_result

# Global error handler instance
_global_error_handler = ErrorHandler("global")

def get_global_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler

def log_error(error: Exception, context: str = "", **kwargs) -> Dict[str, Any]:
    """Log an error using the global error handler."""
    return _global_error_handler.handle_error(error, context, **kwargs)

def log_warning(message: str, context: str = "", **kwargs) -> Dict[str, Any]:
    """Log a warning using the global error handler."""
    return _global_error_handler.handle_warning(message, context, **kwargs)
