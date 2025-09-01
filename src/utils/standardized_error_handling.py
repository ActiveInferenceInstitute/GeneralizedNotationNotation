#!/usr/bin/env python3
"""
Standardized Error Handling for GNN Pipeline

This module provides a unified error handling interface that all pipeline 
modules can use for consistent error reporting, logging, and recovery.
"""

import logging
import uuid
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from pathlib import Path

from .error_handling import PipelineErrorHandler, ErrorCategory, ErrorSeverity
from .logging_utils import setup_step_logging, log_step_error, log_step_warning, log_step_success


class StandardizedErrorHandler:
    """
    Standardized error handler that provides consistent error handling
    patterns across all pipeline modules.
    """
    
    def __init__(self, step_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize standardized error handler.
        
        Args:
            step_name: Name of the pipeline step or module
            logger: Optional logger instance (creates one if not provided)
        """
        self.step_name = step_name
        self.logger = logger or setup_step_logging(step_name)
        self.correlation_id = str(uuid.uuid4())[:8]
        self.pipeline_error_handler = PipelineErrorHandler(self.logger, self.correlation_id)
        
    @contextmanager
    def error_context(self, operation: str, **context):
        """
        Context manager for standardized error handling in operations.
        
        Args:
            operation: Description of the operation being performed
            **context: Additional context information
        """
        try:
            self.logger.debug(f"[{self.correlation_id}] Starting {operation}")
            yield self
        except Exception as e:
            # Create structured error
            error_context = {"operation": operation, **context}
            pipeline_error = self.pipeline_error_handler.create_error(
                step_name=self.step_name,
                error=e,
                category=self._categorize_error(e),
                context=error_context
            )
            
            # Handle the error according to its recovery strategy
            exit_code = self.pipeline_error_handler.handle_error(pipeline_error)
            
            # Re-raise if critical, otherwise log and continue
            if exit_code == 1:  # Critical error
                raise
            else:
                self.logger.warning(f"[{self.correlation_id}] Continuing after recoverable error in {operation}")
    
    def handle_dependency_error(self, dependency_name: str, error: Exception, 
                              install_hint: Optional[str] = None) -> bool:
        """
        Handle dependency-related errors with consistent messaging.
        
        Args:
            dependency_name: Name of the missing dependency
            error: The import or dependency error
            install_hint: Optional installation instruction
            
        Returns:
            bool: True if should continue with graceful degradation, False if critical
        """
        context = {
            "dependency": dependency_name,
            "install_hint": install_hint or f"pip install {dependency_name}"
        }
        
        pipeline_error = self.pipeline_error_handler.create_error(
            step_name=self.step_name,
            error=error,
            category=ErrorCategory.DEPENDENCY,
            context=context
        )
        
        # Log dependency error with helpful information
        if install_hint:
            log_step_warning(
                self.logger, 
                f"{dependency_name} not available - {install_hint}"
            )
        else:
            log_step_warning(
                self.logger,
                f"{dependency_name} not available - install with: pip install {dependency_name}"
            )
        
        # Dependency errors are usually recoverable with graceful degradation
        return True
    
    def handle_file_operation_error(self, operation: str, file_path: Path, 
                                  error: Exception, critical: bool = False) -> bool:
        """
        Handle file operation errors consistently.
        
        Args:
            operation: Description of file operation (read, write, etc.)
            file_path: Path to the file involved
            error: The file operation error
            critical: Whether this error should halt execution
            
        Returns:
            bool: True if should continue, False if should halt
        """
        context = {
            "file_operation": operation,
            "file_path": str(file_path),
            "file_exists": file_path.exists() if isinstance(file_path, Path) else False
        }
        
        category = ErrorCategory.FILE_ERROR
        if not critical:
            log_step_warning(
                self.logger,
                f"File {operation} failed for {file_path}: {error}"
            )
        else:
            log_step_error(
                self.logger,
                f"Critical file {operation} failed for {file_path}: {error}"
            )
        
        pipeline_error = self.pipeline_error_handler.create_error(
            step_name=self.step_name,
            error=error,
            category=category,
            context=context
        )
        
        exit_code = self.pipeline_error_handler.handle_error(pipeline_error)
        return exit_code != 1  # Continue unless critical
    
    def handle_validation_error(self, validation_type: str, details: str, 
                              error: Optional[Exception] = None) -> bool:
        """
        Handle validation errors consistently.
        
        Args:
            validation_type: Type of validation that failed
            details: Details about the validation failure
            error: Optional underlying exception
            
        Returns:
            bool: True if should continue with warnings, False if should halt
        """
        context = {
            "validation_type": validation_type,
            "validation_details": details
        }
        
        if error:
            pipeline_error = self.pipeline_error_handler.create_error(
                step_name=self.step_name,
                error=error,
                category=ErrorCategory.VALIDATION,
                context=context
            )
        else:
            # Create synthetic error for validation failures without exceptions
            validation_error = ValueError(f"{validation_type} validation failed: {details}")
            pipeline_error = self.pipeline_error_handler.create_error(
                step_name=self.step_name,
                error=validation_error,
                category=ErrorCategory.VALIDATION,
                context=context
            )
        
        log_step_warning(self.logger, f"Validation warning: {validation_type} - {details}")
        exit_code = self.pipeline_error_handler.handle_error(pipeline_error)
        return exit_code != 1
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors handled by this instance.
        
        Returns:
            Dict containing error statistics and details
        """
        errors = self.pipeline_error_handler.errors
        
        return {
            "step_name": self.step_name,
            "correlation_id": self.correlation_id,
            "total_errors": len(errors),
            "critical_errors": sum(1 for e in errors if e.severity == ErrorSeverity.CRITICAL),
            "recoverable_errors": sum(1 for e in errors if e.recoverable),
            "error_categories": {
                category.value: sum(1 for e in errors if e.category == category)
                for category in ErrorCategory
            },
            "errors": [
                {
                    "type": error.error_type,
                    "message": error.message,
                    "severity": error.severity.value,
                    "recoverable": error.recoverable,
                    "category": error.category.value
                }
                for error in errors
            ]
        }
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Automatically categorize an error based on its type and message."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        if error_type in ('ImportError', 'ModuleNotFoundError'):
            return ErrorCategory.DEPENDENCY
        elif error_type in ('FileNotFoundError', 'PermissionError', 'OSError'):
            return ErrorCategory.FILE_ERROR  
        elif error_type in ('ValueError', 'TypeError') and 'validation' in error_message:
            return ErrorCategory.VALIDATION
        elif 'timeout' in error_message or error_type == 'TimeoutError':
            return ErrorCategory.TIMEOUT
        elif 'memory' in error_message or error_type == 'MemoryError':
            return ErrorCategory.RESOURCE
        else:
            return ErrorCategory.UNKNOWN


def create_error_handler(step_name: str, logger: Optional[logging.Logger] = None) -> StandardizedErrorHandler:
    """
    Convenience function to create a standardized error handler.
    
    Args:
        step_name: Name of the pipeline step or module
        logger: Optional logger instance
        
    Returns:
        StandardizedErrorHandler instance
    """
    return StandardizedErrorHandler(step_name, logger)


def with_error_handling(step_name: str, logger: Optional[logging.Logger] = None):
    """
    Decorator to add standardized error handling to functions.
    
    Args:
        step_name: Name of the pipeline step or module
        logger: Optional logger instance
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            error_handler = create_error_handler(step_name, logger)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                with error_handler.error_context(f"executing {func.__name__}"):
                    raise
                    
        return wrapper
    return decorator

