#!/usr/bin/env python3
"""
Error Message and Recovery Enhancement Module
==============================================

Provides structured error handling, informative error messages, and recovery
mechanisms for all pipeline operations.
"""

from typing import Optional, Dict, Any, Callable, List
from enum import Enum
import logging
import traceback
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"         # Informational message
    WARNING = "warning"   # Warning, operation can continue
    ERROR = "error"       # Error, operation failed
    CRITICAL = "critical" # Critical error, pipeline halt recommended


@dataclass
class ErrorContext:
    """Structured error context for detailed reporting."""
    operation: str
    severity: ErrorSeverity
    message: str
    error_code: str
    details: Dict[str, Any] = None
    recovery_suggestions: List[str] = None
    original_exception: Optional[Exception] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'operation': self.operation,
            'severity': self.severity.value,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details or {},
            'recovery_suggestions': self.recovery_suggestions or [],
        }


class ErrorRecoveryManager:
    """Manages error handling and recovery strategies."""
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize error recovery manager."""
        self.logger = logger or logging.getLogger(__name__)
        self.error_handlers: Dict[str, Callable] = {}
        self.recovery_strategies: Dict[str, List[str]] = {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default error handlers and recovery strategies."""
        
        # Import error handlers
        self.error_handlers['import'] = self._handle_import_error
        self.recovery_strategies['import'] = [
            'Install missing package: pip install <package>',
            'Check Python version compatibility',
            'Verify package is in requirements.txt',
            'Try running with --verbose for more details',
        ]
        
        # File operation error handlers
        self.error_handlers['file'] = self._handle_file_error
        self.recovery_strategies['file'] = [
            'Verify file path exists and is accessible',
            'Check file permissions (read/write access)',
            'Ensure sufficient disk space is available',
            'Try with a different file path',
        ]
        
        # Resource error handlers
        self.error_handlers['resource'] = self._handle_resource_error
        self.recovery_strategies['resource'] = [
            'Check system memory and disk space',
            'Reduce model size or complexity',
            'Close other applications to free resources',
            'Try with --lightweight-mode for reduced memory usage',
        ]
        
        # Type validation error handlers
        self.error_handlers['validation'] = self._handle_validation_error
        self.recovery_strategies['validation'] = [
            'Review error details for specific field',
            'Check type annotation requirements',
            'Validate input data format',
            'Refer to GNN schema documentation',
        ]
        
        # Execution error handlers
        self.error_handlers['execution'] = self._handle_execution_error
        self.recovery_strategies['execution'] = [
            'Check simulation parameters',
            'Verify model structure is valid',
            'Review execution log for details',
            'Try with smaller time horizon or state space',
        ]
    
    def handle_error(self, context: ErrorContext) -> bool:
        """Handle error with appropriate recovery strategy."""
        
        # Log error with severity
        log_func = getattr(self.logger, context.severity.value, self.logger.error)
        log_func(f"[{context.error_code}] {context.message}")
        
        # Log details if available
        if context.details:
            self.logger.debug(f"Error details: {context.details}")
        
        # Log recovery suggestions
        if context.recovery_suggestions:
            self.logger.info("Recovery suggestions:")
            for i, suggestion in enumerate(context.recovery_suggestions, 1):
                self.logger.info(f"  {i}. {suggestion}")
        
        # Log original exception traceback if available
        if context.original_exception:
            self.logger.debug(f"Exception: {traceback.format_exc()}")
        
        return context.severity != ErrorSeverity.CRITICAL
    
    def _handle_import_error(self, error_code: str, message: str) -> ErrorContext:
        """Handle import errors with recovery suggestions."""
        return ErrorContext(
            operation="Module Import",
            severity=ErrorSeverity.WARNING,
            message=message,
            error_code=error_code,
            recovery_suggestions=self.recovery_strategies.get('import', [])
        )
    
    def _handle_file_error(self, error_code: str, message: str) -> ErrorContext:
        """Handle file operation errors with recovery suggestions."""
        return ErrorContext(
            operation="File Operation",
            severity=ErrorSeverity.ERROR,
            message=message,
            error_code=error_code,
            recovery_suggestions=self.recovery_strategies.get('file', [])
        )
    
    def _handle_resource_error(self, error_code: str, message: str) -> ErrorContext:
        """Handle resource errors with recovery suggestions."""
        return ErrorContext(
            operation="Resource Management",
            severity=ErrorSeverity.ERROR,
            message=message,
            error_code=error_code,
            recovery_suggestions=self.recovery_strategies.get('resource', [])
        )
    
    def _handle_validation_error(self, error_code: str, message: str) -> ErrorContext:
        """Handle validation errors with recovery suggestions."""
        return ErrorContext(
            operation="Validation",
            severity=ErrorSeverity.ERROR,
            message=message,
            error_code=error_code,
            recovery_suggestions=self.recovery_strategies.get('validation', [])
        )
    
    def _handle_execution_error(self, error_code: str, message: str) -> ErrorContext:
        """Handle execution errors with recovery suggestions."""
        return ErrorContext(
            operation="Execution",
            severity=ErrorSeverity.ERROR,
            message=message,
            error_code=error_code,
            recovery_suggestions=self.recovery_strategies.get('execution', [])
        )


class ErrorCodeRegistry:
    """Registry of standardized error codes."""
    
    # Import errors
    IMPORT_NOT_FOUND = "E001"
    IMPORT_VERSION_MISMATCH = "E002"
    IMPORT_INCOMPATIBLE = "E003"
    
    # File errors
    FILE_NOT_FOUND = "E101"
    FILE_PERMISSION_DENIED = "E102"
    FILE_CORRUPTED = "E103"
    FILE_FORMAT_INVALID = "E104"
    
    # Resource errors
    RESOURCE_MEMORY_EXCEEDED = "E201"
    RESOURCE_DISK_SPACE_EXCEEDED = "E202"
    RESOURCE_TIMEOUT = "E203"
    
    # Validation errors
    VALIDATION_TYPE_MISMATCH = "E301"
    VALIDATION_CONSTRAINT_VIOLATION = "E302"
    VALIDATION_DATA_MISSING = "E303"
    VALIDATION_RANGE_EXCEEDED = "E304"
    
    # Execution errors
    EXECUTION_FAILED = "E401"
    EXECUTION_TIMEOUT = "E402"
    EXECUTION_UNSUPPORTED = "E403"
    
    @classmethod
    def get_all_codes(cls) -> Dict[str, str]:
        """Get all registered error codes."""
        return {
            attr: getattr(cls, attr) 
            for attr in dir(cls) 
            if attr.isupper() and not attr.startswith('_')
        }


def format_error_message(
    error_code: str,
    operation: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[str]] = None
) -> str:
    """Format error message with all relevant information."""
    
    lines = [
        f"[{error_code}] {operation} Error",
        f"Message: {message}",
    ]
    
    if details:
        lines.append("Details:")
        for key, value in details.items():
            lines.append(f"  {key}: {value}")
    
    if suggestions:
        lines.append("Recovery suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            lines.append(f"  {i}. {suggestion}")
    
    return "\n".join(lines)


# Global error recovery manager instance
_recovery_manager = ErrorRecoveryManager()


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager."""
    return _recovery_manager


def format_and_log_error(
    error_code: str,
    operation: str,
    message: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    details: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[str]] = None,
    exception: Optional[Exception] = None
) -> ErrorContext:
    """Format, log, and return error context."""
    
    context = ErrorContext(
        operation=operation,
        severity=severity,
        message=message,
        error_code=error_code,
        details=details,
        recovery_suggestions=suggestions,
        original_exception=exception
    )
    
    _recovery_manager.handle_error(context)
    return context


class ErrorReporter:
    """Collects and reports errors during pipeline execution."""
    
    def __init__(self):
        """Initialize error reporter."""
        self.errors: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def collect_error(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None, severity: str = "error"):
        """
        Collect an error for reporting.
        
        Args:
            error_type: Type/category of the error
            message: Error message
            details: Additional error details
            severity: Error severity level
        """
        error_record = {
            "type": error_type,
            "message": message,
            "details": details or {},
            "severity": severity,
            "timestamp": str(logging.LogRecord(
                name="error_reporter",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="",
                args=(),
                exc_info=None
            ).created)
        }
        self.errors.append(error_record)
        self.logger.debug(f"Error collected: {error_type} - {message}")
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all collected errors."""
        return self.errors.copy()
    
    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0
    
    def clear_errors(self):
        """Clear all collected errors."""
        self.errors.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected errors."""
        severity_counts = {}
        error_type_counts = {}
        
        for error in self.errors:
            severity = error.get("severity", "unknown")
            error_type = error.get("type", "unknown")
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.errors),
            "by_severity": severity_counts,
            "by_type": error_type_counts
        }
