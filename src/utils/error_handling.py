#!/usr/bin/env python3
"""
GNN Pipeline Error Handling Framework

This module provides comprehensive error handling, recovery, and reporting capabilities
for the GNN processing pipeline. It implements standardized error patterns across all
24 pipeline steps with configurable retry strategies and detailed diagnostics.
"""

import sys
import logging
import traceback
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic
from contextlib import contextmanager
from pathlib import Path
import json
import hashlib

# Type variable for generic error handling
T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories for error classification."""
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    RESOURCE = "resource"
    VALIDATION = "validation"
    EXECUTION = "execution"
    SECURITY = "security"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    SKIP = "skip"
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"
    MANUAL = "manual"


@dataclass
class PipelineError:
    """Structured error information for pipeline steps."""
    step_name: str
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    recoverable: bool
    recovery_strategy: RecoveryStrategy
    context: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "step_name": self.step_name,
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "recovery_strategy": self.recovery_strategy.value,
            "context": self.context,
            "traceback": self.traceback,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id
        }


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    retry_on: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.NETWORK, ErrorCategory.TIMEOUT, ErrorCategory.RESOURCE
    ])


class PipelineErrorHandler:
    """Centralized error handling and recovery system."""

    def __init__(self, logger: logging.Logger, correlation_id: str):
        self.logger = logger
        self.correlation_id = correlation_id
        self.errors: List[PipelineError] = []
        self.retry_config = RetryConfig()

    def create_error(self,
                    step_name: str,
                    error: Exception,
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    context: Dict[str, Any] = None) -> PipelineError:
        """Create a standardized pipeline error from an exception."""

        # Classify error severity and recovery strategy
        severity, recovery_strategy = self._classify_error(error, category)

        # Extract error context
        error_context = context or {}
        error_context.update({
            "exception_type": type(error).__name__,
            "module": getattr(error, '__module__', 'unknown'),
        })

        # Create structured error
        pipeline_error = PipelineError(
            step_name=step_name,
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            category=category,
            recoverable=self._is_recoverable(error, category),
            recovery_strategy=recovery_strategy,
            context=error_context,
            traceback=traceback.format_exc(),
            correlation_id=self.correlation_id
        )

        self.errors.append(pipeline_error)
        return pipeline_error

    def _classify_error(self, error: Exception, category: ErrorCategory) -> tuple[ErrorSeverity, RecoveryStrategy]:
        """Classify error severity and determine recovery strategy."""

        error_type = type(error).__name__
        error_message = str(error).lower()

        # Critical errors - always fail fast
        if category in [ErrorCategory.SECURITY, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.CRITICAL, RecoveryStrategy.FAIL_FAST

        # Network and timeout errors - retry
        if category in [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT]:
            return ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY

        # File system errors - may be recoverable
        if category == ErrorCategory.FILE_SYSTEM:
            if "permission" in error_message:
                return ErrorSeverity.HIGH, RecoveryStrategy.MANUAL
            return ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY

        # Resource errors - retry with backoff
        if category == ErrorCategory.RESOURCE:
            return ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY

        # Execution errors - depends on type
        if category == ErrorCategory.EXECUTION:
            if "timeout" in error_message:
                return ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY
            return ErrorSeverity.HIGH, RecoveryStrategy.FAIL_FAST

        # Default classification
        return ErrorSeverity.MEDIUM, RecoveryStrategy.CONTINUE

    def _is_recoverable(self, error: Exception, category: ErrorCategory) -> bool:
        """Determine if an error is recoverable."""
        unrecoverable_categories = [ErrorCategory.SECURITY, ErrorCategory.CONFIGURATION]
        return category not in unrecoverable_categories

    def handle_error(self, error: PipelineError) -> int:
        """Handle an error according to its recovery strategy and return exit code."""

        # Log the error with appropriate level
        log_method = {
            ErrorSeverity.LOW: self.logger.debug,
            ErrorSeverity.MEDIUM: self.logger.warning,
            ErrorSeverity.HIGH: self.logger.error,
            ErrorSeverity.CRITICAL: self.logger.critical
        }[error.severity]

        log_method(f"[{error.correlation_id}] {error.category.value}: {error.message}")

        # Add context information to log
        if error.context:
            self.logger.debug(f"[{error.correlation_id}] Error context: {error.context}")

        # Handle according to recovery strategy
        if error.recovery_strategy == RecoveryStrategy.FAIL_FAST:
            self.logger.error(f"[{error.correlation_id}] Critical error - failing fast")
            return 1  # Critical error

        elif error.recovery_strategy == RecoveryStrategy.RETRY:
            self.logger.warning(f"[{error.correlation_id}] Error is retryable but retry logic should be handled by caller")
            return 2  # Warning - recoverable

        elif error.recovery_strategy == RecoveryStrategy.SKIP:
            self.logger.warning(f"[{error.correlation_id}] Skipping step due to error")
            return 2  # Warning - skipped

        elif error.recovery_strategy == RecoveryStrategy.CONTINUE:
            self.logger.warning(f"[{error.correlation_id}] Continuing despite error")
            return 2  # Warning - continued with issues

        else:  # MANUAL
            self.logger.error(f"[{error.correlation_id}] Manual intervention required")
            return 1  # Critical - manual intervention needed

    @contextmanager
    def error_context(self, step_name: str, operation: str):
        """Context manager for error handling in pipeline operations."""
        try:
            self.logger.debug(f"[{self.correlation_id}] Starting {operation} in {step_name}")
            yield
            self.logger.debug(f"[{self.correlation_id}] Completed {operation} in {step_name}")
        except Exception as e:
            error = self.create_error(step_name, e, context={"operation": operation})
            exit_code = self.handle_error(error)

            # Re-raise critical errors
            if exit_code == 1:
                raise e
            # For warnings/errors that should be handled by caller
            elif exit_code == 2:
                pass  # Allow caller to decide how to proceed

    def retry_operation(self, operation: Callable[[], T], error: PipelineError) -> Optional[T]:
        """Retry an operation according to retry configuration."""

        if error.category not in self.retry_config.retry_on:
            self.logger.debug(f"[{self.correlation_id}] Error category {error.category.value} not retryable")
            return None

        for attempt in range(self.retry_config.max_attempts):
            try:
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.backoff_factor ** attempt),
                    self.retry_config.max_delay
                )

                if attempt > 0:
                    self.logger.info(f"[{self.correlation_id}] Retrying operation (attempt {attempt + 1}/{self.retry_config.max_attempts}) after {delay:.1f}s")
                    time.sleep(delay)

                result = operation()
                if attempt > 0:
                    self.logger.info(f"[{self.correlation_id}] Operation succeeded on retry attempt {attempt + 1}")

                return result

            except Exception as e:
                self.logger.warning(f"[{self.correlation_id}] Retry attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_config.max_attempts - 1:
                    self.logger.error(f"[{self.correlation_id}] All retry attempts exhausted")

        return None

    def generate_error_report(self, output_dir: Path) -> Path:
        """Generate a comprehensive error report."""
        report_file = output_dir / f"pipeline_errors_{int(time.time())}.json"

        error_summary = {
            "correlation_id": self.correlation_id,
            "total_errors": len(self.errors),
            "error_breakdown": {},
            "severity_breakdown": {},
            "category_breakdown": {},
            "errors": [error.to_dict() for error in self.errors],
            "generated_at": time.time()
        }

        # Generate breakdowns
        for error in self.errors:
            error_summary["error_breakdown"][error.error_type] = error_summary["error_breakdown"].get(error.error_type, 0) + 1
            error_summary["severity_breakdown"][error.severity.value] = error_summary["severity_breakdown"].get(error.severity.value, 0) + 1
            error_summary["category_breakdown"][error.category.value] = error_summary["category_breakdown"].get(error.category.value, 0) + 1

        with open(report_file, 'w') as f:
            json.dump(error_summary, f, indent=2)

        self.logger.info(f"[{self.correlation_id}] Error report saved to {report_file}")
        return report_file


class ExitCode:
    """Standardized exit codes for pipeline steps."""
    SUCCESS = 0
    CRITICAL_ERROR = 1
    SUCCESS_WITH_WARNINGS = 2

    @staticmethod
    def from_error_severity(severity: ErrorSeverity) -> int:
        """Convert error severity to exit code."""
        return {
            ErrorSeverity.LOW: ExitCode.SUCCESS_WITH_WARNINGS,
            ErrorSeverity.MEDIUM: ExitCode.SUCCESS_WITH_WARNINGS,
            ErrorSeverity.HIGH: ExitCode.CRITICAL_ERROR,
            ErrorSeverity.CRITICAL: ExitCode.CRITICAL_ERROR
        }.get(severity, ExitCode.CRITICAL_ERROR)


def generate_correlation_id(prefix: str = "gnn") -> str:
    """Generate a unique correlation ID for error tracking."""
    timestamp = str(int(time.time()))
    random_suffix = hashlib.md5(f"{prefix}{timestamp}".encode()).hexdigest()[:6]
    return f"{prefix}_{timestamp}_{random_suffix}"


# Convenience functions for common error patterns
def handle_file_system_error(logger: logging.Logger, error: Exception, file_path: Path, operation: str) -> int:
    """Handle file system related errors."""
    handler = PipelineErrorHandler(logger, generate_correlation_id())
    pipeline_error = handler.create_error(
        "file_system",
        error,
        ErrorCategory.FILE_SYSTEM,
        context={"file_path": str(file_path), "operation": operation}
    )
    return handler.handle_error(pipeline_error)


def handle_network_error(logger: logging.Logger, error: Exception, url: str = None) -> int:
    """Handle network related errors."""
    handler = PipelineErrorHandler(logger, generate_correlation_id())
    pipeline_error = handler.create_error(
        "network",
        error,
        ErrorCategory.NETWORK,
        context={"url": url}
    )
    return handler.handle_error(pipeline_error)


def handle_timeout_error(logger: logging.Logger, error: Exception, timeout_seconds: int) -> int:
    """Handle timeout related errors."""
    handler = PipelineErrorHandler(logger, generate_correlation_id())
    pipeline_error = handler.create_error(
        "timeout",
        error,
        ErrorCategory.TIMEOUT,
        context={"timeout_seconds": timeout_seconds}
    )
    return handler.handle_error(pipeline_error)

