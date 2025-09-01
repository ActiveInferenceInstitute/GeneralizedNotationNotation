#!/usr/bin/env python3
"""
Enhanced Logging Configuration for GNN Pipeline

This module provides enhanced logging capabilities with correlation IDs,
performance tracking, and comprehensive diagnostic information.
"""

import logging
import sys
import uuid
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from datetime import datetime
import json

from .logging_utils import setup_step_logging


class CorrelationContext:
    """Thread-local correlation context for tracking requests across pipeline steps."""
    
    _local = threading.local()
    
    @classmethod
    def get_correlation_id(cls) -> str:
        """Get current correlation ID or create a new one."""
        if not hasattr(cls._local, 'correlation_id'):
            cls._local.correlation_id = str(uuid.uuid4())[:8]
        return cls._local.correlation_id
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str):
        """Set correlation ID for current thread."""
        cls._local.correlation_id = correlation_id
    
    @classmethod
    def new_correlation_id(cls) -> str:
        """Generate and set a new correlation ID."""
        correlation_id = str(uuid.uuid4())[:8]
        cls.set_correlation_id(correlation_id)
        return correlation_id


class PerformanceTracker:
    """Track performance metrics for pipeline operations."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.active_operations: Dict[str, float] = {}
    
    def start_operation(self, operation_name: str) -> str:
        """Start tracking an operation."""
        correlation_id = CorrelationContext.get_correlation_id()
        key = f"{operation_name}#{correlation_id}"
        self.active_operations[key] = time.time()
        return key
    
    def end_operation(self, operation_key: str) -> float:
        """End tracking an operation and return duration."""
        if operation_key in self.active_operations:
            duration = time.time() - self.active_operations[operation_key]
            del self.active_operations[operation_key]
            
            # Extract operation name
            operation_name = operation_key.split('#')[0]
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            self.metrics[operation_name].append(duration)
            
            return duration
        return 0.0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics."""
        summary = {}
        for operation, durations in self.metrics.items():
            if durations:
                summary[operation] = {
                    "count": len(durations),
                    "total_time": sum(durations),
                    "average_time": sum(durations) / len(durations),
                    "min_time": min(durations),
                    "max_time": max(durations)
                }
        return summary


class DiagnosticLogger:
    """Enhanced logger with diagnostic capabilities and correlation tracking."""
    
    def __init__(self, step_name: str, output_dir: Optional[Path] = None):
        """
        Initialize diagnostic logger.
        
        Args:
            step_name: Name of the pipeline step
            output_dir: Optional directory for detailed logs
        """
        self.step_name = step_name
        self.output_dir = output_dir
        self.logger = setup_step_logging(step_name, verbose=True)
        self.performance_tracker = PerformanceTracker()
        self.start_time = time.time()
        self.diagnostics: Dict[str, Any] = {}
        
        # Set up correlation ID
        self.correlation_id = CorrelationContext.new_correlation_id()
        
        # Get system information
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system diagnostic information."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "timestamp": datetime.now().isoformat(),
                "correlation_id": self.correlation_id,
                "pid": psutil.Process().pid,
                "memory_total_gb": round(memory.total / (1024**3), 1),
                "memory_available_gb": round(memory.available / (1024**3), 1),
                "memory_percent": memory.percent,
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "disk_total_gb": round(disk.total / (1024**3), 1),
                "disk_free_gb": round(disk.free / (1024**3), 1),
                "disk_percent": round((disk.used / disk.total) * 100, 1)
            }
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "correlation_id": self.correlation_id,
                "error": f"Could not get system info: {e}"
            }
    
    def log_step_start(self, description: str, **context):
        """Log step start with enhanced diagnostics."""
        self.diagnostics["step_start"] = {
            "description": description,
            "system_info": self.system_info,
            "context": context
        }
        
        message = f"[{self.correlation_id}] 🚀 Starting {self.step_name}: {description}"
        if context:
            message += f" | Context: {context}"
        
        self.logger.info(message)
        
        # Log system resource status
        self.logger.debug(
            f"[{self.correlation_id}] System resources: "
            f"Memory {self.system_info.get('memory_percent', 'unknown')}%, "
            f"CPU {self.system_info.get('cpu_percent', 'unknown')}%, "
            f"Disk {self.system_info.get('disk_percent', 'unknown')}%"
        )
    
    def log_step_success(self, message: str, **context):
        """Log step success with performance metrics."""
        duration = time.time() - self.start_time
        
        self.diagnostics["step_success"] = {
            "message": message,
            "duration_seconds": duration,
            "performance_metrics": self.performance_tracker.get_metrics_summary(),
            "context": context
        }
        
        log_message = f"[{self.correlation_id}] ✅ {self.step_name} completed: {message} ({duration:.2f}s)"
        if context:
            log_message += f" | Context: {context}"
        
        self.logger.info(log_message)
    
    def log_step_warning(self, message: str, **context):
        """Log step warning with context."""
        self.diagnostics.setdefault("warnings", []).append({
            "message": message,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
        
        log_message = f"[{self.correlation_id}] ⚠️  {message}"
        if context:
            log_message += f" | Context: {context}"
        
        self.logger.warning(log_message)
    
    def log_step_error(self, message: str, error: Optional[Exception] = None, **context):
        """Log step error with full diagnostic information."""
        error_info = {
            "message": message,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        if error:
            error_info.update({
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "exception_module": getattr(error, '__module__', 'unknown')
            })
        
        self.diagnostics.setdefault("errors", []).append(error_info)
        
        log_message = f"[{self.correlation_id}] ❌ {message}"
        if error:
            log_message += f" | Error: {type(error).__name__}: {error}"
        if context:
            log_message += f" | Context: {context}"
        
        self.logger.error(log_message)
    
    @contextmanager
    def performance_context(self, operation_name: str):
        """Context manager for performance tracking."""
        operation_key = self.performance_tracker.start_operation(operation_name)
        self.logger.debug(f"[{self.correlation_id}] Starting operation: {operation_name}")
        
        try:
            yield
        finally:
            duration = self.performance_tracker.end_operation(operation_key)
            self.logger.debug(
                f"[{self.correlation_id}] Completed operation: {operation_name} ({duration:.3f}s)"
            )
    
    def log_dependency_status(self, dependencies: Dict[str, bool]):
        """Log dependency availability status."""
        available = [name for name, status in dependencies.items() if status]
        unavailable = [name for name, status in dependencies.items() if not status]
        
        self.diagnostics["dependencies"] = {
            "available": available,
            "unavailable": unavailable,
            "total_dependencies": len(dependencies)
        }
        
        self.logger.info(
            f"[{self.correlation_id}] Dependencies: {len(available)}/{len(dependencies)} available"
        )
        
        if unavailable:
            self.logger.warning(
                f"[{self.correlation_id}] Unavailable dependencies: {', '.join(unavailable)}"
            )
    
    def log_file_operations(self, operations: List[Dict[str, Any]]):
        """Log file operation results."""
        successful = [op for op in operations if op.get('success', False)]
        failed = [op for op in operations if not op.get('success', False)]
        
        self.diagnostics["file_operations"] = {
            "successful": len(successful),
            "failed": len(failed),
            "total": len(operations),
            "details": operations
        }
        
        self.logger.info(
            f"[{self.correlation_id}] File operations: {len(successful)}/{len(operations)} successful"
        )
        
        if failed:
            for op in failed:
                self.logger.warning(
                    f"[{self.correlation_id}] File operation failed: {op.get('operation', 'unknown')} "
                    f"on {op.get('file_path', 'unknown')} - {op.get('error', 'unknown error')}"
                )
    
    def save_diagnostic_report(self, output_path: Optional[Path] = None):
        """Save comprehensive diagnostic report."""
        if output_path is None and self.output_dir:
            output_path = self.output_dir / f"{self.step_name}_diagnostic_report.json"
        elif output_path is None:
            return  # No output path specified
        
        # Add final system state
        self.diagnostics["final_system_info"] = self._get_system_info()
        self.diagnostics["total_duration_seconds"] = time.time() - self.start_time
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(self.diagnostics, f, indent=2, default=str)
            
            self.logger.info(f"[{self.correlation_id}] Diagnostic report saved to {output_path}")
        except Exception as e:
            self.logger.error(f"[{self.correlation_id}] Failed to save diagnostic report: {e}")
    
    def get_correlation_id(self) -> str:
        """Get current correlation ID."""
        return self.correlation_id


def create_diagnostic_logger(step_name: str, output_dir: Optional[Path] = None) -> DiagnosticLogger:
    """
    Create a diagnostic logger for a pipeline step.
    
    Args:
        step_name: Name of the pipeline step
        output_dir: Optional directory for diagnostic outputs
        
    Returns:
        DiagnosticLogger instance
    """
    return DiagnosticLogger(step_name, output_dir)


def with_diagnostic_logging(step_name: str, output_dir: Optional[Path] = None):
    """
    Decorator to add diagnostic logging to functions.
    
    Args:
        step_name: Name of the pipeline step
        output_dir: Optional directory for diagnostic outputs
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            diagnostic_logger = create_diagnostic_logger(step_name, output_dir)
            
            try:
                diagnostic_logger.log_step_start(f"executing {func.__name__}")
                
                with diagnostic_logger.performance_context(func.__name__):
                    result = func(*args, **kwargs)
                
                diagnostic_logger.log_step_success(f"{func.__name__} completed successfully")
                return result
                
            except Exception as e:
                diagnostic_logger.log_step_error(f"Error in {func.__name__}", error=e)
                raise
            finally:
                diagnostic_logger.save_diagnostic_report()
                
        return wrapper
    return decorator
