#!/usr/bin/env python3
"""
Resource Management Utilities

This module provides utilities for tracking and managing system resources
during pipeline execution, including memory usage, disk space, and timing.
"""

import os
import time
import psutil
import logging
import functools
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable, TypeVar, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Type variable for generic function decorators
T = TypeVar('T')

class ResourceTracker:
    """Tracks resource usage during operations."""
    
    def __init__(self):
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.current_memory = self.start_memory
        
    def update(self):
        """Update current resource measurements."""
        self.current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, self.current_memory)
        
    def stop(self):
        """Stop tracking and calculate final metrics."""
        self.end_time = time.time()
        self.update()
        
    @property
    def duration(self) -> float:
        """Get operation duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
        
    @property
    def memory_used(self) -> float:
        """Get current memory usage in MB."""
        return self.current_memory - self.start_memory
        
    @property
    def max_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_memory
        
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "duration_seconds": self.duration,
            "memory_used_mb": self.memory_used,
            "peak_memory_mb": self.peak_memory
        }

@contextmanager
def performance_tracker() -> ResourceTracker:
    """Context manager for tracking performance metrics."""
    tracker = ResourceTracker()
    try:
        yield tracker
    finally:
        tracker.stop()

def track_peak_memory(func: Callable[..., T]) -> Callable[..., Tuple[T, float]]:
    """Decorator to track peak memory usage of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[T, float]:
        tracker = ResourceTracker()
        try:
            result = func(*args, **kwargs)
            return result, tracker.peak_memory
        finally:
            tracker.stop()
    return wrapper

@contextmanager
def with_resource_limits(
    max_memory_mb: Optional[float] = None,
    max_time_seconds: Optional[float] = None
):
    """Context manager to enforce resource limits."""
    start_time = time.time()
    process = psutil.Process()
    
    def check_limits():
        if max_memory_mb:
            current_memory = process.memory_info().rss / 1024 / 1024
            if current_memory > max_memory_mb:
                raise RuntimeError(f"Memory limit exceeded: {current_memory:.1f}MB > {max_memory_mb}MB")
                
        if max_time_seconds:
            elapsed = time.time() - start_time
            if elapsed > max_time_seconds:
                raise RuntimeError(f"Time limit exceeded: {elapsed:.1f}s > {max_time_seconds}s")
    
    try:
        yield
    finally:
        check_limits()

def check_disk_space(
    path: Path,
    required_mb: float,
    buffer_factor: float = 1.1
) -> bool:
    """
    Check if sufficient disk space is available.
    
    Args:
        path: Path to check
        required_mb: Required space in MB
        buffer_factor: Safety factor (e.g., 1.1 = 10% extra)
        
    Returns:
        True if sufficient space available
        
    Raises:
        RuntimeError if insufficient space
    """
    import shutil
    
    # Get disk usage
    total, used, free = shutil.disk_usage(path)
    free_mb = free / (1024 * 1024)  # Convert to MB
    
    # Check with buffer
    required_with_buffer = required_mb * buffer_factor
    
    if free_mb < required_with_buffer:
        raise RuntimeError(
            f"Insufficient disk space: {free_mb:.1f}MB free, "
            f"need {required_with_buffer:.1f}MB ({required_mb:.1f}MB + {buffer_factor*100-100:.0f}% buffer)"
        )
        
    return True

def estimate_resources(model_file: Path) -> Dict[str, float]:
    """
    Estimate resource requirements for processing a model.
    
    Args:
        model_file: Path to GNN model file
        
    Returns:
        Dictionary with estimated resources:
        - time: Estimated processing time in seconds
        - memory_mb: Estimated peak memory usage in MB
        - disk_mb: Estimated disk space needed in MB
    """
    # Get file size
    file_size_mb = model_file.stat().st_size / (1024 * 1024)
    
    # Count model complexity
    content = model_file.read_text()
    num_states = content.count("StateSpaceBlock") + content.count("[")
    num_connections = content.count("->")
    
    # Basic estimation heuristics
    time_estimate = 0.1 + (num_states * 0.01) + (num_connections * 0.005)
    memory_estimate = 50 + (num_states * 2) + (num_connections * 1)
    disk_estimate = file_size_mb * 10  # Output files typically 10x input
    
    return {
        "time": time_estimate,
        "memory_mb": memory_estimate,
        "disk_mb": disk_estimate
    }

def log_resource_usage(logger: logging.Logger, tracker: ResourceTracker):
    """Log resource usage metrics."""
    metrics = tracker.to_dict()
    logger.info(
        "Resource usage - Time: %.2fs, Memory: %.1fMB (Peak: %.1fMB)",
        metrics["duration_seconds"],
        metrics["memory_used_mb"],
        metrics["peak_memory_mb"]
    )

def get_system_info() -> Dict[str, Any]:
    """Get system information and resource availability."""
    import platform
    
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": cpu_count,
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_total_gb": memory.total / (1024**3),
        "memory_available_gb": memory.available / (1024**3),
        "memory_percent": memory.percent,
        "disk_usage": {
            str(path): psutil.disk_usage(str(path)).percent
            for path in [Path.home(), Path.cwd()]
        }
    }

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Track performance
    with performance_tracker() as tracker:
        time.sleep(1)  # Simulate work
        tracker.update()
        log_resource_usage(logger, tracker)
        
    # Check resources
    system_info = get_system_info()
    logger.info("System information: %s", system_info) 