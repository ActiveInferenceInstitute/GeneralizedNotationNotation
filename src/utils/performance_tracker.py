"""
Performance Tracker Utilities for GNN Pipeline

Provides real-time performance tracking, operation timing, and resource usage monitoring.
Exposes PerformanceTracker, a global performance_tracker instance, and a track_operation context manager.
"""

from contextlib import contextmanager
import time
import platform
import os
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Define PerformanceTracker here to avoid circular imports
class PerformanceTracker:
    """Track performance metrics across pipeline steps."""
    
    def __init__(self):
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
    
    def record_timing(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Record timing information for an operation."""
        with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = []
            
            self._metrics[operation].append({
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            })
    
    @contextmanager
    def track_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to automatically track operation timing."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(operation, duration, metadata)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all tracked operations."""
        with self._lock:
            summary = {}
            for operation, measurements in self._metrics.items():
                durations = [m['duration'] for m in measurements]
                summary[operation] = {
                    'count': len(durations),
                    'total_duration': sum(durations),
                    'avg_duration': sum(durations) / len(durations) if durations else 0,
                    'min_duration': min(durations) if durations else 0,
                    'max_duration': max(durations) if durations else 0
                }
            return summary

    def get_timestamp(self) -> str:
        """Get current timestamp as ISO format string."""
        return datetime.now().isoformat()

# Global performance tracker instance
performance_tracker = PerformanceTracker()

_monitoring_data = {}
_monitoring_lock = threading.Lock()


__all__ = [
    'PerformanceTracker',
    'performance_tracker',
    'track_operation',
    'get_performance_metrics',
    'start_performance_monitoring',
    'stop_performance_monitoring',
    'generate_performance_report',
]

@contextmanager
def track_operation(operation: str, metadata: dict = None):
    """
    Context manager to track the duration of an operation using the global performance_tracker.
    Args:
        operation: Name of the operation
        metadata: Optional dictionary of metadata
    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        performance_tracker.record_timing(operation, duration, metadata)


def get_performance_metrics() -> dict:
    """
    Retrieve current performance metrics, including tracked operations and system info.
    Returns:
        dict: { 'operations': ..., 'system_info': ... }
    """
    metrics = getattr(performance_tracker, '_metrics', {})
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'memory_mb': psutil.virtual_memory().total // (1024 * 1024) if hasattr(psutil, 'virtual_memory') else None,
        'pid': os.getpid(),
    }
    return {'operations': metrics, 'system_info': system_info}


def start_performance_monitoring():
    """
    Start performance monitoring (records initial system state).
    """
    with _monitoring_lock:
        _monitoring_data['start'] = get_performance_metrics()


def stop_performance_monitoring() -> dict:
    """
    Stop performance monitoring and return monitoring data (delta from start).
    Returns:
        dict: { 'start': ..., 'end': ..., 'delta': ... }
    """
    with _monitoring_lock:
        end = get_performance_metrics()
        start = _monitoring_data.get('start', {})
        delta = {'operations': {}, 'system_info': {}}
        # Compute delta for operations (by operation name)
        for op, records in end.get('operations', {}).items():
            start_records = start.get('operations', {}).get(op, [])
            delta['operations'][op] = len(records) - len(start_records)
        # Compute delta for system_info (memory, etc.)
        for k in end.get('system_info', {}):
            try:
                delta['system_info'][k] = end['system_info'][k] - start.get('system_info', {}).get(k, 0)
            except Exception:
                delta['system_info'][k] = None
        _monitoring_data['end'] = end
        _monitoring_data['delta'] = delta
        return {'start': start, 'end': end, 'delta': delta}


def generate_performance_report() -> dict:
    """
    Generate a performance report (summary of tracked operations and system info).
    Returns:
        dict: { 'summary': ..., 'details': ... }
    """
    metrics = get_performance_metrics()
    summary = {
        'total_operations': sum(len(v) for v in metrics['operations'].values()),
        'unique_operations': list(metrics['operations'].keys()),
        'system_info': metrics['system_info'],
    }
    return {'summary': summary, 'details': metrics} 