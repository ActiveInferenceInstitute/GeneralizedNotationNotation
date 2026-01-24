"""
Resource Monitoring for Test Execution.

This module provides system resource monitoring during test execution.
"""

import logging
import time
import threading
from typing import Dict

# psutil is optional; tests should not fail to import if it's missing
try:
    import psutil as _psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except Exception:
    _psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False


class ResourceMonitor:
    """Monitor system resources during test execution."""
    
    def __init__(self, memory_limit_mb: int = 2048, cpu_limit_percent: int = 80):
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_percent = cpu_limit_percent
        self.peak_memory = 0.0
        self.peak_cpu = 0.0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if not PSUTIL_AVAILABLE:
            # No-op if psutil is not available
            self.monitoring = False
            return
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_resources(self):
        """Monitor system resources in background."""
        if not PSUTIL_AVAILABLE:
            # Passive sleep loop to avoid busy spin when psutil missing
            while self.monitoring:
                try:
                    time.sleep(0.5)
                except Exception:
                    break
            return
        process = _psutil.Process()
        
        while self.monitoring:
            try:
                # Memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, memory_mb)
                
                # CPU usage
                cpu_percent = process.cpu_percent()
                self.peak_cpu = max(self.peak_cpu, cpu_percent)
                
                # Check limits
                if memory_mb > self.memory_limit_mb:
                    logging.warning(f"⚠️ Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.memory_limit_mb}MB)")
                
                if cpu_percent > self.cpu_limit_percent:
                    logging.warning(f"⚠️ CPU usage ({cpu_percent:.1f}%) exceeds limit ({self.cpu_limit_percent}%)")
                    
                time.sleep(0.5)  # Monitor every 500ms
                
            except Exception as e:
                logging.warning(f"Resource monitoring error: {e}")
                break
                
    def get_stats(self) -> Dict[str, float]:
        """Get current resource statistics."""
        if not PSUTIL_AVAILABLE:
            return {
                "peak_memory_mb": self.peak_memory,
                "peak_cpu_percent": self.peak_cpu,
                "current_memory_mb": 0.0,
                "current_cpu_percent": 0.0,
            }
        return {
            "peak_memory_mb": self.peak_memory,
            "peak_cpu_percent": self.peak_cpu,
            "current_memory_mb": _psutil.Process().memory_info().rss / 1024 / 1024,
            "current_cpu_percent": _psutil.Process().cpu_percent(),
        }
