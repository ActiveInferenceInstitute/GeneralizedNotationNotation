#!/usr/bin/env python3
"""
Pipeline Monitoring System for Long-Running Steps

This module provides monitoring and progress tracking for pipeline steps that may
take a long time to complete, with timeout handling and progress reporting.
"""

import threading
import time
import signal
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PipelineStepMonitor:
    """Monitor individual pipeline steps with timeout and progress tracking."""

    def __init__(self):
        self.active_steps: Dict[str, Dict[str, Any]] = {}
        self.step_timeouts: Dict[str, float] = {}
        self.progress_callbacks: Dict[str, Callable] = {}

    def start_step(self, step_name: str, timeout_seconds: int = 360,
                   progress_callback: Optional[Callable] = None) -> str:
        """Start monitoring a pipeline step."""
        step_id = f"{step_name}_{int(time.time())}"

        self.active_steps[step_id] = {
            'step_name': step_name,
            'start_time': time.time(),
            'timeout': timeout_seconds,
            'last_progress': time.time(),
            'status': 'running'
        }

        self.step_timeouts[step_id] = timeout_seconds

        if progress_callback:
            self.progress_callbacks[step_id] = progress_callback

        logger.info(f"📊 Started monitoring {step_name} with {timeout_seconds}s timeout")

        # Set up timeout signal
        def timeout_handler(signum, frame):
            logger.warning(f"⏰ Step {step_name} timed out after {timeout_seconds} seconds")
            self._handle_timeout(step_id)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        return step_id

    def update_progress(self, step_id: str, progress_info: Dict[str, Any]):
        """Update progress for a monitored step."""
        if step_id in self.active_steps:
            self.active_steps[step_id].update({
                'last_progress': time.time(),
                'progress_info': progress_info,
                'status': 'progress_updated'
            })

            logger.debug(f"📈 Updated progress for {self.active_steps[step_id]['step_name']}")

    def complete_step(self, step_id: str, success: bool = True):
        """Mark a step as completed."""
        if step_id in self.active_steps:
            step_info = self.active_steps[step_id]
            duration = time.time() - step_info['start_time']

            if success:
                logger.info(f"✅ Completed {step_info['step_name']} in {duration:.1f}s")
            else:
                logger.warning(f"❌ Failed {step_info['step_name']} after {duration:.1f}s")

            # Cancel timeout
            signal.alarm(0)

            # Remove from active steps
            del self.active_steps[step_id]
            if step_id in self.step_timeouts:
                del self.step_timeouts[step_id]
            if step_id in self.progress_callbacks:
                del self.progress_callbacks[step_id]

    def _handle_timeout(self, step_id: str):
        """Handle step timeout."""
        if step_id in self.active_steps:
            step_info = self.active_steps[step_id]
            step_name = step_info['step_name']

            logger.warning(f"⏰ Step {step_name} timed out after {step_info['timeout']} seconds")
            step_info['status'] = 'timed_out'

            # Try to call progress callback if available
            if step_id in self.progress_callbacks:
                try:
                    self.progress_callbacks[step_id]({
                        'status': 'timeout',
                        'message': f'Step timed out after {step_info["timeout"]} seconds'
                    })
                except Exception as e:
                    logger.error(f"Error calling progress callback: {e}")

    def get_step_status(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a monitored step."""
        return self.active_steps.get(step_id)

    def get_all_active_steps(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently active steps."""
        return self.active_steps.copy()

# Global monitor instance
_monitor = PipelineStepMonitor()

@contextmanager
def monitor_step(step_name: str, timeout_seconds: int = 360,
                progress_callback: Optional[Callable] = None):
    """Context manager for monitoring pipeline steps."""
    step_id = _monitor.start_step(step_name, timeout_seconds, progress_callback)

    try:
        yield step_id
    except Exception as e:
        _monitor.complete_step(step_id, success=False)
        raise
    else:
        _monitor.complete_step(step_id, success=True)

def get_monitor_status() -> Dict[str, Any]:
    """Get current monitoring status."""
    return {
        'active_steps': len(_monitor.active_steps),
        'steps': _monitor.get_all_active_steps()
    }

def update_step_progress(step_id: str, progress_info: Dict[str, Any]):
    """Update progress for a step."""
    _monitor.update_progress(step_id, progress_info)

def log_monitor_status():
    """Log current monitoring status."""
    status = get_monitor_status()

    if status['active_steps'] > 0:
        logger.info(f"📊 Pipeline Monitor Status: {status['active_steps']} active steps")
        for step_id, step_info in status['steps'].items():
            duration = time.time() - step_info['start_time']
            logger.info(f"  - {step_info['step_name']}: {step_info['status']} ({duration:.1f}s)")
    else:
        logger.debug("📊 Pipeline Monitor Status: No active steps")


def generate_pipeline_health_report() -> Dict[str, Any]:
    """
    Generate a comprehensive health report for the pipeline.

    Returns:
        Dictionary containing pipeline health information including:
        - Active steps and their status
        - System resource usage
        - Performance metrics
        - Warning and error counts
    """
    try:
        import psutil
        import os

        # Get system information
        system_info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory()._asdict(),
            'disk_usage': psutil.disk_usage('/')._asdict(),
            'network': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
        }

        # Get pipeline monitoring status
        monitor_status = get_monitor_status()

        # Get process information
        current_process = psutil.Process(os.getpid())
        process_info = {
            'pid': current_process.pid,
            'cpu_percent': current_process.cpu_percent(),
            'memory_info': current_process.memory_info()._asdict(),
            'threads': current_process.num_threads(),
            'create_time': current_process.create_time(),
        }

        # Compile health report
        health_report = {
            'timestamp': time.time(),
            'system_info': system_info,
            'process_info': process_info,
            'pipeline_status': monitor_status,
            'active_steps': len(monitor_status.get('steps', {})),
            'health_status': 'healthy' if monitor_status.get('active_steps', 0) == 0 else 'running',
            'warnings': [],
            'errors': []
        }

        # Add warnings if any
        if monitor_status.get('active_steps', 0) > 5:
            health_report['warnings'].append('High number of concurrent pipeline steps')

        if system_info['memory']['percent'] > 90:
            health_report['warnings'].append('High memory usage detected')

        if system_info['cpu_percent'] > 80:
            health_report['warnings'].append('High CPU usage detected')

        return health_report

    except ImportError:
        # psutil not available, return basic report
        return {
            'timestamp': time.time(),
            'system_info': {'note': 'psutil not available for detailed system metrics'},
            'process_info': {'note': 'psutil not available for detailed process metrics'},
            'pipeline_status': get_monitor_status(),
            'active_steps': len(get_monitor_status().get('steps', {})),
            'health_status': 'unknown',
            'warnings': ['psutil not available for system monitoring'],
            'errors': []
        }
    except Exception as e:
        logger.error(f"Error generating pipeline health report: {e}")
        return {
            'timestamp': time.time(),
            'error': str(e),
            'health_status': 'error',
            'warnings': [],
            'errors': [str(e)]
        }