#!/usr/bin/env python3
"""
Pipeline Health Monitoring System

This module provides comprehensive monitoring capabilities for the GNN pipeline,
including health checks, performance tracking, failure detection, and alerting.
"""

import time
import logging
import threading
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    FAILING = "failing"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class StepMetrics:
    """Metrics for a single pipeline step."""
    step_name: str
    executions: int = 0
    successes: int = 0
    failures: int = 0
    warnings: int = 0
    total_duration: float = 0.0
    last_execution: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=10))
    error_types: Dict[str, int] = field(default_factory=dict)
    health_status: HealthStatus = HealthStatus.UNKNOWN
    
    def update_duration(self, duration: float):
        """Update duration statistics."""
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.recent_durations.append(duration)
        if self.executions > 0:
            self.avg_duration = self.total_duration / self.executions
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.executions == 0:
            return 0.0
        return (self.successes / self.executions) * 100
    
    def get_recent_avg_duration(self) -> float:
        """Get average duration of recent executions."""
        if not self.recent_durations:
            return 0.0
        return sum(self.recent_durations) / len(self.recent_durations)

@dataclass
class Alert:
    """Pipeline alert."""
    level: AlertLevel
    message: str
    step_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

@dataclass
class PipelineHealth:
    """Overall pipeline health status."""
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    total_steps: int = 0
    healthy_steps: int = 0
    degraded_steps: int = 0
    failing_steps: int = 0
    critical_steps: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    active_alerts: int = 0
    total_executions: int = 0
    total_failures: int = 0
    
    def get_health_percentage(self) -> float:
        """Get percentage of healthy steps."""
        if self.total_steps == 0:
            return 0.0
        return (self.healthy_steps / self.total_steps) * 100

class PipelineMonitor:
    """Comprehensive pipeline monitoring system."""
    
    def __init__(self, alert_callbacks: Optional[List[Callable]] = None):
        self.logger = logging.getLogger(__name__)
        self.step_metrics: Dict[str, StepMetrics] = {}
        self.alerts: deque = deque(maxlen=100)  # Keep last 100 alerts
        self.alert_callbacks = alert_callbacks or []
        self.monitoring_active = False
        self.health_thresholds = self._initialize_health_thresholds()
        self.performance_baselines = {}
        self._lock = threading.Lock()
        
    def _initialize_health_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize health check thresholds."""
        return {
            "success_rate": {
                "healthy": 95.0,      # >= 95% success rate
                "degraded": 80.0,     # 80-94% success rate
                "failing": 50.0,      # 50-79% success rate
                # < 50% is critical
            },
            "duration_variance": {
                "healthy": 1.5,       # <= 1.5x baseline
                "degraded": 2.0,      # 1.5-2x baseline
                "failing": 3.0,       # 2-3x baseline
                # > 3x baseline is critical
            },
            "failure_rate": {
                "healthy": 5.0,       # <= 5% failure rate
                "degraded": 20.0,     # 5-20% failure rate
                "failing": 50.0,      # 20-50% failure rate
                # > 50% is critical
            }
        }
    
    def start_monitoring(self):
        """Start pipeline monitoring."""
        with self._lock:
            self.monitoring_active = True
            self.logger.info("Pipeline monitoring started")
    
    def stop_monitoring(self):
        """Stop pipeline monitoring."""
        with self._lock:
            self.monitoring_active = False
            self.logger.info("Pipeline monitoring stopped")
    
    def record_step_start(self, step_name: str, context: Dict[str, Any] = None) -> str:
        """
        Record the start of a pipeline step.
        
        Args:
            step_name: Name of the pipeline step
            context: Additional context information
            
        Returns:
            Execution ID for tracking
        """
        execution_id = f"{step_name}_{int(time.time() * 1000)}"
        
        with self._lock:
            if step_name not in self.step_metrics:
                self.step_metrics[step_name] = StepMetrics(step_name=step_name)
            
            metrics = self.step_metrics[step_name]
            metrics.executions += 1
            metrics.last_execution = datetime.now()
        
        self.logger.debug(f"Step started: {step_name} (ID: {execution_id})")
        return execution_id
    
    def record_step_success(self, step_name: str, execution_id: str, 
                           duration: float, context: Dict[str, Any] = None):
        """Record successful completion of a pipeline step."""
        with self._lock:
            if step_name not in self.step_metrics:
                self.step_metrics[step_name] = StepMetrics(step_name=step_name)
            
            metrics = self.step_metrics[step_name]
            metrics.successes += 1
            metrics.last_success = datetime.now()
            metrics.update_duration(duration)
            
            # Update health status
            self._update_step_health(step_name)
        
        self.logger.debug(f"Step succeeded: {step_name} in {duration:.2f}s")
        
        # Check for performance alerts
        self._check_performance_alerts(step_name, duration)
    
    def record_step_failure(self, step_name: str, execution_id: str,
                           duration: float, error_type: str = "unknown",
                           error_message: str = "", context: Dict[str, Any] = None):
        """Record failure of a pipeline step."""
        with self._lock:
            if step_name not in self.step_metrics:
                self.step_metrics[step_name] = StepMetrics(step_name=step_name)
            
            metrics = self.step_metrics[step_name]
            metrics.failures += 1
            metrics.last_failure = datetime.now()
            metrics.update_duration(duration)
            
            # Track error types
            metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1
            
            # Update health status
            self._update_step_health(step_name)
        
        self.logger.warning(f"Step failed: {step_name} after {duration:.2f}s - {error_type}")
        
        # Generate failure alert
        self._generate_alert(
            level=AlertLevel.ERROR,
            message=f"Step {step_name} failed: {error_message[:100]}",
            step_name=step_name,
            details={
                "error_type": error_type,
                "duration": duration,
                "execution_id": execution_id
            }
        )
    
    def record_step_warning(self, step_name: str, execution_id: str,
                           warning_message: str, context: Dict[str, Any] = None):
        """Record a warning for a pipeline step."""
        with self._lock:
            if step_name not in self.step_metrics:
                self.step_metrics[step_name] = StepMetrics(step_name=step_name)
            
            metrics = self.step_metrics[step_name]
            metrics.warnings += 1
        
        self.logger.warning(f"Step warning: {step_name} - {warning_message}")
        
        # Generate warning alert for critical warnings
        if any(keyword in warning_message.lower() for keyword in ['critical', 'severe', 'timeout']):
            self._generate_alert(
                level=AlertLevel.WARNING,
                message=f"Step {step_name} warning: {warning_message[:100]}",
                step_name=step_name,
                details={"execution_id": execution_id}
            )
    
    def _update_step_health(self, step_name: str):
        """Update health status for a step based on metrics."""
        metrics = self.step_metrics[step_name]
        
        if metrics.executions == 0:
            metrics.health_status = HealthStatus.UNKNOWN
            return
        
        success_rate = metrics.get_success_rate()
        failure_rate = (metrics.failures / metrics.executions) * 100
        
        # Determine health status based on thresholds
        if success_rate >= self.health_thresholds["success_rate"]["healthy"]:
            if failure_rate <= self.health_thresholds["failure_rate"]["healthy"]:
                metrics.health_status = HealthStatus.HEALTHY
            else:
                metrics.health_status = HealthStatus.DEGRADED
        elif success_rate >= self.health_thresholds["success_rate"]["degraded"]:
            metrics.health_status = HealthStatus.DEGRADED
        elif success_rate >= self.health_thresholds["success_rate"]["failing"]:
            metrics.health_status = HealthStatus.FAILING
        else:
            metrics.health_status = HealthStatus.CRITICAL
        
        # Check for recent performance degradation
        if len(metrics.recent_durations) >= 3:
            recent_avg = metrics.get_recent_avg_duration()
            if metrics.avg_duration > 0:
                performance_ratio = recent_avg / metrics.avg_duration
                if performance_ratio > self.health_thresholds["duration_variance"]["failing"]:
                    # Performance is significantly worse than average
                    if metrics.health_status == HealthStatus.HEALTHY:
                        metrics.health_status = HealthStatus.DEGRADED
    
    def _check_performance_alerts(self, step_name: str, duration: float):
        """Check for performance-related alerts."""
        metrics = self.step_metrics[step_name]
        
        # Check against baseline if available
        if step_name in self.performance_baselines:
            baseline = self.performance_baselines[step_name]
            performance_ratio = duration / baseline
            
            if performance_ratio > self.health_thresholds["duration_variance"]["critical"]:
                self._generate_alert(
                    level=AlertLevel.CRITICAL,
                    message=f"Step {step_name} performance critically degraded: {performance_ratio:.1f}x baseline",
                    step_name=step_name,
                    details={"duration": duration, "baseline": baseline, "ratio": performance_ratio}
                )
            elif performance_ratio > self.health_thresholds["duration_variance"]["failing"]:
                self._generate_alert(
                    level=AlertLevel.WARNING,
                    message=f"Step {step_name} performance degraded: {performance_ratio:.1f}x baseline",
                    step_name=step_name,
                    details={"duration": duration, "baseline": baseline, "ratio": performance_ratio}
                )
    
    def _generate_alert(self, level: AlertLevel, message: str, step_name: Optional[str] = None,
                       details: Dict[str, Any] = None):
        """Generate and dispatch an alert."""
        alert = Alert(
            level=level,
            message=message,
            step_name=step_name,
            details=details or {}
        )
        
        with self._lock:
            self.alerts.append(alert)
        
        # Log the alert
        log_method = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.ERROR: self.logger.error,
            AlertLevel.CRITICAL: self.logger.critical
        }[level]
        
        log_method(f"[ALERT:{level.value.upper()}] {message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def get_pipeline_health(self) -> PipelineHealth:
        """Get overall pipeline health status."""
        with self._lock:
            health = PipelineHealth()
            
            if not self.step_metrics:
                return health
            
            # Count steps by health status
            for metrics in self.step_metrics.values():
                health.total_steps += 1
                health.total_executions += metrics.executions
                health.total_failures += metrics.failures
                
                if metrics.health_status == HealthStatus.HEALTHY:
                    health.healthy_steps += 1
                elif metrics.health_status == HealthStatus.DEGRADED:
                    health.degraded_steps += 1
                elif metrics.health_status == HealthStatus.FAILING:
                    health.failing_steps += 1
                elif metrics.health_status == HealthStatus.CRITICAL:
                    health.critical_steps += 1
            
            # Determine overall status
            if health.critical_steps > 0:
                health.overall_status = HealthStatus.CRITICAL
            elif health.failing_steps > 0:
                health.overall_status = HealthStatus.FAILING
            elif health.degraded_steps > 0:
                health.overall_status = HealthStatus.DEGRADED
            elif health.healthy_steps > 0:
                health.overall_status = HealthStatus.HEALTHY
            else:
                health.overall_status = HealthStatus.UNKNOWN
            
            # Count active alerts
            health.active_alerts = len([a for a in self.alerts if not a.resolved])
            
            return health
    
    def get_step_metrics(self, step_name: str) -> Optional[StepMetrics]:
        """Get metrics for a specific step."""
        with self._lock:
            return self.step_metrics.get(step_name)
    
    def get_all_metrics(self) -> Dict[str, StepMetrics]:
        """Get metrics for all steps."""
        with self._lock:
            return dict(self.step_metrics)
    
    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        """Get recent alerts."""
        with self._lock:
            return list(self.alerts)[-count:]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get unresolved alerts."""
        with self._lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_index: int):
        """Mark an alert as resolved."""
        with self._lock:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].resolved = True
    
    def set_performance_baseline(self, step_name: str, baseline_duration: float):
        """Set performance baseline for a step."""
        self.performance_baselines[step_name] = baseline_duration
        self.logger.info(f"Performance baseline set for {step_name}: {baseline_duration:.2f}s")
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        health = self.get_pipeline_health()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": {
                "status": health.overall_status.value,
                "health_percentage": health.get_health_percentage(),
                "total_steps": health.total_steps,
                "healthy_steps": health.healthy_steps,
                "degraded_steps": health.degraded_steps,
                "failing_steps": health.failing_steps,
                "critical_steps": health.critical_steps,
                "active_alerts": health.active_alerts,
                "total_executions": health.total_executions,
                "total_failures": health.total_failures
            },
            "step_details": {},
            "recent_alerts": [],
            "performance_summary": {
                "fastest_step": None,
                "slowest_step": None,
                "most_reliable_step": None,
                "least_reliable_step": None
            }
        }
        
        # Step details
        fastest_duration = float('inf')
        slowest_duration = 0
        highest_success_rate = 0
        lowest_success_rate = 100
        
        for step_name, metrics in self.step_metrics.items():
            step_info = {
                "health_status": metrics.health_status.value,
                "executions": metrics.executions,
                "successes": metrics.successes,
                "failures": metrics.failures,
                "warnings": metrics.warnings,
                "success_rate": metrics.get_success_rate(),
                "avg_duration": metrics.avg_duration,
                "min_duration": metrics.min_duration if metrics.min_duration != float('inf') else 0,
                "max_duration": metrics.max_duration,
                "recent_avg_duration": metrics.get_recent_avg_duration(),
                "last_execution": metrics.last_execution.isoformat() if metrics.last_execution else None,
                "last_success": metrics.last_success.isoformat() if metrics.last_success else None,
                "last_failure": metrics.last_failure.isoformat() if metrics.last_failure else None,
                "error_types": dict(metrics.error_types)
            }
            
            report["step_details"][step_name] = step_info
            
            # Track performance extremes
            if metrics.avg_duration > 0:
                if metrics.avg_duration < fastest_duration:
                    fastest_duration = metrics.avg_duration
                    report["performance_summary"]["fastest_step"] = {
                        "name": step_name,
                        "avg_duration": metrics.avg_duration
                    }
                
                if metrics.avg_duration > slowest_duration:
                    slowest_duration = metrics.avg_duration
                    report["performance_summary"]["slowest_step"] = {
                        "name": step_name,
                        "avg_duration": metrics.avg_duration
                    }
            
            # Track reliability extremes
            success_rate = metrics.get_success_rate()
            if metrics.executions > 0:
                if success_rate > highest_success_rate:
                    highest_success_rate = success_rate
                    report["performance_summary"]["most_reliable_step"] = {
                        "name": step_name,
                        "success_rate": success_rate
                    }
                
                if success_rate < lowest_success_rate:
                    lowest_success_rate = success_rate
                    report["performance_summary"]["least_reliable_step"] = {
                        "name": step_name,
                        "success_rate": success_rate
                    }
        
        # Recent alerts
        recent_alerts = self.get_recent_alerts(10)
        for alert in recent_alerts:
            report["recent_alerts"].append({
                "level": alert.level.value,
                "message": alert.message,
                "step_name": alert.step_name,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "details": alert.details
            })
        
        return report
    
    def save_health_report(self, output_dir: Path, filename: Optional[str] = None) -> Path:
        """Save health report to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_health_report_{timestamp}.json"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / filename
        
        report = self.generate_health_report()
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Health report saved to {report_file}")
        return report_file
    
    def check_circuit_breaker(self, step_name: str, failure_threshold: int = 5,
                             time_window: int = 300) -> bool:
        """
        Check if circuit breaker should be triggered for a step.
        
        Args:
            step_name: Name of the step to check
            failure_threshold: Number of failures to trigger circuit breaker
            time_window: Time window in seconds to check failures
            
        Returns:
            True if circuit breaker should be triggered
        """
        if step_name not in self.step_metrics:
            return False
        
        metrics = self.step_metrics[step_name]
        
        # Check recent failure rate
        recent_time = datetime.now() - timedelta(seconds=time_window)
        
        # For simplicity, use recent metrics
        # In a full implementation, we'd track individual execution times
        if metrics.last_failure and metrics.last_failure > recent_time:
            recent_failure_rate = metrics.failures / max(metrics.executions, 1)
            if recent_failure_rate * metrics.executions >= failure_threshold:
                self._generate_alert(
                    level=AlertLevel.CRITICAL,
                    message=f"Circuit breaker triggered for {step_name}: {failure_threshold} failures in {time_window}s",
                    step_name=step_name,
                    details={
                        "failure_threshold": failure_threshold,
                        "time_window": time_window,
                        "recent_failures": metrics.failures
                    }
                )
                return True
        
        return False
    
    def reset_metrics(self, step_name: Optional[str] = None):
        """Reset metrics for a step or all steps."""
        with self._lock:
            if step_name:
                if step_name in self.step_metrics:
                    del self.step_metrics[step_name]
                    self.logger.info(f"Metrics reset for step: {step_name}")
            else:
                self.step_metrics.clear()
                self.alerts.clear()
                self.logger.info("All metrics reset")

# Global monitor instance
pipeline_monitor = PipelineMonitor()

def start_pipeline_monitoring():
    """Start global pipeline monitoring."""
    pipeline_monitor.start_monitoring()

def stop_pipeline_monitoring():
    """Stop global pipeline monitoring."""
    pipeline_monitor.stop_monitoring()

def record_step_execution(step_name: str, success: bool, duration: float,
                         error_type: str = None, error_message: str = None,
                         context: Dict[str, Any] = None) -> str:
    """
    Convenience function to record step execution.
    
    Args:
        step_name: Name of the pipeline step
        success: Whether the step succeeded
        duration: Duration in seconds
        error_type: Type of error if failed
        error_message: Error message if failed
        context: Additional context
        
    Returns:
        Execution ID
    """
    execution_id = pipeline_monitor.record_step_start(step_name, context)
    
    if success:
        pipeline_monitor.record_step_success(step_name, execution_id, duration, context)
    else:
        pipeline_monitor.record_step_failure(
            step_name, execution_id, duration, 
            error_type or "unknown", error_message or "", context
        )
    
    return execution_id

def get_pipeline_health_status() -> Dict[str, Any]:
    """Get current pipeline health status."""
    health = pipeline_monitor.get_pipeline_health()
    return {
        "status": health.overall_status.value,
        "health_percentage": health.get_health_percentage(),
        "total_steps": health.total_steps,
        "active_alerts": health.active_alerts,
        "last_update": health.last_update.isoformat()
    } 