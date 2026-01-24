#!/usr/bin/env python3
"""
Intelligent Analysis analyzer module.

This module provides analysis helper classes and functions for intelligent
pipeline analysis, including health scoring, pattern detection, and
optimization suggestions.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
import logging


@dataclass
class AnalysisContext:
    """Context object for pipeline analysis."""

    summary_data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_name: str = "GNN Pipeline"
    analysis_type: str = "comprehensive"

    @property
    def overall_status(self) -> str:
        """Get overall pipeline status."""
        return self.summary_data.get("overall_status", "UNKNOWN")

    @property
    def total_duration(self) -> float:
        """Get total pipeline duration in seconds."""
        return self.summary_data.get("total_duration_seconds", 0.0)

    @property
    def steps(self) -> List[Dict[str, Any]]:
        """Get list of pipeline steps."""
        return self.summary_data.get("steps", [])

    @property
    def performance_summary(self) -> Dict[str, Any]:
        """Get performance summary data."""
        return self.summary_data.get("performance_summary", {})

    def get_failed_steps(self) -> List[Dict[str, Any]]:
        """Get list of failed steps."""
        return [s for s in self.steps if s.get("status") == "FAILED"]

    def get_successful_steps(self) -> List[Dict[str, Any]]:
        """Get list of successful steps."""
        return [s for s in self.steps if s.get("status", "").startswith("SUCCESS")]

    def get_warning_steps(self) -> List[Dict[str, Any]]:
        """Get list of steps with warnings."""
        return [s for s in self.steps if "WARNING" in s.get("status", "")]


class IntelligentAnalyzer:
    """
    Main analyzer class for intelligent pipeline analysis.

    Provides methods for comprehensive pipeline analysis including
    health scoring, pattern detection, and optimization recommendations.
    """

    def __init__(self, context: Optional[AnalysisContext] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the analyzer.

        Args:
            context: Optional analysis context
            logger: Optional logger instance
        """
        self.context = context
        self.logger = logger or logging.getLogger(__name__)
        self._analysis_cache: Dict[str, Any] = {}

    def set_context(self, context: AnalysisContext) -> None:
        """Set the analysis context."""
        self.context = context
        self._analysis_cache.clear()

    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on the current context.

        Returns:
            Dictionary containing all analysis results
        """
        if not self.context:
            raise ValueError("No analysis context set")

        results = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_name": self.context.pipeline_name,
            "overall_status": self.context.overall_status,
            "health_score": self.calculate_health_score(),
            "failure_analysis": self.analyze_failures(),
            "performance_analysis": self.analyze_performance(),
            "patterns": self.detect_patterns(),
            "optimizations": self.generate_optimizations()
        }

        self._analysis_cache["full_analysis"] = results
        return results

    def calculate_health_score(self) -> float:
        """Calculate overall pipeline health score (0-100)."""
        if not self.context:
            return 0.0

        return calculate_pipeline_health_score(self.context.summary_data)

    def analyze_failures(self) -> Dict[str, Any]:
        """Analyze pipeline failures in detail."""
        if not self.context:
            return {"error": "No context available"}

        failed_steps = self.context.get_failed_steps()

        analysis = {
            "failure_count": len(failed_steps),
            "failures": [],
            "common_patterns": [],
            "severity_distribution": {}
        }

        severity_counts = {"critical": 0, "major": 0, "minor": 0}
        error_patterns: Dict[str, int] = {}

        for step in failed_steps:
            severity = classify_failure_severity(step)
            severity_counts[severity] += 1

            # Extract error patterns
            error_text = step.get("stderr", "") or ""
            patterns = _extract_error_patterns(error_text)
            for pattern in patterns:
                error_patterns[pattern] = error_patterns.get(pattern, 0) + 1

            analysis["failures"].append({
                "step": step.get("script_name"),
                "severity": severity,
                "exit_code": step.get("exit_code"),
                "patterns": patterns,
                "duration": step.get("duration_seconds")
            })

        analysis["severity_distribution"] = severity_counts
        analysis["common_patterns"] = [
            {"pattern": p, "count": c}
            for p, c in sorted(error_patterns.items(), key=lambda x: -x[1])[:5]
        ]

        return analysis

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze pipeline performance characteristics."""
        if not self.context:
            return {"error": "No context available"}

        steps = self.context.steps
        perf_summary = self.context.performance_summary

        durations = [s.get("duration_seconds", 0) for s in steps if s.get("duration_seconds")]
        memories = [s.get("peak_memory_mb", 0) for s in steps if s.get("peak_memory_mb")]

        analysis = {
            "total_duration": self.context.total_duration,
            "step_count": len(steps),
            "peak_memory_mb": perf_summary.get("peak_memory_mb", 0),
            "duration_stats": {},
            "memory_stats": {},
            "bottlenecks": []
        }

        if durations:
            analysis["duration_stats"] = {
                "mean": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "total": sum(durations)
            }

            # Identify bottlenecks (steps taking > 2x average)
            avg_duration = analysis["duration_stats"]["mean"]
            for step in steps:
                duration = step.get("duration_seconds", 0)
                if duration > avg_duration * 2:
                    analysis["bottlenecks"].append({
                        "step": step.get("script_name"),
                        "duration": duration,
                        "ratio": duration / avg_duration if avg_duration > 0 else 0
                    })

        if memories:
            analysis["memory_stats"] = {
                "mean": sum(memories) / len(memories),
                "min": min(memories),
                "max": max(memories)
            }

        return analysis

    def detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns in pipeline execution."""
        if not self.context:
            return []

        return detect_performance_patterns(self.context.summary_data)

    def generate_optimizations(self) -> List[Dict[str, Any]]:
        """Generate optimization suggestions."""
        if not self.context:
            return []

        return generate_optimization_suggestions(self.context.summary_data)


def calculate_pipeline_health_score(summary_data: Dict[str, Any]) -> float:
    """
    Calculate a health score for the pipeline (0-100).

    Factors:
    - Success rate (40%)
    - Warning rate (20%)
    - Duration efficiency (20%)
    - Memory efficiency (20%)

    Args:
        summary_data: Pipeline summary dictionary

    Returns:
        Health score from 0 to 100
    """
    steps = summary_data.get("steps", [])
    perf = summary_data.get("performance_summary", {})

    if not steps:
        return 0.0

    total_steps = len(steps)

    # Success rate component (40%)
    failed = len([s for s in steps if s.get("status") == "FAILED"])
    success_rate = (total_steps - failed) / total_steps
    success_score = success_rate * 40

    # Warning rate component (20%)
    warnings = len([s for s in steps if "WARNING" in s.get("status", "")])
    warning_rate = warnings / total_steps
    warning_score = (1 - warning_rate) * 20

    # Duration efficiency (20%) - penalize if total > 10 minutes
    total_duration = summary_data.get("total_duration_seconds", 0)
    duration_efficiency = max(0, 1 - (total_duration / 600))  # 10 min baseline
    duration_score = duration_efficiency * 20

    # Memory efficiency (20%) - penalize if peak > 2GB
    peak_memory = perf.get("peak_memory_mb", 0)
    memory_efficiency = max(0, 1 - (peak_memory / 2048))  # 2GB baseline
    memory_score = memory_efficiency * 20

    total_score = success_score + warning_score + duration_score + memory_score
    return min(100.0, max(0.0, total_score))


def classify_failure_severity(step: Dict[str, Any]) -> str:
    """
    Classify the severity of a step failure.

    Args:
        step: Step data dictionary

    Returns:
        Severity level: "critical", "major", or "minor"
    """
    error_text = (step.get("stderr", "") or "").lower()
    exit_code = step.get("exit_code", 0)

    # Critical indicators
    critical_patterns = [
        "memory error", "out of memory", "segmentation fault",
        "kernel died", "fatal error", "assertion failed",
        "cannot allocate", "core dumped"
    ]

    for pattern in critical_patterns:
        if pattern in error_text:
            return "critical"

    # High exit codes often indicate severe errors
    if exit_code > 127 or exit_code < 0:
        return "critical"

    # Major indicators
    major_patterns = [
        "exception", "error:", "failed to", "cannot find",
        "permission denied", "timeout", "not found"
    ]

    for pattern in major_patterns:
        if pattern in error_text:
            return "major"

    return "minor"


def detect_performance_patterns(summary_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect patterns in pipeline performance.

    Args:
        summary_data: Pipeline summary dictionary

    Returns:
        List of detected patterns with descriptions
    """
    patterns = []
    steps = summary_data.get("steps", [])

    if not steps:
        return patterns

    # Pattern 1: Cascading failures
    failed_indices = [i for i, s in enumerate(steps) if s.get("status") == "FAILED"]
    if len(failed_indices) >= 2:
        consecutive = all(
            failed_indices[i] + 1 == failed_indices[i + 1]
            for i in range(len(failed_indices) - 1)
        )
        if consecutive:
            patterns.append({
                "type": "cascading_failure",
                "description": f"Detected {len(failed_indices)} consecutive failures starting at step {failed_indices[0] + 1}",
                "severity": "high",
                "recommendation": "Fix the first failure to potentially resolve downstream issues"
            })

    # Pattern 2: Memory growth
    memories = [(i, s.get("peak_memory_mb", 0)) for i, s in enumerate(steps) if s.get("peak_memory_mb")]
    if len(memories) >= 3:
        memory_values = [m[1] for m in memories]
        growth_rate = (memory_values[-1] - memory_values[0]) / len(memory_values)
        if growth_rate > 100:  # Growing by 100MB+ per step
            patterns.append({
                "type": "memory_growth",
                "description": f"Memory usage growing at ~{growth_rate:.0f}MB per step",
                "severity": "medium",
                "recommendation": "Check for memory leaks or consider adding cleanup between steps"
            })

    # Pattern 3: Duration variance
    durations = [s.get("duration_seconds", 0) for s in steps if s.get("duration_seconds")]
    if durations:
        mean_duration = sum(durations) / len(durations)
        variance = sum((d - mean_duration) ** 2 for d in durations) / len(durations)
        if variance > mean_duration ** 2:
            patterns.append({
                "type": "high_variance",
                "description": "Large variance in step durations detected",
                "severity": "low",
                "recommendation": "Some steps may benefit from optimization or parallelization"
            })

    # Pattern 4: Late-stage failures
    total = len(steps)
    late_failures = [i for i in failed_indices if i > total * 0.7]
    if late_failures:
        patterns.append({
            "type": "late_failure",
            "description": f"{len(late_failures)} failure(s) in final 30% of pipeline",
            "severity": "medium",
            "recommendation": "Consider adding earlier validation to fail fast"
        })

    return patterns


def generate_optimization_suggestions(summary_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate optimization suggestions based on analysis.

    Args:
        summary_data: Pipeline summary dictionary

    Returns:
        List of optimization suggestions
    """
    suggestions = []
    steps = summary_data.get("steps", [])
    perf = summary_data.get("performance_summary", {})

    # Suggestion 1: Parallelization opportunities
    independent_steps = []
    for i, step in enumerate(steps):
        if not step.get("dependency_warnings"):
            independent_steps.append(step.get("script_name"))

    if len(independent_steps) >= 3:
        suggestions.append({
            "type": "parallelization",
            "impact": "high",
            "description": f"{len(independent_steps)} steps may be parallelizable",
            "steps": independent_steps[:5],
            "estimated_savings": "20-40% duration reduction"
        })

    # Suggestion 2: Caching opportunities
    slow_steps = [
        s for s in steps
        if s.get("duration_seconds", 0) > 30
    ]
    if slow_steps:
        suggestions.append({
            "type": "caching",
            "impact": "medium",
            "description": f"{len(slow_steps)} slow steps could benefit from caching",
            "steps": [s.get("script_name") for s in slow_steps[:3]],
            "estimated_savings": "30-50% on repeated runs"
        })

    # Suggestion 3: Memory optimization
    peak_memory = perf.get("peak_memory_mb", 0)
    if peak_memory > 1024:
        suggestions.append({
            "type": "memory_optimization",
            "impact": "medium",
            "description": f"Peak memory {peak_memory:.0f}MB exceeds 1GB",
            "recommendation": "Consider streaming processing or chunked data loading",
            "target": "Reduce peak to under 1GB"
        })

    # Suggestion 4: Early termination
    total_duration = summary_data.get("total_duration_seconds", 0)
    failed_step_durations = sum(
        s.get("duration_seconds", 0) for s in steps
        if s.get("status") == "FAILED"
    )
    if failed_step_durations > total_duration * 0.3:
        suggestions.append({
            "type": "early_termination",
            "impact": "high",
            "description": "Significant time spent on failed steps",
            "recommendation": "Add fail-fast checks or pre-validation",
            "potential_savings": f"{failed_step_durations:.0f}s on failure cases"
        })

    return suggestions


def _extract_error_patterns(error_text: str) -> List[str]:
    """Extract common error patterns from error text."""
    patterns = []

    # Common error patterns
    pattern_regexes = [
        (r"ModuleNotFoundError.*?'(\w+)'", "missing_module"),
        (r"ImportError.*?(\w+)", "import_error"),
        (r"FileNotFoundError.*?'(.+?)'", "file_not_found"),
        (r"PermissionError", "permission_error"),
        (r"TimeoutError", "timeout"),
        (r"MemoryError", "memory_error"),
        (r"ConnectionError", "connection_error"),
        (r"TypeError.*?(\w+)", "type_error"),
        (r"ValueError", "value_error"),
        (r"KeyError.*?'(\w+)'", "key_error")
    ]

    for regex, pattern_name in pattern_regexes:
        if re.search(regex, error_text, re.IGNORECASE):
            patterns.append(pattern_name)

    return patterns
