"""
Test Configuration Classes for GNN Processing Pipeline.

This module provides data classes for test execution configuration and results.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class TestExecutionConfig:
    """Configuration for test execution."""
    timeout_seconds: int = 3600  # Increased to 60 minutes for comprehensive test suite
    max_failures: int = 10
    parallel: bool = True
    coverage: bool = True
    verbose: bool = False
    markers: List[str] = None
    memory_limit_mb: int = 2048
    cpu_limit_percent: int = 80


@dataclass
class TestExecutionResult:
    """Results from test execution."""
    success: bool
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    execution_time: float
    memory_peak_mb: float
    coverage_percentage: Optional[float] = None
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    
    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)
