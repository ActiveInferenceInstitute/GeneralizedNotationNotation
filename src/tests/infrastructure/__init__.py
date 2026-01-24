"""
Test Infrastructure Module.

This module provides the core infrastructure for test execution in the GNN pipeline.
All components are re-exported for convenient access.
"""

# Re-export all public classes and functions for backward compatibility
from .test_config import TestExecutionConfig, TestExecutionResult
from .resource_monitor import ResourceMonitor, PSUTIL_AVAILABLE
from .test_runner import TestRunner
from .report_generator import (
    generate_markdown_report,
    generate_fallback_report,
    generate_timeout_report,
    generate_error_report,
    # Backward-compatible aliases
    _generate_markdown_report,
    _generate_fallback_report,
    _generate_timeout_report,
    _generate_error_report,
)
from .utils import (
    check_test_dependencies,
    build_pytest_command,
    extract_collection_errors,
    parse_test_statistics,
    parse_coverage_statistics,
    # Backward-compatible aliases
    _extract_collection_errors,
    _parse_test_statistics,
    _parse_coverage_statistics,
)

__all__ = [
    # Configuration
    "TestExecutionConfig",
    "TestExecutionResult",
    # Monitoring
    "ResourceMonitor",
    "PSUTIL_AVAILABLE",
    # Runners
    "TestRunner",
    # Report generation
    "generate_markdown_report",
    "generate_fallback_report",
    "generate_timeout_report",
    "generate_error_report",
    "_generate_markdown_report",
    "_generate_fallback_report",
    "_generate_timeout_report",
    "_generate_error_report",
    # Utilities
    "check_test_dependencies",
    "build_pytest_command",
    "extract_collection_errors",
    "parse_test_statistics",
    "parse_coverage_statistics",
    "_extract_collection_errors",
    "_parse_test_statistics",
    "_parse_coverage_statistics",
]
