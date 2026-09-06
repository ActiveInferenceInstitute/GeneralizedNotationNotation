"""
Test Infrastructure Module.

This module provides the core infrastructure for test execution in the GNN pipeline.
All components are re-exported for convenient access.
"""

# Re-export public classes and functions for test infrastructure callers.
from typing import Any

from .report_generator import (
    _generate_error_report,
    _generate_fallback_report,
    # Underscore-prefixed exports.
    _generate_markdown_report,
    _generate_timeout_report,
    generate_error_report,
    generate_fallback_report,
    generate_markdown_report,
    generate_timeout_report,
)
from .resource_monitor import PSUTIL_AVAILABLE, ResourceMonitor
from .test_config import TestExecutionConfig, TestExecutionResult
from .test_runner import TestRunner
from .utils import (
    # Underscore-prefixed exports.
    _extract_collection_errors,
    _parse_coverage_statistics,
    _parse_test_statistics,
    build_pytest_command,
    check_test_dependencies,
    extract_collection_errors,
    parse_coverage_statistics,
    parse_test_statistics,
)

__all__: list[Any] = [
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
