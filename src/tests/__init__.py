"""
Comprehensive Test Suite for GNN Processing Pipeline.

This module provides a complete testing framework with:
- Unit tests for all pipeline components
- Integration tests for end-to-end workflows
- Performance tests for scalability validation
- Coverage analysis for code quality assurance
- Safe-to-fail implementations for graceful degradation

Test Categories:
- Fast: Quick validation tests (< 1s each)
- Standard: Integration and module tests (< 10s each)
- Slow: Complex scenarios and benchmarks (< 60s each)
- Performance: Resource usage and scalability tests

Architecture:
- Modular test organization by component
- Comprehensive fixtures and utilities
- Real implementations and real artifacts
- Performance regression testing
- MCP integration testing
- Pipeline orchestration testing
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Ensure src is in Python path for imports
SRC_DIR = Path(__file__).parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    # Also expose tests as a top-level import alias so 'from tests.conftest import *' works
    try:
        import types as _types

        pkg = _types.ModuleType("tests")
        pkg.__path__ = [str(Path(__file__).parent)]
        sys.modules.setdefault("tests", pkg)
    except Exception:
        pass

from utils.test_utils import (
    COVERAGE_TARGETS,
    PROJECT_ROOT,
    SRC_DIR,
    TEST_CATEGORIES,
    TEST_CONFIG,
    TEST_DIR,
    TEST_STAGES,
    assert_directory_structure,
    assert_file_exists,
    assert_valid_json,
    create_sample_gnn_content,
    create_test_files,
    create_test_gnn_files,
    generate_comprehensive_report,
    generate_html_report_file,
    generate_json_report_file,
    generate_markdown_report_file,
    get_memory_usage,
    get_sample_pipeline_arguments,
    get_test_args,
    get_test_filesystem_structure,
    is_safe_mode,
    performance_tracker,
    run_all_tests,
    track_peak_memory,
    validate_report_data,
    validate_test_environment,
    with_resource_limits,
)

from .conftest import PYTEST_MARKERS
from .runner import run_tests
from .test_runner_modular import create_test_runner

# Export public interface
__all__: list[Any] = [
    # Core test constants
    "SRC_DIR",
    "PROJECT_ROOT",
    "TEST_DIR",
    "TEST_CONFIG",
    "TEST_CATEGORIES",
    "TEST_STAGES",
    "COVERAGE_TARGETS",
    "PYTEST_MARKERS",
    # Test runner functions
    "run_tests",
    "create_test_runner",
    # Utility functions
    "is_safe_mode",
    "validate_test_environment",
    "get_test_args",
    "get_sample_pipeline_arguments",
    "create_test_gnn_files",
    "create_test_files",
    "create_sample_gnn_content",
    "get_test_filesystem_structure",
    "run_all_tests",
    # Performance tracking functions
    "performance_tracker",
    "get_memory_usage",
    "track_peak_memory",
    "with_resource_limits",
    # Validation functions
    "assert_file_exists",
    "assert_valid_json",
    "assert_directory_structure",
    # Report functions
    "validate_report_data",
    "generate_html_report_file",
    "generate_markdown_report_file",
    "generate_json_report_file",
    "generate_comprehensive_report",
    # Module metadata
    "__version__",
    "__author__",
    "__description__",
]

# Module metadata
__version__ = "1.6.0"

FEATURES: dict[str, Any] = {
    "pytest_integration": True,
    "test_discovery": True,
    "fixture_management": True,
    "coverage_tracking": True,
    "mcp_integration": True,
}

__author__ = "Active Inference Institute"
__description__ = "Comprehensive testing for GNN Processing Pipeline"


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "tests",
        "version": __version__,
        "description": "Comprehensive test suite infrastructure and execution",
        "features": FEATURES,
    }
