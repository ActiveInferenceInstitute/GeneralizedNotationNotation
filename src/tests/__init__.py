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
- Real implementations (no mocks)
- Performance regression testing
- MCP integration testing
- Pipeline orchestration testing
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator

# Ensure src is in Python path for imports
SRC_DIR = Path(__file__).parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import necessary utilities and helpers from test_utils
from .test_utils import (
    # Constants
    TEST_DIR,
    SRC_DIR,
    PROJECT_ROOT,
    TEST_CONFIG,
    TEST_CATEGORIES,
    TEST_STAGES,
    COVERAGE_TARGETS,
    
    # Utility functions
    is_safe_mode,
    validate_test_environment,
    get_test_args,
    get_sample_pipeline_arguments,
    create_test_gnn_files,
    create_test_files,
    create_sample_gnn_content,
    get_mock_filesystem_structure,
    run_all_tests,
    
    # Performance tracking functions
    performance_tracker,
    get_memory_usage,
    track_peak_memory,
    with_resource_limits,
    
    # Validation functions
    assert_file_exists,
    assert_valid_json,
    assert_directory_structure,
    
    # Report functions
    validate_report_data,
    generate_html_report_file,
    generate_markdown_report_file,
    generate_json_report_file,
    generate_comprehensive_report
)

# Import pytest markers from conftest
try:
    from .conftest import PYTEST_MARKERS
except ImportError:
    # Fallback definition if conftest import fails
    PYTEST_MARKERS = {
        "unit": "Unit tests for individual components",
        "integration": "Integration tests for component interactions", 
        "performance": "Performance and resource usage tests",
        "slow": "Tests that take significant time to complete",
        "fast": "Quick tests for rapid feedback",
        "safe_to_fail": "Tests safe to run without side effects",
        "destructive": "Tests that may modify system state",
        "external": "Tests requiring external dependencies",
        "core": "Core module tests",
        "utilities": "Utility function tests",
        "environment": "Environment validation tests",
        "render": "Rendering and code generation tests",
        "export": "Export functionality tests",
        "parsers": "Parser and format tests"
    }

# Export public interface
__all__ = [
    # Core test constants
    "TEST_DIR",
    "SRC_DIR",
    "PROJECT_ROOT",
    "TEST_CONFIG",
    "TEST_CATEGORIES",
    "TEST_STAGES", 
    "COVERAGE_TARGETS",
    "PYTEST_MARKERS",
    
    # Utility functions
    "is_safe_mode",
    "validate_test_environment",
    "get_test_args",
    "get_sample_pipeline_arguments",
    "create_test_gnn_files",
    "create_test_files",
    "create_sample_gnn_content",
    "get_mock_filesystem_structure",
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
    "__description__"
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "Comprehensive testing for GNN Processing Pipeline"