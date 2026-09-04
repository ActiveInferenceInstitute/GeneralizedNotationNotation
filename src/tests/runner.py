"""
Test Runner for GNN Processing Pipeline.

This module provides comprehensive test execution capabilities for the GNN pipeline.
It implements the core test execution logic, following the thin orchestrator pattern
where 2_tests.py delegates all functionality to this module.

Architecture:
  The module provides multiple test execution modes:
  - run_tests() - Main entry point that routes to appropriate mode
  - run_fast_pipeline_tests() - Fast tests for quick pipeline validation (default)
  - run_comprehensive_tests() - All tests including slow/performance tests
  - run_fast_reliable_tests() - Essential tests recovery mode

  The ModularTestRunner class provides category-based execution with:
  - Resource monitoring (memory, CPU)
  - Timeout handling per category
  - Parallel execution support
  - Comprehensive error recovery

Key Features:
  - Staged test execution (fast, comprehensive, reliable)
  - Parallel test execution with resource monitoring
  - Comprehensive reporting and analytics (JSON, Markdown)
  - Graceful error handling and recovery
  - Performance regression detection
  - Memory usage tracking
  - Coverage analysis integration
  - Collection error detection (import/syntax errors)
  - Category-based test organization

Test Categories:
  Tests are organized into categories defined in MODULAR_TEST_CATEGORIES:
  - gnn, render, mcp, audio, visualization, pipeline, etc.
  Each category has its own timeout, max failures, and parallel execution settings.

Usage:
  from tests import run_tests
  from pathlib import Path
  import logging

  logger = logging.getLogger(__name__)
  success = run_tests(
      logger=logger,
      output_dir=Path("output/2_tests_output"),
      verbose=True,
      fast_only=True
  )

Dependencies:
  - pytest: Test framework
  - pytest-cov: Coverage analysis (optional)
  - pytest-timeout: Per-test timeouts (optional)
  - psutil: Resource monitoring (optional)
"""

import json
import logging
from pathlib import Path
from typing import cast

from utils.pipeline_template import (
    log_step_error,
    log_step_start,
)

# Canonical single-source TestRunner + dependency probe live in
# tests.infrastructure (see infrastructure/test_runner.py).
from .infrastructure import (
    TestRunner,
    check_test_dependencies,
)


def run_tests(
    logger: logging.Logger,
    output_dir: Path,
    verbose: bool = False,
    fast_only: bool = True,  # Default to fast tests for pipeline integration
    comprehensive: bool = False,
    generate_coverage: bool = False,  # Disable coverage by default for speed
    auto_fallback: bool = True,  # Automatically recovery to comprehensive if no fast tests collected
) -> bool:
    """
    Run optimized test suite with improved performance and reliability.

    Args:
        logger: Logger instance
        output_dir: Output directory for test results
        verbose: Enable verbose output
        fast_only: Run only fast tests
        comprehensive: Run comprehensive test suite (all tests)
        generate_coverage: Generate coverage report
        auto_fallback: If fast tests collect 0 tests, automatically try comprehensive

    Returns:
        True if tests pass, False otherwise
    """
    try:
        log_step_start(logger, "Running optimized test suite")

        # Check dependencies (pytest required; cov/xdist/etc. optional — see infrastructure.utils)
        dependencies = check_test_dependencies(logger)
        if not dependencies.get("pytest"):
            log_step_error(logger, "pytest is not installed; aborting test step")
            return False

        # For pipeline integration, run a focused subset of tests
        if fast_only and not comprehensive:
            logger.info("🏃 Running fast pipeline test subset for quick validation")
            success = run_fast_pipeline_tests(logger, output_dir, verbose)

            # Auto-recovery: if no tests collected and recovery enabled, try comprehensive
            if not success and auto_fallback:
                if _check_zero_tests_collected(output_dir, logger):
                    logger.warning(
                        "⚠️ Fast test suite yielded 0 tests. Automatically falling back to comprehensive mode."
                    )
                    return run_comprehensive_tests(logger, output_dir, verbose)

            return success

        # For comprehensive mode, run all tests but with better timeout handling
        if comprehensive:
            logger.info("🔬 Running comprehensive test suite with enhanced monitoring")
            return run_comprehensive_tests(logger, output_dir, verbose)

        # Default to fast tests with improved reliability
        logger.info("⚡ Running fast test suite with reliability improvements")
        return run_fast_reliable_tests(logger, output_dir, verbose)

    except Exception as e:
        log_step_error(logger, f"Test execution failed: {e}")
        return False


def _check_zero_tests_collected(output_dir: Path, logger: logging.Logger) -> bool:
    """Check if the test execution report shows zero tests collected."""
    try:
        summary_file = output_dir / "test_execution_report.json"
        if summary_file.exists():
            summary = json.loads(summary_file.read_text())
            tests_run = summary.get("execution_summary", {}).get("tests_run", 0)
            return cast("bool", tests_run == 0)
    except Exception as e:
        logger.debug(f"Could not check test count: {e}")
    return False


# Re-export from test_runner_modes sub-module.
from .test_runner_modes import (
    run_comprehensive_tests,
    run_fast_pipeline_tests,
    run_fast_reliable_tests,
)

# Re-export from test_runner_modular sub-module.
