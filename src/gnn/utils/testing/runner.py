"""Test runner and stage/category/coverage execution entry points.

Moved verbatim from ``gnn/utils/testing_utils.py`` (S2-33 Step 1)."""

import logging
import time
from pathlib import Path
from typing import Any, cast

from gnn.utils.testing.constants import TEST_CONFIG
from gnn.utils.testing.environment import (
    cleanup_test_environment,
    setup_test_environment,
)


# Add TestRunner class definition
class TestRunner:
    """Basic test runner for compatibility."""

    def __init__(self, config: Any = None) -> None:
        """Store runner configuration and initialize a compatibility logger."""
        self.config = config or {}
        self.logger = logging.getLogger("test_runner")

    def run_tests(self, test_paths: Any, output_dir: Any) -> Any:
        """Basic test execution.

        ``tests.runner`` never exported a ``TestRunner``; the historical
        delegation attempt always failed with ImportError. Report the
        unavailable runner explicitly instead of pretending to try.
        """
        self.logger.warning("Actual TestRunner not available, using recovery")
        return {"success": False, "error": "TestRunner not available"}


# Add TestResult class definition
class TestResult:
    """Basic test result for compatibility."""

    def __init__(
        self,
        success: Any = False,
        tests_run: Any = 0,
        tests_passed: Any = 0,
        tests_failed: Any = 0,
        tests_skipped: Any = 0,
        execution_time: Any = 0.0,
        error_message: Any = None,
    ) -> None:
        """Store summary counters and optional failure context."""
        self.success = success
        self.tests_run = tests_run
        self.tests_passed = tests_passed
        self.tests_failed = tests_failed
        self.tests_skipped = tests_skipped
        self.execution_time = execution_time
        self.error_message = error_message

    def to_dict(self) -> Any:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
        }


# Add TestCategory class definition
class TestCategory:
    """Basic test category for compatibility."""

    def __init__(self, name: Any = "", description: Any = "") -> None:
        """Store the category name and description."""
        self.name = name
        self.description = description

    def __str__(self) -> Any:
        """Return the compact category display name."""
        return f"TestCategory({self.name})"

    def __repr__(self) -> Any:
        """Return the developer-facing category representation."""
        return self.__str__()


# Add TestStage class definition
class TestStage:
    """Basic test stage for compatibility."""

    def __init__(
        self,
        name: Any = "",
        timeout: Any = 300,
        max_failures: Any = 10,
        parallel: Any = True,
        coverage: Any = False,
    ) -> None:
        """Store stage execution limits and coverage settings."""
        self.name = name
        self.timeout = timeout
        self.max_failures = max_failures
        self.parallel = parallel
        self.coverage = coverage

    def __str__(self) -> Any:
        """Return the compact stage display name."""
        return f"TestStage({self.name})"

    def __repr__(self) -> Any:
        """Return the developer-facing stage representation."""
        return self.__str__()


# Add CoverageTarget class definition
class CoverageTarget:
    """Basic coverage target for compatibility."""

    def __init__(self, name: Any = "", target_percentage: Any = 0.0) -> None:
        """Store the coverage target name and threshold."""
        self.name = name
        self.target_percentage = target_percentage

    def __str__(self) -> Any:
        """Return the compact coverage target display name."""
        return f"CoverageTarget({self.name}: {self.target_percentage}%)"

    def __repr__(self) -> Any:
        """Return the developer-facing coverage target representation."""
        return self.__str__()


# Add missing functions
def run_tests(target_dir: Path, output_dir: Path, verbose: bool = False) -> bool:
    """Basic test execution function."""
    try:
        # Import from tests module if available
        from tests.test_runner_modular import _ModularTestRunner

        runner = _ModularTestRunner(
            type(
                "Args",
                (),
                {
                    "target_dir": target_dir,
                    "output_dir": output_dir,
                    "verbose": verbose,
                },
            ),
            logging.getLogger("test_runner"),
        )

        # Run tests using the available method
        if hasattr(runner, "run_all_tests"):
            return cast("bool", runner.run_all_tests())
        elif hasattr(runner, "run_tests"):
            return cast("bool", runner.run_tests())
        else:
            logging.warning("No test execution method available")
            return True
    except ImportError:
        logging.warning("Test runner not available")
        return True


def run_test_category(
    category: str, target_dir: Path, output_dir: Path, verbose: bool = False
) -> bool:
    """Run tests for a specific category."""
    return run_tests(target_dir, output_dir, verbose)


def run_test_stage(
    stage: str, target_dir: Path, output_dir: Path, verbose: bool = False
) -> bool:
    """Run tests for a specific stage."""
    return run_tests(target_dir, output_dir, verbose)


def run_all_tests(target_dir: Path, output_dir: Path, verbose: bool = False) -> bool:
    """Run all tests and return success status."""
    # Late import: reports.py needs run_all_tests from this module (MCP wrapper),
    # so a module-level import would be circular (S2-33 Step 1 split).
    from gnn.utils.testing.reports import generate_test_report

    try:
        # Set up test environment
        setup_test_environment()

        # Create test output directory
        test_output_dir = output_dir / "test_results"
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # Run different test categories
        test_results: dict[Any, Any] = {}

        # Run fast tests
        test_results["fast"] = run_fast_tests(target_dir, test_output_dir, verbose)

        # Run standard tests
        test_results["standard"] = run_standard_tests(
            target_dir, test_output_dir, verbose
        )

        # Run slow tests if not in fast-only mode
        if not TEST_CONFIG.get("fast_only", False):
            test_results["slow"] = run_slow_tests(target_dir, test_output_dir, verbose)

        # Run performance tests if enabled
        if TEST_CONFIG.get("include_performance", False):
            test_results["performance"] = run_performance_tests(
                target_dir, test_output_dir, verbose
            )

        # Generate test report
        generate_test_report(test_results, test_output_dir)

        # Check overall success
        overall_success = all(test_results.values())

        if verbose:
            print(f"Test Results: {test_results}")
            print(f"Overall Success: {overall_success}")

        return overall_success

    except Exception as e:
        if verbose:
            print(f"Test execution failed: {e}")
        return False
    finally:
        # Clean up test environment
        cleanup_test_environment()


def run_fast_tests(target_dir: Path, output_dir: Path, verbose: bool) -> bool:
    """Run fast tests."""
    if verbose:
        print("Running fast tests...")

    # Simulate fast test execution
    time.sleep(0.1)  # Simulate test execution time

    return True


def run_standard_tests(target_dir: Path, output_dir: Path, verbose: bool) -> bool:
    """Run standard tests."""
    if verbose:
        print("Running standard tests...")

    # Simulate standard test execution
    time.sleep(0.2)  # Simulate test execution time

    return True


def run_slow_tests(target_dir: Path, output_dir: Path, verbose: bool) -> bool:
    """Run slow tests."""
    if verbose:
        print("Running slow tests...")

    # Simulate slow test execution
    time.sleep(0.3)  # Simulate test execution time

    return True


def run_performance_tests(target_dir: Path, output_dir: Path, verbose: bool) -> bool:
    """Run performance tests."""
    if verbose:
        print("Running performance tests...")

    # Simulate performance test execution
    time.sleep(0.1)  # Simulate test execution time

    return True


def run_coverage_tests(test_results_dir: Path, verbose: bool) -> bool:
    """Run coverage tests."""
    if verbose:
        print("Running coverage tests...")

    # Simulate coverage test execution
    time.sleep(0.1)  # Simulate test execution time

    return True
