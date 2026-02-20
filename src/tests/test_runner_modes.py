"""
Test runner mode functions.

Provides run_fast_pipeline_tests(), run_comprehensive_tests(),
and run_fast_reliable_tests() execution modes.

Extracted from runner.py for maintainability.
"""

import logging
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from .categories import MODULAR_TEST_CATEGORIES

# Import from infrastructure
from .infrastructure import (
    TestExecutionConfig,
    TestExecutionResult,
    _generate_markdown_report,
    _generate_fallback_report,
    _generate_timeout_report,
    _generate_error_report,
    _extract_collection_errors,
    _parse_test_statistics,
    _parse_coverage_statistics,
)


def run_fast_pipeline_tests(logger: logging.Logger, output_dir: Path, verbose: bool = False) -> bool:
    """
    Run FAST tests for quick pipeline validation.
    
    This runs only fast tests (marked with 'not slow') to keep pipeline execution efficient.
    """
    if os.getenv("SKIP_TESTS_IN_PIPELINE"):
        logger.info("Skipping tests (SKIP_TESTS_IN_PIPELINE set)")
        logger.info("Set SKIP_TESTS_IN_PIPELINE='' or unset to run tests in pipeline")
        return True

    logger.info("Running fast test subset for quick pipeline validation")
    logger.info("To skip tests in pipeline: export SKIP_TESTS_IN_PIPELINE=1")
    logger.info("To customize timeout: export FAST_TESTS_TIMEOUT=<seconds>")

    try:
        import pytest_timeout
        has_timeout = True
    except ImportError:
        has_timeout = False

    cmd = [
        sys.executable, "-m", "pytest",
        "--tb=short",
        "--maxfail=5",
        "--durations=10",
        "-ra",
    ]
    
    if has_timeout:
        timeout_value = os.getenv("FAST_TESTS_TIMEOUT", "600")
        cmd.extend(["--timeout", timeout_value])

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    cmd.extend([
        "-m", "not slow",
        "--ignore=src/tests/test_llm_ollama.py",
        "--ignore=src/tests/test_llm_ollama_integration.py",
        "--ignore=src/tests/test_pipeline_performance.py",
        "--ignore=src/tests/test_pipeline_recovery.py",
        "--ignore=src/tests/test_report_integration.py",
    ])
    
    cmd.append("src/tests/")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    project_root = Path(__file__).parent.parent.parent
    
    logger.info(f"Executing fast tests: {' '.join(cmd)}")

    try:
        overall_timeout = int(os.getenv("FAST_TESTS_TIMEOUT", "600")) + 30
        
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=overall_timeout
        )

        output_file = output_dir / "pytest_reliable_output.txt"
        with open(output_file, "w") as f:
            f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

        collection_errors = _extract_collection_errors(result.stdout, result.stderr)
        if collection_errors:
            logger.warning(f"Found {len(collection_errors)} collection errors (import/syntax issues)")
            for err in collection_errors[:3]:
                logger.warning(f"  Collection error: {err[:200]}")

        test_stats = _parse_test_statistics(result.stdout)
        coverage_stats = _parse_coverage_statistics(result.stdout)

        summary = {
            "execution_summary": {
                "test_mode": "fast_pipeline",
                "command": " ".join(cmd),
                "exit_code": result.returncode,
                "tests_run": test_stats.get("total", 0),
                "tests_passed": test_stats.get("passed", 0),
                "tests_failed": test_stats.get("failed", 0),
                "tests_skipped": test_stats.get("skipped", 0),
                "tests_errors": test_stats.get("errors", 0),
                "collection_errors": len(collection_errors),
                "coverage": coverage_stats,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        summary_file = output_dir / "test_execution_report.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        _generate_markdown_report(output_dir, summary)

        success = result.returncode == 0
        total = test_stats.get("total", 0)
        passed = test_stats.get("passed", 0)
        failed = test_stats.get("failed", 0)
        skipped = test_stats.get("skipped", 0)

        if success:
            logger.info(f"Fast tests completed: {total} run, {passed} passed, {skipped} skipped")
        else:
            logger.warning(f"Fast tests had failures: {total} run, {passed} passed, {failed} failed, {skipped} skipped")

        return success

    except subprocess.TimeoutExpired:
        logger.error("Fast test execution timed out")
        _generate_timeout_report(output_dir, cmd, overall_timeout)
        return False
    except Exception as e:
        logger.error(f"Fast test execution failed: {e}")
        _generate_error_report(output_dir, cmd, str(e))
        return False


def run_comprehensive_tests(logger: logging.Logger, output_dir: Path, verbose: bool = False) -> bool:
    """
    Run comprehensive tests including slow and performance tests.
    """
    logger.info("Running comprehensive test suite (all categories)")

    from .test_runner_modular import ModularTestRunner

    class ComprehensiveArgs:
        def __init__(self, output_dir, verbose):
            self.output_dir = str(output_dir)
            self.verbose = verbose
            self.categories = None
            self.parallel = True

    args = ComprehensiveArgs(output_dir, verbose)
    runner = ModularTestRunner(args, logger)
    results = runner.run_all_categories()

    success = results.get("overall_success", False)
    if success:
        logger.info("Comprehensive tests completed successfully")
    else:
        logger.warning("Comprehensive tests had some failures")

    return success


def run_fast_reliable_tests(logger: logging.Logger, output_dir: Path, verbose: bool = False) -> bool:
    """
    Run a reliable subset of fast tests with improved error handling.
    """
    logger.info("Running reliable fast test subset")

    reliable_tests = [
        "test_core_modules.py",
        "test_fast_suite.py",
        "test_main_orchestrator.py"
    ]

    cmd = [
        sys.executable, "-m", "pytest",
        "--tb=short",
        "--maxfail=3",
        "--durations=3",
        "-v" if verbose else "-q"
    ]

    test_dir = Path(__file__).parent
    for test_file in reliable_tests:
        test_path = test_dir / test_file
        if test_path.exists():
            cmd.append(str(test_path))

    logger.info(f"Executing reliable tests: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=90
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "pytest_reliable_output.txt", "w") as f:
            f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

        success = result.returncode == 0
        if success:
            logger.info("Reliable fast tests completed successfully")
        else:
            logger.warning("Reliable fast tests had some failures")

        return success

    except subprocess.TimeoutExpired:
        logger.error("Reliable test execution timed out")
        return False
    except Exception as e:
        logger.error(f"Reliable test execution failed: {e}")
        return False
