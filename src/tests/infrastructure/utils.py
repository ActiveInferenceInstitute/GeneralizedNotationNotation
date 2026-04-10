"""
Test Utility Functions for GNN Processing Pipeline.

This module provides utility functions for pytest command building,
output parsing, and dependency checking.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from utils.test_utils import TEST_DIR


def check_test_dependencies(logger: logging.Logger) -> Dict[str, Any]:
    """
    Check if required test dependencies are available.
    
    Verifies that pytest and optional dependencies (pytest-cov, pytest-timeout)
    are installed and available.
    
    Args:
        logger: Logger instance for reporting
    
    Returns:
        Dictionary mapping package label to availability (``pytest`` required for the
        test step; ``pytest-cov``, ``pytest-xdist``, ``psutil``, ``coverage`` optional).
    """
    dependencies = {
        "pytest": False,
        "pytest-cov": False,
        "pytest-xdist": False,
        "psutil": False,
        "coverage": False
    }

    try:
        import pytest  # noqa: F811 - presence check
        dependencies["pytest"] = True
    except ImportError:
        pass

    try:
        import pytest_cov  # noqa: F811 - presence check
        dependencies["pytest-cov"] = True
    except ImportError:
        pass

    try:
        import xdist  # noqa: F811 - presence check
        dependencies["pytest-xdist"] = True
    except ImportError:
        pass

    try:
        import psutil  # noqa: F811 - presence check
        dependencies["psutil"] = True
    except ImportError:
        pass

    try:
        import coverage  # noqa: F811 - presence check
        dependencies["coverage"] = True
    except ImportError:
        pass

    # Log results: only pytest is required for fast pipeline tests; others are dev/CI extras.
    if not dependencies["pytest"]:
        logger.warning("pytest is not installed; the test step cannot run.")
    else:
        optional_missing = [
            name for name, available in dependencies.items()
            if not available and name != "pytest"
        ]
        if optional_missing:
            logger.info(
                "Optional test tooling not installed: %s "
                "(parallel runs, coverage reports: uv sync --extra dev)",
                optional_missing,
            )
        else:
            logger.info("All test tooling packages present (pytest + optional extras).")

    return dependencies


def build_pytest_command(
    test_markers: List[str] = None,
    timeout_seconds: int = 600,
    max_failures: int = 20,
    parallel: bool = True,
    verbose: bool = False,
    generate_coverage: bool = True,
    fast_only: bool = False,
    include_slow: bool = False
) -> List[str]:
    """
    Build pytest command with appropriate options.
    
    Constructs a pytest command line with all necessary flags and options
    based on the provided parameters. Handles test filtering, timeout,
    coverage, and execution mode settings.
    
    Args:
        test_markers: List of pytest markers to include (e.g., ['fast', 'unit'])
        timeout_seconds: Maximum execution time per test (default: 600)
        max_failures: Maximum number of test failures before stopping (default: 20)
        parallel: Enable parallel test execution (default: True)
        verbose: Enable verbose output (default: False)
        generate_coverage: Generate coverage reports (default: True)
        fast_only: Run only fast tests, exclude slow tests (default: False)
        include_slow: Include slow tests (default: False)
    
    Returns:
        List of command arguments for subprocess.run()
    
    Example:
        cmd = build_pytest_command(
            test_markers=['fast'],
            timeout_seconds=120,
            max_failures=5,
            verbose=True,
            fast_only=True
        )
        # Returns: ['python', '-m', 'pytest', '--verbose', '--tb=short', ...]
    """
    cmd = [
        sys.executable, "-m", "pytest",
        "--verbose" if verbose else "--quiet",
        "--tb=short",
        "--log-cli-level=WARNING",
        f"--maxfail={max_failures}",
        "--durations=10",
        "--disable-warnings"
    ]

    # Add markers
    if test_markers:
        for marker in test_markers:
            cmd.extend(["-m", marker])

    # Add coverage if enabled
    if generate_coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=json",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])

    # Add parallel execution if enabled
    if parallel:
        cmd.extend(["-n", "auto"])

    # Add test path
    cmd.append(str(TEST_DIR))

    return cmd


def extract_collection_errors(stdout: str, stderr: str) -> List[str]:
    """
    Extract and parse collection errors from pytest output.
    
    Detects import errors, syntax errors, and other collection failures that
    prevent tests from being collected. Returns a list of unique error messages
    with actionable information.
    
    Args:
        stdout: Standard output from pytest execution
        stderr: Standard error from pytest execution
    
    Returns:
        List of unique error messages (strings)
    
    Error Types Detected:
        - ERROR collecting: Test file collection failures
        - NameError: Missing variable/import names
        - ImportError: Module import failures
        - SyntaxError: Code syntax issues
    
    Example:
        errors = extract_collection_errors(pytest_stdout, pytest_stderr)
        # Returns: ["test_file.py: ImportError: No module named 'missing_module'"]
    """
    errors = []
    combined_output = stdout + "\n" + stderr

    # Look for ERROR collecting patterns
    error_patterns = [
        r'ERROR collecting ([^\n]+)\n([^\n]+: [^\n]+)',
        r'NameError: name \'([^\']+)\' is not defined',
        r'ImportError: ([^\n]+)',
        r'SyntaxError: ([^\n]+)',
    ]

    for pattern in error_patterns:
        matches = re.finditer(pattern, combined_output, re.MULTILINE)
        for match in matches:
            error_msg = match.group(0)
            # Extract the key part of the error
            if 'ERROR collecting' in error_msg:
                # Extract the file and error message
                lines = error_msg.split('\n')
                if len(lines) >= 2:
                    file_line = lines[0].replace('ERROR collecting ', '').strip()
                    error_line = lines[1].strip()
                    errors.append(f"{file_line}: {error_line}")
            elif 'NameError' in error_msg:
                var_name = match.group(1) if match.groups() else 'unknown'
                errors.append(f"NameError: '{var_name}' is not defined (missing import?)")
            elif 'ImportError' in error_msg:
                import_name = match.group(1) if match.groups() else 'unknown'
                errors.append(f"ImportError: {import_name}")
            elif 'SyntaxError' in error_msg:
                syntax_error = match.group(1) if match.groups() else 'unknown'
                errors.append(f"SyntaxError: {syntax_error}")

    # Also check for "ERRORS" section
    if "ERRORS" in combined_output or "ERROR collecting" in combined_output:
        # Extract all unique error messages
        error_section = re.search(r'=+\s+ERRORS\s+=+(.*?)(?=\n=+|\Z)', combined_output, re.DOTALL)
        if error_section:
            error_text = error_section.group(1)
            # Extract individual error blocks
            error_blocks = re.findall(r'ERROR collecting ([^\n]+)\n([^\n]+: [^\n]+)', error_text)
            for file_path, error_msg in error_blocks:
                errors.append(f"{file_path}: {error_msg}")

    # Remove duplicates while preserving order
    seen = set()
    unique_errors = []
    for error in errors:
        if error not in seen:
            seen.add(error)
            unique_errors.append(error)

    return unique_errors


def parse_test_statistics(pytest_output: str) -> Dict[str, int]:
    """Parse pytest output to extract test statistics.

    Returns keys used by pipeline reporting and modular runner: ``total``, ``passed``,
    ``failed``, ``skipped``, ``errors``. Legacy ``tests_*`` keys are included for
    callers that still expect them.
    """
    passed = failed = skipped = errors = total = 0

    try:
        lines = pytest_output.split("\n")

        # Final summary line (quiet or verbose), e.g. "==== 1783 passed, 30 skipped in 40s ===="
        for line in reversed(lines):
            line = line.strip()
            if " in " not in line:
                continue
            if not (
                " passed" in line
                or " failed" in line
                or " skipped" in line
                or re.search(r"\d+\s+errors?\b", line)
            ):
                continue

            pm = re.search(r"(\d+)\s+passed", line)
            if pm:
                passed = int(pm.group(1))
            fm = re.search(r"(\d+)\s+failed", line)
            if fm:
                failed = int(fm.group(1))
            sm = re.search(r"(\d+)\s+skipped", line)
            if sm:
                skipped = int(sm.group(1))
            em = re.search(r"(\d+)\s+errors?\b", line)
            if em:
                errors = int(em.group(1))

            if passed or failed or skipped or errors:
                break

        # Recovery: verbose per-node lines (avoid log noise without "::")
        if passed == 0 and failed == 0 and skipped == 0 and errors == 0:
            for line in lines:
                ls = line.strip()
                if "::" not in ls:
                    continue
                if " PASSED" in ls:
                    passed += 1
                elif " FAILED" in ls:
                    failed += 1
                elif " SKIPPED" in ls:
                    skipped += 1
                elif " ERROR" in ls:
                    errors += 1

        total = passed + failed + skipped + errors

    except Exception as e:
        logging.warning(f"Failed to parse test statistics: {e}")
        passed = failed = skipped = errors = 0
        total = 0

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "tests_run": total,
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_skipped": skipped,
        "tests_errors": errors,
    }


def parse_coverage_statistics(
    coverage_json_path: Path | str | None,
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """Parse coverage JSON file to extract coverage statistics."""
    logger = logger or logging.getLogger(__name__)
    try:
        if coverage_json_path is None:
            return {}
        if isinstance(coverage_json_path, str):
            # Accidental pytest stdout or other non-path text (fast pipeline has no JSON in stdout)
            if "\n" in coverage_json_path or len(coverage_json_path) > 4096:
                return {}
            coverage_json_path = Path(coverage_json_path)
        if not coverage_json_path.exists():
            return {"error": "Coverage file not found"}

        with open(coverage_json_path) as f:
            coverage_data = json.load(f)

        # Extract overall coverage
        total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0)

        # Extract per-file coverage
        files_coverage = {}
        for file_path, file_data in coverage_data.get("files", {}).items():
            files_coverage[file_path] = file_data.get("summary", {}).get("percent_covered", 0.0)

        return {
            "total_coverage": total_coverage,
            "files_coverage": files_coverage,
            "files_count": len(files_coverage)
        }

    except Exception as e:
        logger.warning(f"Failed to parse coverage statistics: {e}")
        return {"error": str(e)}


# Backward-compatible aliases (underscore-prefixed versions)
_extract_collection_errors = extract_collection_errors
_parse_test_statistics = parse_test_statistics
_parse_coverage_statistics = parse_coverage_statistics
