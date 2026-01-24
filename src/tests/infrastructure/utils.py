"""
Test Utility Functions for GNN Processing Pipeline.

This module provides utility functions for pytest command building,
output parsing, and dependency checking.
"""

import logging
import re
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

from utils.test_utils import TEST_DIR


def check_test_dependencies(logger: logging.Logger) -> Dict[str, Any]:
    """
    Check if required test dependencies are available.
    
    Verifies that pytest and optional dependencies (pytest-cov, pytest-timeout)
    are installed and available.
    
    Args:
        logger: Logger instance for reporting
    
    Returns:
        Dictionary with dependency status:
        {
            'pytest': bool,
            'pytest_cov': bool,
            'pytest_timeout': bool,
            'all_required': bool
        }
    """
    dependencies = {
        "pytest": False,
        "pytest-cov": False,
        "pytest-xdist": False,
        "psutil": False,
        "coverage": False
    }
    
    try:
        import pytest
        dependencies["pytest"] = True
    except ImportError:
        pass
    
    try:
        import pytest_cov
        dependencies["pytest-cov"] = True
    except ImportError:
        pass
    
    try:
        import xdist
        dependencies["pytest-xdist"] = True
    except ImportError:
        pass
    
    try:
        import psutil
        dependencies["psutil"] = True
    except ImportError:
        pass

    try:
        import coverage
        dependencies["coverage"] = True
    except ImportError:
        pass
    
    # Log results
    missing_deps = [name for name, available in dependencies.items() if not available]
    if missing_deps:
        logger.warning(f"⚠️ Missing test dependencies: {missing_deps}")
    else:
        logger.info("✅ All test dependencies available")
            
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
    """Parse pytest output to extract test statistics."""
    stats = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0
    }

    try:
        lines = pytest_output.split('\n')

        # Look for the summary line at the end (e.g., "534 passed, 12 skipped in 3.45s")
        for line in reversed(lines):
            line = line.strip()
            # Check if this line contains test results (has passed/failed/skipped + " in ")
            if (" passed" in line or " failed" in line or " skipped" in line) and " in " in line:
                # Parse patterns like: "534 passed, 12 skipped in 3.45s"
                # or "22 passed, 5 failed, 3 skipped in 1.23s"
                
                # Extract passed count
                passed_match = re.search(r'(\d+)\s+passed', line)
                if passed_match:
                    stats["tests_passed"] = int(passed_match.group(1))
                    stats["tests_run"] += int(passed_match.group(1))
                
                # Extract failed count
                failed_match = re.search(r'(\d+)\s+failed', line)
                if failed_match:
                    stats["tests_failed"] = int(failed_match.group(1))
                    stats["tests_run"] += int(failed_match.group(1))
                
                # Extract skipped count
                skipped_match = re.search(r'(\d+)\s+skipped', line)
                if skipped_match:
                    stats["tests_skipped"] = int(skipped_match.group(1))
                    stats["tests_run"] += int(skipped_match.group(1))
                
                # If we found any stats, break
                if stats["tests_run"] > 0:
                    break

        # Fallback: count individual test results if summary line not found
        # Look for lines like "test_foo.py::test_bar PASSED [  1%]"
        if stats["tests_passed"] == 0 and stats["tests_failed"] == 0:
            for line in lines:
                line_stripped = line.strip()
                # Match pytest verbose output format: "test_file.py::TestClass::test_method PASSED"
                if " PASSED" in line_stripped:
                    stats["tests_passed"] += 1
                elif " FAILED" in line_stripped:
                    stats["tests_failed"] += 1
                elif " SKIPPED" in line_stripped:
                    stats["tests_skipped"] += 1
            
            # Update tests_run from counted results
            if stats["tests_passed"] > 0 or stats["tests_failed"] > 0 or stats["tests_skipped"] > 0:
                stats["tests_run"] = stats["tests_passed"] + stats["tests_failed"] + stats["tests_skipped"]

        # Also look for collected items line if no results found yet
        if stats["tests_run"] == 0:
            for line in lines:
                if "collected" in line and ("item" in line or "test" in line):
                    # Extract number before "collected"
                    match = re.search(r'(\d+)\s+(?:item|test)', line)
                    if match:
                        stats["tests_run"] = int(match.group(1))
                        break

    except Exception as e:
        logging.warning(f"Failed to parse test statistics: {e}")

    return stats


def parse_coverage_statistics(coverage_json_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Parse coverage JSON file to extract coverage statistics."""
    try:
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
