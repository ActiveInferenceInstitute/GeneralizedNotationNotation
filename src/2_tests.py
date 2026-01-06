#!/usr/bin/env python3
"""
Step 2: Test Suite Execution (Thin Orchestrator)

This script orchestrates comprehensive tests for the GNN pipeline in staged execution.
It follows the thin orchestrator pattern, delegating all core functionality to the
tests module (specifically tests.run_tests() from runner.py).

Architecture:
  This script acts as a thin wrapper that:
  1. Parses command-line arguments
  2. Sets up logging and visual output
  3. Creates output directories
  4. Delegates to tests.run_tests() for actual test execution
  5. Returns standardized exit codes (0=success, 1=failure)

  All test execution logic is in src/tests/runner.py, which provides:
  - run_fast_pipeline_tests() - Fast tests for quick validation (default)
  - run_comprehensive_tests() - All tests including slow/performance
  - run_fast_reliable_tests() - Essential tests fallback

Usage:
  # Fast tests (default - used in pipeline)
  python src/2_tests.py --fast-only --verbose
  
  # Comprehensive test suite (all tests)
  python src/2_tests.py --comprehensive --verbose
  
  # As part of full pipeline (runs fast tests by default)
  python src/main.py

Command-Line Arguments:
  --output-dir DIR     Output directory for test results (default: 'output')
  --verbose            Enable verbose output with detailed logging
  --fast-only          Run only fast tests (default: True)
  --comprehensive      Run comprehensive test suite (overrides fast-only)
  --target-dir DIR     Target directory (unused for tests, kept for API compatibility)

Test Mode Selection:
  The script determines test mode based on flags:
  - If --comprehensive: Runs all tests via run_comprehensive_tests()
  - If --fast-only (default): Runs fast tests via run_fast_pipeline_tests()
  - Otherwise: Falls back to run_fast_reliable_tests()

Environment Variables:
  SKIP_TESTS_IN_PIPELINE  Set to any value to skip tests entirely
  FAST_TESTS_TIMEOUT      Override timeout for fast tests (default: 600 seconds)

Expected Outputs:
  - Test execution reports in output/2_tests_output/
  - pytest_comprehensive_output.txt - Full pytest output
  - test_execution_report.json - Structured test results
  - Clear pass/fail status for each test
  - Performance metrics and timing data

Error Handling:
  - Returns exit code 0 on success
  - Returns exit code 1 on failure (with detailed error messages)
  - Detects and reports collection errors (import/syntax errors)
  - Provides actionable suggestions for common issues

Troubleshooting:
  - Check that pytest is installed: pip install pytest pytest-cov
  - Check that src/tests/ contains test files
  - Check that the output directory is writable
  - Review output/2_tests_output/pytest_comprehensive_output.txt for details
  - Set SKIP_TESTS_IN_PIPELINE=1 to skip tests in pipeline runs
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import get_output_dir_for_script
from utils.logging_utils import log_step_error
from utils.pipeline_template import setup_step_logging
from utils.visual_logging import create_visual_logger, VisualConfig, format_step_header, format_status_message

def main() -> int:
    """
    Main entry point for the tests step.
    
    This function:
    1. Parses command-line arguments
    2. Sets up enhanced visual logging
    3. Creates output directory
    4. Determines test mode (fast/comprehensive/reliable)
    5. Calls tests.run_tests() with appropriate parameters
    6. Returns exit code based on test results
    
    Returns:
        0: Tests passed successfully
        1: Tests failed or execution error occurred
    """
    import argparse

    # Simple argument parser
    parser = argparse.ArgumentParser(description="Run GNN Pipeline Tests")
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--fast-only', action='store_true', default=True, help='Run only fast tests (default)')
    parser.add_argument('--comprehensive', action='store_true', help='Run all tests (overrides fast-only)')
    parser.add_argument('--target-dir', type=str, help='Target directory (unused for tests)')
    args = parser.parse_args()

    # Setup enhanced visual logging
    visual_config = VisualConfig(
        enable_colors=True,
        enable_progress_bars=True,
        enable_emoji=True,
        enable_animation=True,
        show_timestamps=args.verbose,
        show_correlation_ids=True,
        compact_mode=False
    )

    visual_logger = create_visual_logger("2_tests.py", visual_config)

    # Setup logging
    logger = setup_step_logging("2_tests.py", args.verbose)

    # Get output directory
    try:
        output_dir = Path(args.output_dir)
        step_output_dir = get_output_dir_for_script("2_tests", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return 1

    # Import and run tests
    try:
        from tests import run_tests

        # Determine test mode with visual indicators
        if args.comprehensive:
            comprehensive = True
            fast_only = False
            test_mode = "comprehensive"
        elif args.fast_only:
            comprehensive = False
            fast_only = True
            test_mode = "fast"
        else:
            # Default for pipeline: run fast tests only (comprehensive is too slow)
            # Use --comprehensive flag for full test suite
            comprehensive = False
            fast_only = True
            test_mode = "fast (default)"

        # Enhanced test mode announcement
        visual_logger.print_header(
            "🧪 GNN Pipeline Test Suite",
            f"Running {test_mode} test mode | Output: {step_output_dir}"
        )

        logger.info(f"🧪 Running {test_mode} test suite")
        logger.info(f"📍 Output directory: {step_output_dir}")
        
        # Check for environment overrides
        import os
        if os.getenv("SKIP_TESTS_IN_PIPELINE"):
            logger.info("⏭️  Environment variable SKIP_TESTS_IN_PIPELINE is set - tests will be skipped")
        if os.getenv("FAST_TESTS_TIMEOUT"):
            logger.info(f"⏱️  Custom timeout set: {os.getenv('FAST_TESTS_TIMEOUT')} seconds")

        # Run tests
        success = run_tests(
            logger=logger,
            output_dir=step_output_dir,
            verbose=args.verbose,
            fast_only=fast_only,
            comprehensive=comprehensive,
            generate_coverage=False  # Disable coverage for speed
        )

        # If no tests were collected, that's a failure
        if not success:
            # Check if it's due to zero tests collected
            summary_file = step_output_dir / "test_execution_report.json"
            if summary_file.exists():
                import json
                summary = json.loads(summary_file.read_text())
                tests_run = summary.get("execution_summary", {}).get("tests_run", 0)
                if tests_run == 0:
                    logger.error("❌ Test execution failed: No tests were collected")
                    logger.error("💡 Check that test files exist and follow pytest naming conventions (test_*.py)")
                    logger.error("💡 Ensure test functions are named with 'test_' prefix")
                    return 1

        return 0 if success else 1

    except Exception as e:
        log_step_error(logger, f"Test execution failed: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

