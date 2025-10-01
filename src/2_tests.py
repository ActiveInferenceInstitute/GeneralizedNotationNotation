#!/usr/bin/env python3
"""
Step 2: Test Suite Execution (Thin Orchestrator)

This script orchestrates comprehensive tests for the GNN pipeline in staged execution.
It is a thin orchestrator that delegates core functionality to the test runner.

How to run:
  python src/2_tests.py --fast-only --verbose  # Fast tests (default)
  python src/2_tests.py --comprehensive         # All tests
  python src/main.py  # (runs as part of the pipeline with fast tests)

Expected outputs:
  - All test logs and reports in output/2_tests_output/
  - Clear pass/fail status for each test
  - Performance metrics and timing data

If you encounter errors:
  - Check that pytest is installed: pip install pytest pytest-cov
  - Check that src/tests/ contains test files
  - Check that the output directory is writable
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import get_output_dir_for_script
from utils.logging_utils import log_step_error
from utils.pipeline_template import setup_step_logging

def main() -> int:
    """Main entry point for the tests step."""
    import argparse

    # Simple argument parser
    parser = argparse.ArgumentParser(description="Run GNN Pipeline Tests")
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--fast-only', action='store_true', default=True, help='Run only fast tests (default)')
    parser.add_argument('--comprehensive', action='store_true', help='Run all tests (overrides fast-only)')
    parser.add_argument('--target-dir', type=str, help='Target directory (unused for tests)')
    args = parser.parse_args()

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

        # Determine test mode
        fast_only = args.fast_only and not args.comprehensive
        comprehensive = args.comprehensive

        logger.info(f"🧪 Running {'comprehensive' if comprehensive else 'fast'} test suite")

        # Run tests
        success = run_tests(
            logger=logger,
            output_dir=step_output_dir,
            verbose=args.verbose,
            fast_only=fast_only,
            comprehensive=comprehensive,
            generate_coverage=False  # Disable coverage for speed
        )

        return 0 if success else 1

    except Exception as e:
        log_step_error(logger, f"Test execution failed: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

