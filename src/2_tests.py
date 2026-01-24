#!/usr/bin/env python3
"""
Step 2: Test Suite Execution (Thin Orchestrator)

Orchestrates comprehensive tests for the GNN pipeline. Follows thin orchestrator pattern,
delegating all core functionality to the tests module (tests.run_tests() from runner.py).

Usage:
  python src/2_tests.py --fast-only --verbose      # Fast tests (default)
  python src/2_tests.py --comprehensive --verbose  # All tests

Environment Variables:
  SKIP_TESTS_IN_PIPELINE  Set to skip tests entirely
  FAST_TESTS_TIMEOUT      Override timeout (default: 600 seconds)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import get_output_dir_for_script
from utils.logging_utils import log_step_error
from utils.pipeline_template import setup_step_logging
from utils.visual_logging import create_visual_logger, VisualConfig


def main() -> int:
    """Main entry point for the tests step."""
    import argparse
    import os

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run GNN Pipeline Tests")
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--fast-only', action='store_true', default=True, help='Run only fast tests (default)')
    parser.add_argument('--comprehensive', action='store_true', help='Run all tests (overrides fast-only)')
    parser.add_argument('--target-dir', type=str, help='Target directory (unused for tests)')
    args = parser.parse_args()

    # Setup logging
    visual_config = VisualConfig(
        enable_colors=True, enable_progress_bars=True, enable_emoji=True,
        enable_animation=True, show_timestamps=args.verbose,
        show_correlation_ids=True, compact_mode=False
    )
    visual_logger = create_visual_logger("2_tests.py", visual_config)
    logger = setup_step_logging("2_tests.py", args.verbose)

    # Create output directory
    try:
        output_dir = Path(args.output_dir)
        step_output_dir = get_output_dir_for_script("2_tests", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return 1

    # Determine test mode
    comprehensive = args.comprehensive
    fast_only = not comprehensive and args.fast_only
    test_mode = "comprehensive" if comprehensive else "fast"

    # Log environment overrides
    if os.getenv("SKIP_TESTS_IN_PIPELINE"):
        logger.info("⏭️ SKIP_TESTS_IN_PIPELINE set - tests will be skipped")
    if os.getenv("FAST_TESTS_TIMEOUT"):
        logger.info(f"⏱️ Custom timeout: {os.getenv('FAST_TESTS_TIMEOUT')} seconds")

    # Run tests via runner module
    try:
        from tests import run_tests

        visual_logger.print_header(
            "🧪 GNN Pipeline Test Suite",
            f"Running {test_mode} test mode | Output: {step_output_dir}"
        )
        logger.info(f"🧪 Running {test_mode} test suite")
        logger.info(f"📍 Output directory: {step_output_dir}")

        success = run_tests(
            logger=logger,
            output_dir=step_output_dir,
            verbose=args.verbose,
            fast_only=fast_only,
            comprehensive=comprehensive,
            generate_coverage=False,
            auto_fallback=True  # Runner handles fallback to comprehensive if needed
        )

        if not success:
            logger.error("❌ Test execution failed")
            logger.error("💡 Check that test files exist and follow pytest naming conventions (test_*.py)")

        return 0 if success else 1

    except Exception as e:
        log_step_error(logger, f"Test execution failed: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
