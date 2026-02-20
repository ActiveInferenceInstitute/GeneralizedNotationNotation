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

from pipeline.config import get_output_dir_for_script
from utils.pipeline_template import create_standardized_pipeline_script

def _test_runner_wrapper(target_dir, output_dir, logger, **kwargs):
    """Wrapper to map standard pipeline args to run_tests."""
    import os
    
    # Log environment overrides
    if os.getenv("SKIP_TESTS_IN_PIPELINE"):
        logger.info("⏭️ SKIP_TESTS_IN_PIPELINE set - tests will be skipped")
        return True
    if os.getenv("FAST_TESTS_TIMEOUT"):
        logger.info(f"⏱️ Custom timeout: {os.getenv('FAST_TESTS_TIMEOUT')} seconds")
        
    comprehensive = kwargs.get('comprehensive', False)
    fast_only = not comprehensive and kwargs.get('fast_only', True)
    test_mode = "comprehensive" if comprehensive else "fast"
    verbose = kwargs.get('verbose', False)
    
    logger.info(f"🧪 Running {test_mode} test suite")
    logger.info(f"📍 Output directory: {output_dir}")
    
    try:
        from tests import run_tests
        success = run_tests(
            logger=logger,
            output_dir=output_dir,
            verbose=verbose,
            fast_only=fast_only,
            comprehensive=comprehensive,
            generate_coverage=False,
            auto_fallback=True
        )
        
        if not success:
            logger.error("❌ Test execution failed")
            logger.error("💡 Check that test files exist and follow pytest naming conventions (test_*.py)")
            
        return success
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

additional_args = {
    "fast-only": {"action": "store_true", "default": True, "help": "Run only fast tests (default)", "flag": "--fast-only"},
    "comprehensive": {"action": "store_true", "help": "Run all tests (overrides fast-only)", "flag": "--comprehensive"}
}

run_script = create_standardized_pipeline_script(
    "2_tests.py",
    _test_runner_wrapper,
    "Run GNN Pipeline Tests",
    additional_arguments=additional_args
)

def main() -> int:
    """Main entry point for the tests step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
