#!/usr/bin/env python3
"""
Step 2: Test Suite Execution (Thin Orchestrator)

This script orchestrates comprehensive tests for the GNN pipeline in staged execution.
It is a thin orchestrator that delegates core functionality to the tests module.

How to run:
  python src/2_tests.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - All test logs and reports in the specified output directory (default: output/)
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that pytest and required plugins are installed (pip install pytest pytest-cov pytest-xdist pytest-json-report)
  - Check that src/tests/ contains test files
  - Check that the output directory is writable
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import test processing function with fallback
try:
    from tests.runner import run_tests
    TEST_MODULE_AVAILABLE = True
except ImportError as e:
    TEST_MODULE_AVAILABLE = False
    logging.warning(f"Test module not available: {e}")

    def run_tests(target_dir, output_dir, logger, **kwargs):
        """Fallback test processing when module unavailable."""
        logger.warning("⚠️ Test module not available, using fallback")
        return True

# Create standardized pipeline script
run_script = create_standardized_pipeline_script(
    "2_tests.py",
    run_tests,
    "Comprehensive test suite execution",
    additional_arguments={
        "fast_only": {"type": bool, "help": "Run only fast tests"},
        "include_slow": {"type": bool, "help": "Include slow test categories"},
        "include_performance": {"type": bool, "help": "Include performance test categories"},
        "comprehensive": {"type": bool, "help": "Run all test categories including comprehensive suite"}
    }
)

def main() -> int:
    """Main entry point for the test step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main()) 