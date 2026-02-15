#!/usr/bin/env python3
"""
Step 16: Analysis Processing (Thin Orchestrator)

This step orchestrates analysis processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/analysis/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the analysis module.

Pipeline Flow:
    main.py → 16_analysis.py (this script) → analysis/ (modular implementation)

How to run:
  python src/16_analysis.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Analysis processing results in the specified output directory
  - Comprehensive analysis reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that analysis dependencies are installed
  - Check that src/analysis/ contains analysis modules
  - Check that the output directory is writable
  - Verify analysis configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from analysis import process_analysis
except ImportError as e:
    def process_analysis(target_dir, output_dir, logger=None, **kwargs):
        """Fallback analysis processing when module unavailable."""
        import logging
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.warning(f"Analysis module not available - using fallback: {e}")
        logger.info("Install analysis support with: uv pip install -e .[analysis]")
        return True

run_script = create_standardized_pipeline_script(
    "16_analysis.py",
    process_analysis,
    "Analysis processing for GNN models",
)

def main() -> int:
    """Main entry point for the analysis step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())