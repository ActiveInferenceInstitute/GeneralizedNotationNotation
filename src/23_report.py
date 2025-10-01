#!/usr/bin/env python3
"""
Step 23: Report Generation (Thin Orchestrator)

This step orchestrates report generation for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/report/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the report module.

Pipeline Flow:
    main.py → 23_report.py (this script) → report/ (modular implementation)

How to run:
  python src/23_report.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Report generation results in the specified output directory
  - Comprehensive report summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that report dependencies are installed
  - Check that src/report/ contains report modules
  - Check that the output directory is writable
  - Verify report configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from report import process_report
except ImportError:
    def process_report(target_dir, output_dir, **kwargs):
        """Fallback report processing when module unavailable."""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Report module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "23_report.py",
    process_report,
    "Report generation for GNN models",
)

def main() -> int:
    """Main entry point for the report step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())