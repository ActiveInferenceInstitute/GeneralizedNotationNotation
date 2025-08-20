#!/usr/bin/env python3
"""
Step 23: Report Processing (Thin Orchestrator)

This step orchestrates report processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/report/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the report module.

Pipeline Flow:
    main.py → 23_23_report.py (this script) → report/ (modular implementation)

How to run:
  python src/23_23_report.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Report processing results in the specified output directory
  - Comprehensive report reports and summaries
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
from report import process_report

# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "23_report.py",
    process_report,
    "Report processing"
)

def main() -> int:
    """Main entry point for the 23_report step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
