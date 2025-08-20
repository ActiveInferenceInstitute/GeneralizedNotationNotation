#!/usr/bin/env python3
"""
Step 22: Gui Processing (Thin Orchestrator)

This step orchestrates gui processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/gui/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the gui module.

Pipeline Flow:
    main.py → 22_22_gui.py (this script) → gui/ (modular implementation)

How to run:
  python src/22_22_gui.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Gui processing results in the specified output directory
  - Comprehensive gui reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that gui dependencies are installed
  - Check that src/gui/ contains gui modules
  - Check that the output directory is writable
  - Verify gui configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from gui import process_gui

# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "22_22_gui.py",
    process_gui,
    "Gui processing"
)

def main() -> int:
    """Main entry point for the 22_gui step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
