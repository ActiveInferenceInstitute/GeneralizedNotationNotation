#!/usr/bin/env python3
"""
Step 8: Visualization Processing (Thin Orchestrator)

This step handles visualization processing for GNN files with comprehensive
safe-to-fail patterns and robust output management.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/visualization/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the visualization module.

Pipeline Flow:
    main.py → 8_visualization.py (this script) → visualization/ (modular implementation)

How to run:
  python src/8_visualization.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Visualization results in the specified output directory
  - Matrix visualizations, network graphs, and combined analysis plots
  - Comprehensive visualization reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that visualization dependencies are installed
  - Check that src/visualization/ contains visualization modules
  - Check that the output directory is writable
  - Verify visualization configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from visualization import process_visualization_main

# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "8_visualization.py",
    process_visualization_main,
    "Matrix and network visualization processing"
)

def main() -> int:
    """Main entry point for the visualization step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
