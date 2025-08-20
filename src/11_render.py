#!/usr/bin/env python3
"""
Step 11: Render Processing (Thin Orchestrator)

This step orchestrates render processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/render/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the render module.

Pipeline Flow:
    main.py → 11_11_render.py (this script) → render/ (modular implementation)

How to run:
  python src/11_11_render.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Render processing results in the specified output directory
  - Comprehensive render reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that render dependencies are installed
  - Check that src/render/ contains render modules
  - Check that the output directory is writable
  - Verify render configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from render import process_render

# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "11_render.py",
    process_render,
    "Render processing"
)

def main() -> int:
    """Main entry point for the 11_render step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
