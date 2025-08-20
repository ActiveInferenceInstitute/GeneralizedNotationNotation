#!/usr/bin/env python3
"""
Step 12: Execute Processing (Thin Orchestrator)

This step orchestrates execute processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/execute/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the execute module.

Pipeline Flow:
    main.py → 12_execute.py (this script) → execute/ (modular implementation)

How to run:
  python src/12_execute.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Execute processing results in the specified output directory
  - Comprehensive execute reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that execute dependencies are installed
  - Check that src/execute/ contains execute modules
  - Check that the output directory is writable
  - Verify execute configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from execute import process_execute

# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "12_execute.py",
    process_execute,
    "Execute processing"
)

def main() -> int:
    """Main entry point for the 12_execute step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
