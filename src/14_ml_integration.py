#!/usr/bin/env python3
"""
Step 14: Ml Integration Processing (Thin Orchestrator)

This step orchestrates ml_integration processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/ml_integration/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the ml_integration module.

Pipeline Flow:
    main.py → 14_14_ml_integration.py (this script) → ml_integration/ (modular implementation)

How to run:
  python src/14_14_ml_integration.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Ml_Integration processing results in the specified output directory
  - Comprehensive ml_integration reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that ml_integration dependencies are installed
  - Check that src/ml_integration/ contains ml_integration modules
  - Check that the output directory is writable
  - Verify ml_integration configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from ml_integration import process_ml_integration

# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "14_14_ml_integration.py",
    process_ml_integration,
    "Ml_Integration processing"
)

def main() -> int:
    """Main entry point for the 14_ml_integration step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
