#!/usr/bin/env python3
"""
Step 6: Validation Processing (Thin Orchestrator)

This step performs validation and quality assurance on GNN models,
including semantic validation, performance profiling, and consistency checking.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/validation/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the validation module.

Pipeline Flow:
    main.py → 6_validation.py (this script) → validation/ (modular implementation)

How to run:
  python src/6_validation.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Validation results in the specified output directory
  - Semantic validation reports and scores
  - Performance profiling and resource estimates
  - Consistency checking and quality metrics
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that validation dependencies are installed
  - Check that src/validation/ contains validation modules
  - Check that the output directory is writable
  - Verify validation configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import standardized pipeline helper and module function
from utils.pipeline_template import create_standardized_pipeline_script
from validation import process_validation

# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "6_validation",
    process_validation,
    "Validation and quality assurance processing"
)

def main() -> int:
    """Main entry point for the validation step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main()) 