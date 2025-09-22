#!/usr/bin/env python3
"""
Step 5: Type Checking and Validation (Thin Orchestrator)

This step performs type checking and validation on GNN files.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/type_checker/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the type_checker module.

Pipeline Flow:
    main.py → 5_type_checker.py (this script) → type_checker/ (modular implementation)

How to run:
  python src/5_type_checker.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Type checking results in the specified output directory
  - Syntax validation reports and type consistency analysis
  - Resource estimation and performance metrics
  - POMDP validation results if POMDP mode enabled
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that type checker dependencies are installed
  - Check that src/type_checker/ contains type checking modules
  - Check that the output directory is writable
  - Verify type checking configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from type_checker import process_type_checking_standardized

# Create standardized pipeline script
run_script = create_standardized_pipeline_script(
    "5_type_checker.py",
    process_type_checking_standardized,
    "Type checking and validation of GNN files",
    additional_arguments={
        "strict": {"type": bool, "help": "Enable strict validation mode"},
        "estimate_resources": {"type": bool, "help": "Estimate computational resources"},
        "pomdp_mode": {"type": bool, "help": "Enable POMDP validation mode"}
    }
)

def main() -> int:
    """Main entry point for the type checking step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
