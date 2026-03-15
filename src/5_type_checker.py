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
  - Comprehensive validation reports and analysis
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that type checker dependencies are installed
  - Check that src/type_checker/ contains type checker modules
  - Check that the output directory is writable
  - Verify type checker configuration and requirements
"""

import sys
import logging
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Hard import: type_checker is a core module and must always be available.
# ImportError here means the module is broken or missing — fail loudly.
from type_checker import GNNTypeChecker

def _type_check_dispatch(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs: Any) -> bool:
    """Dispatch to GNNTypeChecker."""
    return GNNTypeChecker().validate_gnn_files(target_dir, output_dir, **kwargs)

run_script = create_standardized_pipeline_script(
    "5_type_checker.py",
    _type_check_dispatch,
    "Type checking and validation of GNN files",
    additional_arguments={
        "strict": {"type": bool, "help": "Enable strict validation mode"},
        "estimate_resources": {"type": bool, "help": "Enable resource estimation"}
    }
)

def main() -> int:
    """Main entry point for the type checker step."""
    return run_script()

if __name__ == "__main__":
    raise SystemExit(main())
