#!/usr/bin/env python3
"""
Step 18: Security Processing (Thin Orchestrator)

This step orchestrates security processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/security/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the security module.

Pipeline Flow:
    main.py → 18_18_security.py (this script) → security/ (modular implementation)

How to run:
  python src/18_18_security.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Security processing results in the specified output directory
  - Comprehensive security reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that security dependencies are installed
  - Check that src/security/ contains security modules
  - Check that the output directory is writable
  - Verify security configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from security import process_security

# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "18_18_security.py",
    process_security,
    "Security processing"
)

def main() -> int:
    """Main entry point for the 18_security step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
