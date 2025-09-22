#!/usr/bin/env python3
"""
Step 1: Project Setup and Environment Validation with UV (Thin Orchestrator)

This step handles project initialization, UV environment setup,
dependency installation, and environment validation using modern
Python packaging standards.

How to run:
  python src/1_setup.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Environment setup results in the specified output directory
  - UV environment creation and validation
  - Dependency installation and verification
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that UV is installed and available
  - Check that src/setup/ contains setup modules
  - Check that the output directory is writable
  - Verify system requirements and permissions
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from setup import process_setup_standardized

# Create standardized pipeline script
run_script = create_standardized_pipeline_script(
    "1_setup.py",
    process_setup_standardized,
    "Project setup and environment validation",
)

def main() -> int:
    """Main entry point for the setup step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main()) 