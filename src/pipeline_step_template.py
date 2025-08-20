#!/usr/bin/env python3
"""
GNN Pipeline Step Template (Thin Orchestrator)

This template provides the standardized pattern for all GNN pipeline steps (0-23).
It implements the thin orchestrator pattern that delegates core functionality to
modular implementations while handling argument parsing, logging, and orchestration.

How to run:
  python src/N_step_name.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - [Module-specific outputs]
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module. It handles argument parsing, logging setup, output
    directory management, and calls the actual processing functions from the module.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from [module_name] import [main_function]

# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "[N]_step_name.py",
    [main_function],
    "[Brief step description]"
)

def main() -> int:
    """Main entry point for the pipeline step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
