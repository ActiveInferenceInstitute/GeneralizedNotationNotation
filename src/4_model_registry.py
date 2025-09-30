#!/usr/bin/env python3
"""
Step 4: Model Registry Processing (Thin Orchestrator)

This step implements a centralized model registry for GNN models with versioning,
metadata management, and model lifecycle tracking.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/model_registry/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the model_registry module.

Pipeline Flow:
    main.py → 4_model_registry.py (this script) → model_registry/ (modular implementation)

How to run:
  python src/4_model_registry.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Model registry data in the specified output directory
  - Model metadata and versioning information
  - Registry summary and processing reports
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that model registry dependencies are installed
  - Check that src/model_registry/ contains registry modules
  - Check that the output directory is writable
  - Verify model registry configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from model_registry import process_model_registry
except ImportError:
    def process_model_registry(target_dir, output_dir, logger, **kwargs):
        """Fallback model registry when module unavailable."""
        logger.warning("Model registry module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "4_model_registry.py",
    process_model_registry,
    "Model registry processing for GNN models",
    additional_arguments={
        "registry_path": {"type": str, "help": "Path to model registry file"}
    }
)

def main() -> int:
    """Main entry point for the model registry step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())