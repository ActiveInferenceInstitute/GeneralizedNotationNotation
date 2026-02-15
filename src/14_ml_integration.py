#!/usr/bin/env python3
"""
Step 14: ML Integration Processing (Thin Orchestrator)

This step orchestrates ML integration processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/ml_integration/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the ml_integration module.

Pipeline Flow:
    main.py → 14_ml_integration.py (this script) → ml_integration/ (modular implementation)

How to run:
  python src/14_ml_integration.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - ML integration processing results in the specified output directory
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

# Import module function
try:
    from ml_integration import process_ml_integration
except ImportError:
    def process_ml_integration(target_dir, output_dir, logger=None, **kwargs):
        """Fallback ml_integration processing when module unavailable."""
        import logging
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.warning("ML integration module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "14_ml_integration.py",
    process_ml_integration,
    "ML integration processing for GNN models",
)

def main() -> int:
    """Main entry point for the ml_integration step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
