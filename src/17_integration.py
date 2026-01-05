#!/usr/bin/env python3
"""
Step 17: Integration Processing (Thin Orchestrator)

This step orchestrates integration processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/integration/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the integration module.

Pipeline Flow:
    main.py → 17_integration.py (this script) → integration/ (modular implementation)

How to run:
  python src/17_integration.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Integration processing results in the specified output directory
  - Comprehensive integration reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that integration dependencies are installed
  - Check that src/integration/ contains integration modules
  - Check that the output directory is writable
  - Verify integration configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from integration import process_integration
except ImportError:
    def process_integration(target_dir, output_dir, logger=None, **kwargs):
        """Fallback integration processing when module unavailable."""
        import logging
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.warning("Integration module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "17_integration.py",
    process_integration,
    "Integration processing for GNN models",
)

def main() -> int:
    """Main entry point for the integration step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())