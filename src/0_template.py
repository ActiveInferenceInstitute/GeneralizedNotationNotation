#!/usr/bin/env python3
"""
Step 0: Pipeline Template with Infrastructure Demonstration (Thin Orchestrator)

This step demonstrates the thin orchestrator pattern in the GNN pipeline architecture.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/template/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the template module.

Pipeline Flow:
    main.py → 0_template.py (this script) → template/ (modular implementation)

How to run:
  python src/0_template.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Template processing results in the specified output directory
  - Infrastructure demonstration and pattern validation
  - Comprehensive error handling and recovery demonstrations
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that template dependencies are installed
  - Check that src/template/ contains template modules
  - Check that the output directory is writable
  - Verify template configuration and pattern setup
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from template import process_template_standardized
except ImportError:
    def process_template_standardized(target_dir, output_dir, logger, **kwargs):
        """Fallback template processing when module unavailable."""
        logger.warning("Template module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "0_template.py",
    process_template_standardized,
    "Pipeline template and infrastructure demonstration",
    additional_arguments={
        "simulate_error": {"type": bool, "help": "Simulate an error for testing"}
    }
)

def main() -> int:
    """Main entry point for the template step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main()) 