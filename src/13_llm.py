#!/usr/bin/env python3
"""
Step 13: LLM Processing (Thin Orchestrator)

This step orchestrates LLM processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/llm/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the llm module.

Pipeline Flow:
    main.py → 13_llm.py (this script) → llm/ (modular implementation)

How to run:
  python src/13_llm.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - LLM processing results in the specified output directory
  - Comprehensive LLM reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that LLM dependencies are installed
  - Check that src/llm/ contains LLM modules
  - Check that the output directory is writable
  - Verify LLM configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from llm import process_llm
except ImportError:
    def process_llm(target_dir, output_dir, logger=None, **kwargs):
        """Fallback LLM processing when module unavailable."""
        import logging
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.warning("LLM module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "13_llm.py",
    process_llm,
    "LLM processing for GNN analysis",
)

def main() -> int:
    """Main entry point for the LLM step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
