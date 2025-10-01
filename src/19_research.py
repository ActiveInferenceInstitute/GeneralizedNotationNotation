#!/usr/bin/env python3
"""
Step 19: Research Processing (Thin Orchestrator)

This step orchestrates research processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/research/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the research module.

Pipeline Flow:
    main.py → 19_research.py (this script) → research/ (modular implementation)

How to run:
  python src/19_research.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Research processing results in the specified output directory
  - Comprehensive research reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that research dependencies are installed
  - Check that src/research/ contains research modules
  - Check that the output directory is writable
  - Verify research configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from research import process_research
except ImportError:
    def process_research(target_dir, output_dir, **kwargs):
        """Fallback research processing when module unavailable."""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Research module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "19_research.py",
    process_research,
    "Research processing for GNN models",
)

def main() -> int:
    """Main entry point for the research step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())