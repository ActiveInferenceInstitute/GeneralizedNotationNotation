#!/usr/bin/env python3
"""
Step 21: MCP Processing (Thin Orchestrator)

This step orchestrates Model Context Protocol processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/mcp/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the mcp module.

Pipeline Flow:
    main.py → 21_mcp.py (this script) → mcp/ (modular implementation)

How to run:
  python src/21_mcp.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - MCP processing results in the specified output directory
  - Comprehensive MCP reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that mcp dependencies are installed
  - Check that src/mcp/ contains mcp modules
  - Check that the output directory is writable
  - Verify mcp configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from mcp import process_mcp
except ImportError:
    def process_mcp(target_dir, output_dir, logger=None, **kwargs):
        """Fallback mcp processing when module unavailable."""
        import logging
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.warning("MCP module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "21_mcp.py",
    process_mcp,
    "Model Context Protocol processing for GNN models",
)

def main() -> int:
    """Main entry point for the mcp step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())