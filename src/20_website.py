#!/usr/bin/env python3
"""
Step 20: Website Generation (Thin Orchestrator)

This step orchestrates website generation for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/website/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the website module.

Pipeline Flow:
    main.py → 20_website.py (this script) → website/ (modular implementation)

How to run:
  python src/20_website.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Website generation results in the specified output directory
  - Comprehensive website reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that website dependencies are installed
  - Check that src/website/ contains website modules
  - Check that the output directory is writable
  - Verify website configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script

# Import module function
try:
    from website import process_website
except ImportError:
    def process_website(target_dir, output_dir, logger=None, **kwargs):
        """Fallback website processing when module unavailable."""
        import logging
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.warning("Website module not available - using fallback")
        return True

run_script = create_standardized_pipeline_script(
    "20_website.py",
    process_website,
    "Website generation for GNN models",
    additional_arguments={
        "website_html_filename": {"type": str, "default": "gnn_pipeline_summary_website.html", "help": "Output HTML filename"}
    }
)

def main() -> int:
    """Main entry point for the website step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())