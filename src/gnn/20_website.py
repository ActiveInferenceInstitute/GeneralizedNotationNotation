#!/usr/bin/env python3
"""
Step 20: Website Generation (Thin Orchestrator)

This step orchestrates website generation for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/gnn/website/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the website module.

Pipeline Flow:
    main.py → 20_website.py (this script) → website/ (modular implementation)

How to run:
  uv run python src/gnn/20_website.py --target-dir input/gnn_files --output-dir output --verbose
  uv run python src/gnn/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Website generation results in the specified output directory
  - Comprehensive website reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that website dependencies are installed
  - Check that src/gnn/website/ contains website modules
  - Check that the output directory is writable
  - Verify website configuration and requirements
"""

from typing import cast

from gnn.utils.pipeline_orchestration.pipeline_template import (
    create_standardized_pipeline_script,
)

# Hard import: website is a core module and must always be available.
from gnn.website import process_website

run_script = create_standardized_pipeline_script(
    "20_website.py",
    process_website,
    "Website generation for GNN models",
)


def main() -> int:
    """Main entry point for the website step."""
    return cast("int", run_script())


if __name__ == "__main__":
    raise SystemExit(main())
