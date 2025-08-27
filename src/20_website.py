#!/usr/bin/env python3
"""
Step 20: Website Processing (Thin Orchestrator)

This step orchestrates website processing for GNN models.

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
  - Website processing results in the specified output directory
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
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning,
    create_standardized_pipeline_script,
)
from utils.argument_utils import ArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

from website import (
    process_website,
    generate_website,
    generate_html_report,
)

run_script = create_standardized_pipeline_script(
    "20_website.py",
    lambda target_dir, output_dir, logger, **kwargs: _run_website_processing(
        target_dir, output_dir, logger, **kwargs
    ),
    "Website processing for GNN pipeline documentation",
)


def _run_website_processing(target_dir: Path, output_dir: Path, logger, **kwargs) -> bool:
    """
    Standardized website processing function.

    Args:
        target_dir: Directory containing GNN files for website generation
        output_dir: Output directory for website results
        logger: Logger instance for this step
        **kwargs: Additional processing options

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info("🚀 Processing website")

        # Get configuration
        config = get_pipeline_config()
        step_config = config.get_step_config("20_website") if hasattr(config, 'get_step_config') else None

        # Set up output directory
        step_output_dir = get_output_dir_for_script("20_website.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Log processing parameters
        logger.info(f"Processing GNN files from: {target_dir}")
        logger.info(f"Output directory: {step_output_dir}")

        # Extract website-specific parameters
        website_theme = kwargs.get('website_theme', 'default')
        include_interactive = kwargs.get('include_interactive', True)

        if website_theme:
            logger.info(f"Website theme: {website_theme}")
        if include_interactive:
            logger.info("Including interactive elements")

        # Validate input directory
        if not target_dir.exists():
            log_step_error(logger, f"Input directory does not exist: {target_dir}")
            return False

        # Find GNN files
        pattern = "**/*.md" if kwargs.get('recursive', False) else "*.md"
        gnn_files = list(target_dir.glob(pattern))

        if not gnn_files:
            log_step_warning(logger, f"No GNN files found in {target_dir}")
            return True  # Not an error, just no files to process

        logger.info(f"Found {len(gnn_files)} GNN files for website generation")

        # Process website via module API
        logger.info("Website module available, processing files...")
        return process_website(target_dir=target_dir, output_dir=step_output_dir, **kwargs)

    except Exception as e:
        log_step_error(logger, f"Website processing failed: {e}")
        return False


def main() -> int:
    """Main entry point for the website step."""
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
