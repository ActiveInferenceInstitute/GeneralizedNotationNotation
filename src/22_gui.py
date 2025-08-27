#!/usr/bin/env python3
"""
Step 22: GUI Processing (Thin Orchestrator)

This step orchestrates GUI processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/gui/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the gui module.

Pipeline Flow:
    main.py → 22_gui.py (this script) → gui/ (modular implementation)

How to run:
  python src/22_gui.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - GUI processing results in the specified output directory
  - Comprehensive GUI reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that GUI dependencies are installed
  - Check that src/gui/ contains GUI modules
  - Check that the output directory is writable
  - Verify GUI configuration and requirements
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

from gui import (
    process_gui,
    run_gui,
)

run_script = create_standardized_pipeline_script(
    "22_gui.py",
    lambda target_dir, output_dir, logger, **kwargs: _run_gui_processing(
        target_dir, output_dir, logger, **kwargs
    ),
    "GUI processing for interactive GNN construction",
)


def _run_gui_processing(target_dir: Path, output_dir: Path, logger, **kwargs) -> bool:
    """
    Standardized GUI processing function.

    Args:
        target_dir: Directory containing GNN files for GUI processing
        output_dir: Output directory for GUI results
        logger: Logger instance for this step
        **kwargs: Additional processing options

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info("🚀 Processing GUI")

        # Get configuration
        config = get_pipeline_config()
        step_config = config.get_step_config("22_gui") if hasattr(config, 'get_step_config') else None

        # Set up output directory
        step_output_dir = get_output_dir_for_script("22_gui.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Log processing parameters
        logger.info(f"Processing GNN files from: {target_dir}")
        logger.info(f"Output directory: {step_output_dir}")

        # Extract GUI-specific parameters
        gui_mode = kwargs.get('gui_mode', 'generate_artifacts')
        interactive_mode = kwargs.get('interactive_mode', False)

        if gui_mode:
            logger.info(f"GUI mode: {gui_mode}")
        if interactive_mode:
            logger.info("Running in interactive mode")

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

        logger.info(f"Found {len(gnn_files)} GNN files for GUI processing")

        # Process GUI via module API
        logger.info("GUI module available, processing files...")
        return process_gui(target_dir=target_dir, output_dir=step_output_dir, **kwargs)

    except Exception as e:
        log_step_error(logger, f"GUI processing failed: {e}")
        return False


def main() -> int:
    """Main entry point for the GUI step."""
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
