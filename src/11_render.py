#!/usr/bin/env python3
"""
Step 11: Render Processing (Thin Orchestrator)

This step orchestrates render processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/render/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the render module.

Pipeline Flow:
    main.py → 11_render.py (this script) → render/ (modular implementation)

How to run:
  python src/11_render.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Render processing results in the specified output directory
  - Comprehensive render reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that render dependencies are installed
  - Check that src/render/ contains render modules
  - Check that the output directory is writable
  - Verify render configuration and requirements
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

from render import (
    process_render,
    render_gnn_spec,
    get_module_info,
    get_available_renderers
)

run_script = create_standardized_pipeline_script(
    "11_render.py",
    lambda target_dir, output_dir, logger, **kwargs: _run_render_processing(
        target_dir, output_dir, logger, **kwargs
    ),
    "Render processing for GNN specifications",
)


def _run_render_processing(target_dir: Path, output_dir: Path, logger, **kwargs) -> bool:
    """
    Standardized render processing function.

    Args:
        target_dir: Directory containing GNN files to render
        output_dir: Output directory for render results
        logger: Logger instance for this step
        **kwargs: Additional processing options

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info("🚀 Processing render")

        # Get configuration
        config = get_pipeline_config()
        step_config = config.get_step_config("11_render") if hasattr(config, 'get_step_config') else None

        # Set up output directory
        step_output_dir = get_output_dir_for_script("11_render.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Log processing parameters
        logger.info(f"Processing GNN files from: {target_dir}")
        logger.info(f"Output directory: {step_output_dir}")

        # Extract render-specific parameters
        render_format = kwargs.get('render_format', 'all')
        target_language = kwargs.get('target_language', None)

        if render_format:
            logger.info(f"Render format: {render_format}")
        if target_language:
            logger.info(f"Target language: {target_language}")

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

        logger.info(f"Found {len(gnn_files)} GNN files to render")

        # Process render via module API
        logger.info("Render module available, processing files...")
        return process_render(target_dir=target_dir, output_dir=step_output_dir, **kwargs)

    except Exception as e:
        log_step_error(logger, f"Render processing failed: {e}")
        return False


def main() -> int:
    """Main entry point for the render step."""
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
