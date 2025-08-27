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

from integration import (
    process_integration,
)

run_script = create_standardized_pipeline_script(
    "17_integration.py",
    lambda target_dir, output_dir, logger, **kwargs: _run_integration_processing(
        target_dir, output_dir, logger, **kwargs
    ),
    "Integration processing for GNN system coordination",
)


def _run_integration_processing(target_dir: Path, output_dir: Path, logger, **kwargs) -> bool:
    """
    Standardized integration processing function.

    Args:
        target_dir: Directory containing GNN files for integration
        output_dir: Output directory for integration results
        logger: Logger instance for this step
        **kwargs: Additional processing options

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info("🚀 Processing integration")

        # Get configuration
        config = get_pipeline_config()
        step_config = config.get_step_config("17_integration") if hasattr(config, 'get_step_config') else None

        # Set up output directory
        step_output_dir = get_output_dir_for_script("17_integration.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Log processing parameters
        logger.info(f"Processing GNN files from: {target_dir}")
        logger.info(f"Output directory: {step_output_dir}")

        # Extract integration-specific parameters
        integration_mode = kwargs.get('integration_mode', 'coordinated')
        system_coordination = kwargs.get('system_coordination', True)

        if integration_mode:
            logger.info(f"Integration mode: {integration_mode}")
        if system_coordination:
            logger.info("System coordination enabled")

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

        logger.info(f"Found {len(gnn_files)} GNN files for integration")

        # Process integration via module API
        logger.info("Integration module available, processing files...")
        return process_integration(target_dir=target_dir, output_dir=step_output_dir, **kwargs)

    except Exception as e:
        log_step_error(logger, f"Integration processing failed: {e}")
        return False


def main() -> int:
    """Main entry point for the integration step."""
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
