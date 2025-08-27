#!/usr/bin/env python3
"""
Step 23: Report Processing (Thin Orchestrator)

This step orchestrates report processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/report/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the report module.

Pipeline Flow:
    main.py → 23_report.py (this script) → report/ (modular implementation)

How to run:
  python src/23_report.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Report processing results in the specified output directory
  - Comprehensive report reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that report dependencies are installed
  - Check that src/report/ contains report modules
  - Check that the output directory is writable
  - Verify report configuration and requirements
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

from report import (
    process_report,
)

run_script = create_standardized_pipeline_script(
    "23_report.py",
    lambda target_dir, output_dir, logger, **kwargs: _run_report_processing(
        target_dir, output_dir, logger, **kwargs
    ),
    "Report processing for comprehensive GNN analysis",
)


def _run_report_processing(target_dir: Path, output_dir: Path, logger, **kwargs) -> bool:
    """
    Standardized report processing function.

    Args:
        target_dir: Directory containing GNN files for report generation
        output_dir: Output directory for report results
        logger: Logger instance for this step
        **kwargs: Additional processing options

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info("🚀 Processing report")

        # Get configuration
        config = get_pipeline_config()
        step_config = config.get_step_config("23_report") if hasattr(config, 'get_step_config') else None

        # Set up output directory
        step_output_dir = get_output_dir_for_script("23_report.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Log processing parameters
        logger.info(f"Processing GNN files from: {target_dir}")
        logger.info(f"Output directory: {step_output_dir}")

        # Extract report-specific parameters
        report_format = kwargs.get('report_format', 'comprehensive')
        include_visualizations = kwargs.get('include_visualizations', True)

        if report_format:
            logger.info(f"Report format: {report_format}")
        if include_visualizations:
            logger.info("Including visualizations in report")

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

        logger.info(f"Found {len(gnn_files)} GNN files for report generation")

        # Process report via module API
        logger.info("Report module available, processing files...")
        return process_report(target_dir=target_dir, output_dir=step_output_dir, **kwargs)

    except Exception as e:
        log_step_error(logger, f"Report processing failed: {e}")
        return False


def main() -> int:
    """Main entry point for the report step."""
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
