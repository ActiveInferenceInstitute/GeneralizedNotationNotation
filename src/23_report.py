#!/usr/bin/env python3
"""
Step 23: Report Generation (Thin Orchestrator)

This step orchestrates comprehensive analysis report generation from pipeline artifacts.
It is a thin orchestrator that delegates core functionality to the report module.

How to run:
  python src/23_report.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Comprehensive analysis reports in the specified output directory
  - Statistical summaries and performance metrics
  - Research findings and recommendations
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_success,
    log_step_error,
    log_step_warning
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

from report import process_report


def process_report_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    try:
        if verbose:
            logger.setLevel(logging.DEBUG)

        step_output_dir = get_output_dir_for_script("23_report.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)

        if not target_dir.exists():
            log_step_error(logger, f"Input directory does not exist: {target_dir}")
            return False

        pattern = "**/*.md" if recursive else "*.md"
        gnn_files = list(target_dir.glob(pattern))
        if not gnn_files:
            log_step_warning(logger, f"No GNN files found in {target_dir}")
            return True

        success = process_report(
            target_dir=target_dir,
            output_dir=step_output_dir,
            verbose=verbose,
            recursive=recursive,
            **kwargs
        )

        if success:
            log_step_success(logger, f"Successfully generated reports for {len(gnn_files)} GNN models")
        else:
            log_step_error(logger, "Report generation failed")

        return success
    except Exception as e:
        log_step_error(logger, f"Report generation failed: {e}")
        if verbose:
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def main():
    args = EnhancedArgumentParser.parse_step_arguments("23_report")
    logger = setup_step_logging("report", args)

    try:
        output_dir = get_output_dir_for_script("23_report.py", Path(args.output_dir))
        success = process_report_standardized(
            target_dir=Path(args.target_dir) if hasattr(args, 'target_dir') else Path("input/gnn_files"),
            output_dir=output_dir,
            logger=logger,
            recursive=getattr(args, 'recursive', False),
            verbose=getattr(args, 'verbose', False)
        )
        return 0 if success else 1
    except Exception as e:
        log_step_error(logger, f"Report generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


