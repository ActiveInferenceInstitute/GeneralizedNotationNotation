#!/usr/bin/env python3
"""
Step 22: GUI (Interactive GNN Constructor) - Thin Orchestrator

This step launches/produces artifacts for an interactive GUI that allows users to
construct and edit GNN models. The GUI presents controls on the left (buttons,
options, fields) and a synchronized plaintext GNN markdown editor on the right.

How to run:
  python src/22_gui.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - GUI assets and logs in the step-specific output directory
  - Optionally a saved GNN markdown file reflecting edits (if run in export mode)

Notes:
  - This is a thin orchestrator. Core logic is implemented in src/gui/.
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
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

from gui import run_gui


def process_gui_standardized(
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

        # output_dir is already the step-specific directory (resolved in main)
        step_output_dir = output_dir
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Launch or generate GUI artifacts
        success = run_gui(
            target_dir=target_dir,
            output_dir=step_output_dir,
            logger=logger,
            verbose=verbose,
            **kwargs
        )

        if success:
            log_step_success(logger, "GUI step completed")
        else:
            log_step_error(logger, "GUI step failed")

        return success
    except Exception as e:
        log_step_error(logger, f"GUI processing failed: {e}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        return False


def main():
    args = EnhancedArgumentParser.parse_step_arguments("22_gui")
    logger = setup_step_logging("gui", args)

    output_dir = get_output_dir_for_script("22_gui.py", Path(args.output_dir))

    success = process_gui_standardized(
        target_dir=Path(args.target_dir) if hasattr(args, 'target_dir') else Path("input/gnn_files"),
        output_dir=output_dir,
        logger=logger,
        recursive=getattr(args, 'recursive', False),
        verbose=getattr(args, 'verbose', False)
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())


