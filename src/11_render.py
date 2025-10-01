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
    """Execute render processing with proper delegation to render module."""
    try:
        # Extract verbose flag from kwargs, defaulting to False
        verbose = kwargs.pop('verbose', False)
        
        # Call the render module's main processing function
        # Note: process_render expects (target_dir, output_dir, verbose, **kwargs)
        result = process_render(target_dir, output_dir, verbose, **kwargs)
        return result
    except Exception as e:
        logger.error(f"Render processing failed: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

def main() -> int:
    """Main entry point for the render step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
