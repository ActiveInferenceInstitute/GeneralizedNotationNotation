#!/usr/bin/env python3
"""
Step 12: Execute Processing (Thin Orchestrator)

This step orchestrates execute processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/execute/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the execute module.

Pipeline Flow:
    main.py → 12_execute.py (this script) → execute/ (modular implementation)

How to run:
  python src/12_execute.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Execute processing results in the specified output directory
  - Comprehensive execute reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that execute dependencies are installed
  - Check that src/execute/ contains execute modules
  - Check that the output directory is writable
  - Verify execute configuration and requirements
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

from execute import (
    process_execute,
    execute_simulation_from_gnn,
    validate_execution_environment,
    get_execution_health_status,
)

run_script = create_standardized_pipeline_script(
    "12_execute.py",
    lambda target_dir, output_dir, logger, **kwargs: _run_execute_processing(
        target_dir, output_dir, logger, **kwargs
    ),
    "Execute processing for GNN simulations",
)


def _run_execute_processing(target_dir: Path, output_dir: Path, logger, **kwargs) -> bool:
    """
    Standardized execute processing function with enhanced dependency handling.

    Args:
        target_dir: Directory containing GNN files to execute
        output_dir: Output directory for execute results
        logger: Logger instance for this step
        **kwargs: Additional processing options

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info("🚀 Processing execute")

        # Check dependencies for execute step
        try:
            from utils.dependency_manager import log_dependency_status
            log_dependency_status("12_execute", logger)
        except ImportError:
            logger.warning("⚠️ Dependency manager not available - proceeding with basic checks")
            
            # Basic fallback dependency checking
            missing_deps = []
            try:
                import numpy
            except ImportError:
                missing_deps.append("numpy")
            
            # Check for simulation engines
            simulation_engines = []
            try:
                import pymdp
                simulation_engines.append("PyMDP")
            except ImportError:
                logger.warning("⚠️ PyMDP not available - PyMDP simulations will be skipped")
            
            try:
                import subprocess
                result = subprocess.run(["julia", "--version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    simulation_engines.append("Julia/RxInfer")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.warning("⚠️ Julia not available - RxInfer simulations will be skipped")
                
            if missing_deps:
                log_step_error(logger, f"Critical dependencies missing: {missing_deps}")
                return False
            
            if simulation_engines:
                logger.info(f"✅ Available simulation engines: {simulation_engines}")
            else:
                log_step_warning(logger, "No simulation engines available - execution will be limited")

        # Get configuration
        config = get_pipeline_config()
        step_config = config.get_step_config("12_execute") if hasattr(config, 'get_step_config') else None

        # Set up output directory
        step_output_dir = get_output_dir_for_script("12_execute.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Log processing parameters
        logger.info(f"Processing GNN files from: {target_dir}")
        logger.info(f"Output directory: {step_output_dir}")

        # Extract execute-specific parameters
        simulation_engine = kwargs.get('simulation_engine', 'auto')
        validate_only = kwargs.get('validate_only', False)

        if simulation_engine:
            logger.info(f"Simulation engine: {simulation_engine}")
        if validate_only:
            logger.info("Running in validation-only mode")

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

        logger.info(f"Found {len(gnn_files)} GNN files to execute")

        # Process execute via module API
        logger.info("Execute module available, processing files...")
        return process_execute(target_dir=target_dir, output_dir=step_output_dir, **kwargs)

    except Exception as e:
        log_step_error(logger, f"Execute processing failed: {e}")
        return False


def main() -> int:
    """Main entry point for the execute step."""
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
