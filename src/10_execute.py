#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 10: Execute

This script executes rendered GNN simulators.

Usage:
    python 10_execute.py [options]
    (Typically called by main.py)
"""

import sys
import subprocess
import json
import time
import logging
from pathlib import Path
import argparse

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    performance_tracker,
    UTILS_AVAILABLE
)

from pipeline import (
    STEP_METADATA,
    get_output_dir_for_script
)

from execute.executor import execute_rendered_simulators

# Initialize logger for this step
logger = setup_step_logging("10_execute", verbose=False)

# Import execution functionality
try:
    from execute.pymdp.pymdp_runner import run_pymdp_scripts
    PYMDP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyMDP execution not available: {e}")
    PYMDP_AVAILABLE = False
    run_pymdp_scripts = None

try:
    from execute.rxinfer.rxinfer_runner import run_rxinfer_scripts
    RXINFER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RxInfer execution not available: {e}")
    RXINFER_AVAILABLE = False
    run_rxinfer_scripts = None

try:
    from execute.discopy.discopy_executor import DisCoPyExecutor, run_discopy_analysis
    DISCOPY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DisCoPy execution not available: {e}")
    DISCOPY_AVAILABLE = False
    DisCoPyExecutor = None
    run_discopy_analysis = None

try:
    from execute.activeinference_jl.activeinference_runner import run_activeinference_analysis
    ACTIVEINFERENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ActiveInference.jl execution not available: {e}")
    ACTIVEINFERENCE_AVAILABLE = False
    run_activeinference_analysis = None

try:
    from execute.jax.jax_runner import run_jax_scripts
    JAX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"JAX execution not available: {e}")
    JAX_AVAILABLE = False
    run_jax_scripts = None

logger.debug(f"Execution modules availability - PyMDP: {PYMDP_AVAILABLE}, RxInfer: {RXINFER_AVAILABLE}, DisCoPy: {DISCOPY_AVAILABLE}, ActiveInference.jl: {ACTIVEINFERENCE_AVAILABLE}, JAX: {JAX_AVAILABLE}")

def main(parsed_args: argparse.Namespace):
    """Main function for execution operations."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("10_execute.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Execute rendered simulators')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Execute rendered simulators
    success = execute_rendered_simulators(
        logger=logger,
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        recursive=getattr(parsed_args, 'recursive', False),
        verbose=getattr(parsed_args, 'verbose', False)
    )
    
    if success:
        log_step_success(logger, "Execution completed successfully")
        return 0
    else:
        log_step_error(logger, "Execution failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("10_execute")
    else:
        # Fallback argument parsing
        parser = argparse.ArgumentParser(description="Execute rendered simulators")
        parser.add_argument("--target-dir", type=Path, required=True,
                          help="Target directory containing GNN files")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--recursive", action="store_true",
                          help="Search recursively in subdirectories")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 