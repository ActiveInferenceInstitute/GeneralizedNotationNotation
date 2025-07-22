#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 10: Execute

This script executes rendered simulation scripts.

Usage:
    python 10_execute.py [options]
    (Typically called by main.py)
"""

import sys
import logging
from pathlib import Path
from typing import Dict
import argparse

# Add src directory to Python path for imports
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

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
    get_output_dir_for_script
)

from execute.executor import execute_rendered_simulators
from utils.pipeline_template import create_standardized_pipeline_script

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

def process_execute_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized execution processing function with consistent signature.
    
    Args:
        target_dir: Directory containing rendered simulators to execute
        output_dir: Output directory for execution results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Update logger verbosity if needed
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Step 10 should process rendered simulators from step 9
        # Check if target_dir exists, if not, look for rendered simulators in output_dir
        if target_dir is None or not target_dir.exists():
            # Look for rendered simulators in the expected location from step 9
            render_dir = output_dir / "gnn_rendered_simulators"
            if render_dir.exists():
                target_dir = render_dir
                logger.info(f"Using rendered simulators from step 9: {target_dir}")
            else:
                # Check if there are any existing execution logs to use as source
                execute_log_dir = output_dir / "execute_logs"
                if execute_log_dir.exists():
                    target_dir = execute_log_dir
                    logger.info(f"Using existing execution directory: {target_dir}")
                else:
                    log_step_warning(logger, f"No rendered simulators found in expected locations: {render_dir}")
                    log_step_warning(logger, "This may be expected if step 9 (render) did not complete successfully")
                    # Don't return False here - let the execution step handle the empty directory gracefully
                    target_dir = render_dir  # Use the expected directory even if it doesn't exist
        
        # Call the existing execute_rendered_simulators function
        success = execute_rendered_simulators(
            target_dir=target_dir,
            output_dir=output_dir,
            recursive=recursive,
            logger=logger,
            verbose=verbose
        )
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Execute failed: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "10_execute.py",
    process_execute_standardized,
    "Execute rendered simulators"
)

if __name__ == '__main__':
    sys.exit(run_script()) 