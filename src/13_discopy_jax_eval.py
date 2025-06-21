#!/usr/bin/env python3
"""
Pipeline Step 13: GNN to DisCoPy JAX Diagram Evaluation & Visualization

This script takes GNN model specifications as input, translates them into
DisCoPy MatrixDiagrams with JAX-backed tensors, evaluates these diagrams,
and saves visualizations of the output tensors.
"""

import sys
from pathlib import Path
from typing import Optional, List

# Import centralized utilities - this is the key improvement
from utils import (
    execute_pipeline_step_template,
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    UTILS_AVAILABLE
)

from pipeline import get_output_dir_for_script

# Initialize logger for this step
logger = setup_step_logging("13_discopy_jax_eval", verbose=False)

# Check for JAX availability first
try:
    from discopy_translator_module.translator import (
        JAX_CORE_AVAILABLE,
        DISCOPY_MATRIX_MODULE_AVAILABLE,
        JAX_AVAILABLE,
        discopy_backend,
        gnn_file_to_discopy_matrix_diagram
    )
    from discopy_translator_module.visualize_jax_output import (
        plot_tensor_output, 
        MATPLOTLIB_AVAILABLE
    )
    
    JAX_FULLY_OPERATIONAL = (
        JAX_CORE_AVAILABLE and 
        DISCOPY_MATRIX_MODULE_AVAILABLE and 
        JAX_AVAILABLE and
        discopy_backend is not None
    )
    
    logger.debug(f"JAX availability check: JAX_CORE={JAX_CORE_AVAILABLE}, "
                f"DISCOPY_MATRIX={DISCOPY_MATRIX_MODULE_AVAILABLE}, "
                f"JAX_AVAILABLE={JAX_AVAILABLE}, "
                f"FULLY_OPERATIONAL={JAX_FULLY_OPERATIONAL}")
    
except ImportError as e:
    log_step_warning(logger, f"Failed to import JAX/DisCoPy modules: {e}")
    JAX_FULLY_OPERATIONAL = False
    gnn_file_to_discopy_matrix_diagram = None
    plot_tensor_output = None
    MATPLOTLIB_AVAILABLE = False

def process_gnn_file_for_jax_eval(
    gnn_file_path: Path, 
    eval_output_dir: Path, 
    jax_seed: int, 
    verbose_logging: bool
) -> bool:
    """
    Processes a single GNN file:
    1. Translates it to a DisCoPy MatrixDiagram using the translator module.
    2. Evaluates the diagram using JAX backend.
    3. Saves visualizations of the output tensors.
    """
    logger.info(f"Processing GNN file for JAX evaluation: {gnn_file_path.name}")
    
    if not gnn_file_to_discopy_matrix_diagram:
        log_step_error(logger, "DisCoPy GNN to MatrixDiagram translator not available. Skipping file.")
        return False

    if not JAX_FULLY_OPERATIONAL:
        log_step_error(logger, "JAX is not fully operational. Cannot perform JAX evaluation. Skipping file.")
        return False

    try:
        matrix_diagram = gnn_file_to_discopy_matrix_diagram(gnn_file_path, verbose=verbose_logging)

        if matrix_diagram is None:
            log_step_warning(logger, f"No DisCoPy MatrixDiagram could be generated for {gnn_file_path.name}. Skipping JAX evaluation.")
            return False

        # Evaluate the diagram using JAX backend
        logger.debug(f"Evaluating MatrixDiagram for {gnn_file_path.name} using JAX backend...")
        
        # Set JAX random seed if provided
        if jax_seed is not None:
            try:
                import jax
                key = jax.random.PRNGKey(jax_seed)
                logger.debug(f"Set JAX PRNG seed to {jax_seed}")
            except ImportError:
                log_step_warning(logger, "JAX not available for seed setting")
                key = None
        
        # Evaluate the matrix diagram
        try:
            evaluation_result = matrix_diagram.eval()
            logger.info(f"Successfully evaluated MatrixDiagram for {gnn_file_path.name}")
            
            # Save visualization if matplotlib is available
            if plot_tensor_output and MATPLOTLIB_AVAILABLE:
                viz_file = eval_output_dir / f"{gnn_file_path.stem}_jax_evaluation.png"
                plot_tensor_output(evaluation_result, str(viz_file))
                logger.info(f"Saved JAX evaluation visualization: {viz_file}")
            
            return True
            
        except Exception as eval_error:
            log_step_error(logger, f"Failed to evaluate MatrixDiagram for {gnn_file_path.name}: {eval_error}")
            return False
            
    except Exception as e:
        log_step_error(logger, f"Error processing {gnn_file_path.name}: {e}")
        return False

def main(parsed_args) -> int:
    """Main function for DisCoPy JAX evaluation step."""
    
    log_step_start(logger, "Starting DisCoPy JAX diagram evaluation")
    
    # Update logger verbosity based on arguments
    if getattr(parsed_args, 'verbose', False):
        import logging
        logger.setLevel(logging.DEBUG)
    
    # Check JAX availability early
    if not JAX_FULLY_OPERATIONAL:
        log_step_error(logger, 
            "JAX is not fully operational. This step requires JAX and DisCoPy[matrix] to be installed. "
            "Try: pip install jax discopy[matrix]")
        return 1
    
    # Get input directory - use target_dir if available, fallback to discopy_jax_gnn_input_dir
    input_dir = getattr(parsed_args, 'discopy_jax_gnn_input_dir', None)
    if not input_dir:
        input_dir = getattr(parsed_args, 'target_dir', Path("src/gnn/examples"))
    
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    
    # Set up output directory
    output_dir = Path(getattr(parsed_args, 'output_dir', 'output'))
    eval_output_dir = get_output_dir_for_script("13_discopy_jax_eval.py", output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get JAX seed
    jax_seed = getattr(parsed_args, 'discopy_jax_seed', 0)
    
    # Get recursive flag - this is the critical fix for the original issue
    recursive = getattr(parsed_args, 'recursive', True)
    
    logger.info(f"Processing GNN files from: {input_dir}")
    logger.info(f"Recursive processing: {'enabled' if recursive else 'disabled'}")
    logger.info(f"JAX seed: {jax_seed}")
    logger.info(f"Output directory: {eval_output_dir}")
    
    # Find GNN files
    if not input_dir.exists():
        log_step_error(logger, f"Input directory does not exist: {input_dir}")
        return 1
    
    # Use recursive or non-recursive pattern based on argument
    pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(input_dir.glob(pattern))
    
    if not gnn_files:
        log_step_warning(logger, f"No .md files found in {input_dir} using pattern '{pattern}'")
        return 2  # Warning exit code
    
    logger.info(f"Found {len(gnn_files)} GNN files to process")
    
    # Process each file
    successful_files = 0
    failed_files = 0
    
    for gnn_file in gnn_files:
        try:
            success = process_gnn_file_for_jax_eval(
                gnn_file, 
                eval_output_dir, 
                jax_seed, 
                getattr(parsed_args, 'verbose', False)
            )
            
            if success:
                successful_files += 1
            else:
                failed_files += 1
                
        except Exception as e:
            log_step_error(logger, f"Unexpected error processing {gnn_file}: {e}")
            failed_files += 1
    
    # Report results
    total_files = successful_files + failed_files
    logger.info(f"Processing complete: {successful_files}/{total_files} files successful")
    
    if failed_files == 0:
        log_step_success(logger, "All GNN files processed successfully for JAX evaluation")
        return 0
    elif successful_files > 0:
        log_step_warning(logger, f"Partial success: {failed_files} files failed")
        return 2  # Success with warnings
    else:
        log_step_error(logger, "All files failed to process")
        return 1

# Use the standardized execution template
if __name__ == '__main__':
    step_dependencies = [
        "discopy_translator_module.translator",
        "discopy_translator_module.visualize_jax_output"
    ]
    
    exit_code = execute_pipeline_step_template(
        step_name="13_discopy_jax_eval",
        step_description="DisCoPy JAX diagram evaluation and visualization",
        main_function=main,
        import_dependencies=step_dependencies
    )
    
    sys.exit(exit_code) 