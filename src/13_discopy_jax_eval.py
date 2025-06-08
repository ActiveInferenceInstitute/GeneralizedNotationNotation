#!/usr/bin/env python3
"""
Pipeline Step 13: GNN to DisCoPy JAX Diagram Evaluation & Visualization

This script takes GNN model specifications as input, translates them into
DisCoPy MatrixDiagrams with JAX-backed tensors, evaluates these diagrams,
and saves visualizations of the output tensors.
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import numpy
from typing import Optional, Tuple, List, Dict, Any
import shutil # For cleaning up output directory

# Import centralized utilities
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    UTILS_AVAILABLE
)

# Initialize logger for this step
logger = setup_step_logging("13_discopy_jax_eval", verbose=False)

# --- Python Path Setup for 'src' imports ---
# Ensure the script can find other modules in the 'src' directory, especially when run as a subprocess.
# This makes the script more robust to different execution contexts.
_current_script_path = Path(__file__).resolve()
_src_dir = _current_script_path.parent.parent # Assuming src/13_discopy_jax_eval.py, so parent.parent is project root
_project_root = _src_dir # If script is in src, then src_dir is project_root/src. If project_root is desired, use parent.parent

# Add project_root to sys.path if not already present, to allow 'from src.module import ...'
# This specifically targets the directory containing the 'src' folder.
_generalized_notation_notation_root = _current_script_path.parent.parent # This should be GeneralizedNotationNotation/
if str(_generalized_notation_notation_root) not in sys.path:
    sys.path.insert(0, str(_generalized_notation_notation_root))
    # print(f"[13_discopy_jax_eval.py sys.path debug] Added to sys.path: {str(_generalized_notation_notation_root)}", file=sys.stderr)
# --- End Python Path Setup ---

# --- JAX and DisCoPy Availability Flags & Placeholders (initialized assuming not available) ---
_JAX_CORE_AVAILABLE_FROM_TRANSLATOR = False
_DISCOPY_JAX_BACKEND_FROM_TRANSLATOR = None # Will hold the actual backend object or placeholder
_DISCOPY_MATRIX_COMPONENTS_AVAILABLE_FROM_TRANSLATOR = False # New flag

try:
    # Import all relevant flags and the backend object from the translator
    from src.discopy_translator_module.translator import JAX_CORE_AVAILABLE as _JAX_CORE_AVAILABLE_FROM_TRANSLATOR, \
                                                       DISCOPY_MATRIX_MODULE_AVAILABLE as _DISCOPY_MATRIX_COMPONENTS_AVAILABLE_FROM_TRANSLATOR, \
                                                       JAX_AVAILABLE as _OVERALL_JAX_READY_FROM_TRANSLATOR, \
                                                       discopy_backend as _DISCOPY_JAX_BACKEND_FROM_TRANSLATOR, \
                                                       Ty as _DisCoPy_Ty, Dim as _DisCoPy_Dim, Box as _DisCoPy_Box, Diagram as _DisCoPy_Diagram, Matrix as _DisCoPy_Matrix # Import DisCoPy base classes

    # The following debug logs will be moved outside this try-except block
    # logger.debug(f"Successfully imported flags from translator: JAX_CORE={_JAX_CORE_AVAILABLE_FROM_TRANSLATOR}, "
    #              f"DISCOPY_MATRIX_COMPONENTS={_DISCOPY_MATRIX_COMPONENTS_AVAILABLE_FROM_TRANSLATOR}, "
    #              f"OVERALL_JAX_READY={_OVERALL_JAX_READY_FROM_TRANSLATOR}")
    # logger.debug(f"Imported discopy_backend from translator is of type: {type(_DISCOPY_JAX_BACKEND_FROM_TRANSLATOR)}")
    # logger.debug(f"Imported DisCoPy types: Ty={type(_DisCoPy_Ty)}, Dim={type(_DisCoPy_Dim)}, Box={type(_DisCoPy_Box)}, Diagram={type(_DisCoPy_Diagram)}, Matrix={type(_DisCoPy_Matrix)}")

except ImportError as e_translator_core:
    logging.basicConfig(level=logging.ERROR) # Basic fallback logger if module logger isn't even setup
    # logger is defined at module level, should be available here.
    # If this critical error occurs, the script will likely abort in check_jax_fully_operational().
    logger.critical(f"Failed to import core JAX flags or types from translator: {e_translator_core}")
    _JAX_CORE_AVAILABLE_FROM_TRANSLATOR = False
    _DISCOPY_MATRIX_COMPONENTS_AVAILABLE_FROM_TRANSLATOR = False
    _OVERALL_JAX_READY_FROM_TRANSLATOR = False
    _DISCOPY_JAX_BACKEND_FROM_TRANSLATOR = None 
    # Placeholders for DisCoPy types will be used as they are initialized globally

# Log the imported flags and backend type for debugging - MOVED HERE from the try block
logger.debug(f"Flags from translator after import attempt: JAX_CORE={_JAX_CORE_AVAILABLE_FROM_TRANSLATOR}, "
             f"DISCOPY_MATRIX_COMPONENTS={_DISCOPY_MATRIX_COMPONENTS_AVAILABLE_FROM_TRANSLATOR}, "
             f"OVERALL_JAX_READY={_OVERALL_JAX_READY_FROM_TRANSLATOR}")
logger.debug(f"Imported discopy_backend from translator is of type: {type(_DISCOPY_JAX_BACKEND_FROM_TRANSLATOR)}")
# Ensure _DisCoPy_Ty etc. are defined even if import failed (they should be due to global placeholders)
# Using string 'PlaceholderBase_local_ref' for logging if actual PlaceholderBase not imported.
logger.debug(f"Imported DisCoPy types after import attempt: Ty={type(globals().get('_DisCoPy_Ty', 'PlaceholderBase_local_ref'))}, "
             f"Dim={type(globals().get('_DisCoPy_Dim', 'PlaceholderBase_local_ref'))}, Box={type(globals().get('_DisCoPy_Box', 'PlaceholderBase_local_ref'))}, "
             f"Diagram={type(globals().get('_DisCoPy_Diagram', 'PlaceholderBase_local_ref'))}, Matrix={type(globals().get('_DisCoPy_Matrix', 'PlaceholderBase_local_ref'))}")

try:
    # These are the primary functions and flags this script uses directly from the translator
    from src.discopy_translator_module.translator import gnn_file_to_discopy_matrix_diagram, PlaceholderBase as TranslatorPlaceholderBase
    from src.discopy_translator_module.visualize_jax_output import plot_tensor_output, MATPLOTLIB_AVAILABLE
    from src.utils.logging_utils import setup_standalone_logging

    # Check if core JAX was found by the translator
    if _JAX_CORE_AVAILABLE_FROM_TRANSLATOR:
        # Attempt to import discopy.matrix.backend directly in this script as a fallback or primary source
        # This allows this script to potentially work even if translator's dynamic import had an issue,
        # as long as JAX is present and discopy[matrix] is correctly installed.
        try:
            from discopy.matrix import backend as discopy_backend_direct
            # If translator also found it, they should be the same. If not, prefer the direct import.
            if _DISCOPY_JAX_BACKEND_FROM_TRANSLATOR is None:
                logger_init_info_backend = logging.getLogger(__name__)
                logger_init_info_backend.info("DisCoPy JAX backend successfully imported directly in 13_discopy_jax_eval.py.")
            discopy_backend_to_use = discopy_backend_direct
        except ImportError as e_discopy_matrix_backend:
            # This case means JAX is found by translator, but discopy.matrix.backend is not importable here.
            # It could indicate a partial/broken discopy install regarding JAX features.
            # We might still use the one from translator if it was found there.
            if _DISCOPY_JAX_BACKEND_FROM_TRANSLATOR is not None:
                logger_init_warn_backend = logging.getLogger(__name__)
                logger_init_warn_backend.warning(
                    f"JAX core is available, but failed to import discopy.matrix.backend directly: {e_discopy_matrix_backend}. "
                    f"Will rely on backend from translator: {_DISCOPY_JAX_BACKEND_FROM_TRANSLATOR}."
                )
                discopy_backend_to_use = _DISCOPY_JAX_BACKEND_FROM_TRANSLATOR
            else:
                # Both attempts failed
                logging.basicConfig(level=logging.ERROR) # Ensure this is visible
                logger_init_fail_backend = logging.getLogger(__name__)
                logger_init_fail_backend.error(
                    f"JAX core is available, but failed to import discopy.matrix.backend (directly and via translator): {e_discopy_matrix_backend}. "
                    "This might indicate an incomplete DisCoPy installation for JAX matrix features. "
                    "Try reinstalling with JAX support: pip install \"discopy[matrix]\". Aborting 13_discopy_jax_eval.py step."
                )
                discopy_backend_to_use = None # Ensure it's None if import fails
    else:
        # JAX_CORE_AVAILABLE_FROM_TRANSLATOR is False
        discopy_backend_to_use = None

except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init_fail = logging.getLogger(__name__)
    logger_init_fail.error(f"Failed to import necessary modules for 13_discopy_jax_eval.py: {e}. Ensure translator.py, visualize_jax_output.py, and logging_utils.py are accessible, and JAX/DisCoPy matrix modules are installed if JAX features are intended.")
    # Define placeholders if imports fail
    gnn_file_to_discopy_matrix_diagram = None 
    plot_tensor_output = None
    MATPLOTLIB_AVAILABLE = False
    setup_standalone_logging = None
    # _JAX_CORE_AVAILABLE_FROM_TRANSLATOR is already False or not set
    discopy_backend_to_use = None

# Final check for JAX availability for the script
JAX_FULLY_OPERATIONAL = _JAX_CORE_AVAILABLE_FROM_TRANSLATOR and \
                        _DISCOPY_MATRIX_COMPONENTS_AVAILABLE_FROM_TRANSLATOR and \
                        _OVERALL_JAX_READY_FROM_TRANSLATOR and \
                        _DISCOPY_JAX_BACKEND_FROM_TRANSLATOR is not None and \
                        not isinstance(_DISCOPY_JAX_BACKEND_FROM_TRANSLATOR, TranslatorPlaceholderBase) # Check if it's not a placeholder type

if JAX_FULLY_OPERATIONAL and not _OVERALL_JAX_READY_FROM_TRANSLATOR: # Should ideally not happen
    logger_init_anomaly = logging.getLogger(__name__)
    logger_init_anomaly.error("Anomaly: JAX_FULLY_OPERATIONAL is True but _OVERALL_JAX_READY_FROM_TRANSLATOR is False. Resetting JAX_FULLY_OPERATIONAL to False.")
    JAX_FULLY_OPERATIONAL = False
elif not JAX_FULLY_OPERATIONAL and _JAX_CORE_AVAILABLE_FROM_TRANSLATOR and _DISCOPY_MATRIX_COMPONENTS_AVAILABLE_FROM_TRANSLATOR and _OVERALL_JAX_READY_FROM_TRANSLATOR and _DISCOPY_JAX_BACKEND_FROM_TRANSLATOR is not None and isinstance(_DISCOPY_JAX_BACKEND_FROM_TRANSLATOR, TranslatorPlaceholderBase):
    logger_init_placeholder_warn = logging.getLogger(__name__)
    logger_init_placeholder_warn.warning(
        f"JAX core library is available, but the DisCoPy JAX backend resolved to a placeholder type ({type(_DISCOPY_JAX_BACKEND_FROM_TRANSLATOR)}). "
        "This usually means DisCoPy's JAX-specific components (like discopy.matrix.backend) could not be imported correctly, "
        "even if JAX itself is present. JAX-dependent operations in this script will likely fail or be non-functional. "
        "Consider reinstalling DisCoPy with matrix/JAX support: pip install --upgrade \"discopy[matrix]\". Aborting 13_discopy_jax_eval.py step."
    )
# Redundant checks for placeholder removed as direct isinstance is used now.

DEFAULT_OUTPUT_SUBDIR = "discopy_jax_eval"

# Initialize variables for standalone execution context
cli_args: argparse.Namespace | None = None
log_level_to_set: int = logging.INFO
_log_level_standalone: int = logging.INFO
exit_code: int = 1

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the GNN to DisCoPy JAX evaluation script."""
    parser = argparse.ArgumentParser(description="Transforms GNN models to DisCoPy JAX MatrixDiagrams, evaluates, and saves output visualizations.")
    
    # Define a mutually exclusive group for input directories
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--gnn-input-dir",
        type=Path,
        help="Directory containing GNN files (e.g., .gnn.md, .md) to process."
    )
    input_group.add_argument(
        "--target-dir",
        type=Path,
        help="Alternative name for the directory containing GNN files (compatible with main.py)."
    )
    
    parser.add_argument(
        "--output-dir", 
        type=Path,
        required=True,
        help=f"Main pipeline output directory. JAX evaluation outputs will be saved in a '{DEFAULT_OUTPUT_SUBDIR}' subdirectory here."
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively search for GNN files in the gnn-input-dir. Default: True."
    )
    parser.add_argument(
        "--jax-seed",
        type=int,
        default=0,
        help="Seed for JAX pseudo-random number generator used in tensor initializations. Default: 0."
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose (DEBUG level) logging for this script and dependent modules. Default: False."
    )
    
    args = parser.parse_args()
    
    # If target-dir is provided but gnn-input-dir is not, use target-dir as gnn-input-dir
    if args.target_dir is not None and args.gnn_input_dir is None:
        args.gnn_input_dir = args.target_dir
    
    return args

def process_gnn_file_for_jax_eval(gnn_file_path: Path, eval_output_dir: Path, jax_seed: int, verbose_logging: bool) -> bool:
    """
    Processes a single GNN file for JAX evaluation:
    1. Translates to a DisCoPy MatrixDiagram using the translator module.
    2. Evaluates the diagram using JAX.
    3. Saves a visualization of the output tensor.
    """
    logger.info(f"Processing GNN file for JAX evaluation: {gnn_file_path.name}")
    if not gnn_file_to_discopy_matrix_diagram or not JAX_FULLY_OPERATIONAL or not discopy_backend_to_use:
        logger.error("DisCoPy JAX GNN translator, core JAX, or DisCoPy JAX backend is not fully available/operational. Skipping file.")
        return False

    try:
        # The translator's verbose flag is controlled by this script's verbose_logging flag
        diagram = gnn_file_to_discopy_matrix_diagram(gnn_file_path, verbose=verbose_logging, jax_seed=jax_seed)

        if diagram is None:
            logger.warning(f"No DisCoPy MatrixDiagram could be generated for {gnn_file_path.name}. Skipping evaluation.")
            return False

        logger.info(f"Successfully created MatrixDiagram for {gnn_file_path.name}. Evaluating with JAX...")
        
        eval_result_tensor = None
        with discopy_backend_to_use('jax'): # Ensure JAX backend is active for evaluation
            eval_result_tensor = diagram.eval() # This performs the JAX computation
        
        # Log shape safely
        output_shape_info = "unknown (placeholder or not an array)"
        if hasattr(eval_result_tensor, 'array'):
            actual_array = eval_result_tensor.array
            # Check if actual_array is a JAX or NumPy array before accessing .shape
            if JAX_FULLY_OPERATIONAL and _JAX_CORE_AVAILABLE_FROM_TRANSLATOR: # implies jnp should be available
                import jax.numpy as jnp # ensure jnp is in scope for isinstance
                import numpy
                if isinstance(actual_array, (jnp.ndarray, numpy.ndarray)):
                    output_shape_info = str(actual_array.shape)
            elif isinstance(actual_array, numpy.ndarray): # Fallback if JAX not fully up but we got a numpy array somehow
                    output_shape_info = str(actual_array.shape)
            # If it's a placeholder, output_shape_info remains "unknown..."
        
        logger.info(f"Evaluation successful for {gnn_file_path.name}. Output tensor shape: {output_shape_info}")

        # Save output tensor visualization
        # Use the GNN file stem for the output visualization name
        output_tensor_path_base = eval_output_dir / gnn_file_path.stem
        
        if plot_tensor_output:
            plot_title = f"JAX Evaluation Output: {gnn_file_path.name}"
            plot_tensor_output(eval_result_tensor.array, output_tensor_path_base, title=plot_title, verbose=verbose_logging)
            logger.info(f"Saved JAX evaluation output visualization based on: {output_tensor_path_base}")
        else:
            logger.warning(f"Tensor visualization function not available. Raw output for {gnn_file_path.name} will not be plotted, check logs for data details if saved by plotter fallback.")

        return True

    except ImportError as e_missing_module:
        logger.error(f"Missing import for JAX evaluation of {gnn_file_path.name}: {e_missing_module}. Ensure JAX and all discopy[matrix] dependencies are installed.")
        return False
    except Exception as e:
        logger.error(f"Failed to process GNN file {gnn_file_path.name} for JAX evaluation: {e}", exc_info=True)
        return False

def main_discopy_jax_eval_step(args: argparse.Namespace) -> int:
    """Main execution function for the 13_discopy_jax_eval.py pipeline step."""
    
    log_step_start(logger, "Starting DisCoPy JAX evaluation step")
    
    if not _JAX_CORE_AVAILABLE_FROM_TRANSLATOR:
        error_msg = ("Core JAX library (jax, jax.numpy) was not found or could not be imported by the translator module. This step requires JAX. " 
                    "Please install JAX and jaxlib (e.g., 'pip install \"jax[cpu]\") and ensure DisCoPy was " 
                    "installed with matrix support (e.g., 'pip install \"discopy[matrix]\"). Aborting 13_discopy_jax_eval.py step.")
        log_step_error(logger, error_msg)
        return 1
    
    if not _DISCOPY_MATRIX_COMPONENTS_AVAILABLE_FROM_TRANSLATOR:
        logger.critical(
            "DisCoPy's matrix components (e.g., discopy.matrix.Dim, discopy.matrix.Box, discopy.matrix.backend) were not found by the translator, "
            "even if JAX core might be present. This step requires these components for JAX operations. "
            "Try reinstalling DisCoPy with matrix support: 'pip install --upgrade --force-reinstall --no-cache-dir \"discopy[matrix]\'. Aborting 13_discopy_jax_eval.py step."
        )
        return 1

    if not gnn_file_to_discopy_matrix_diagram:
        logger.critical(
            "The core GNN to DisCoPy MatrixDiagram translator function " 
            "(gnn_file_to_discopy_matrix_diagram from translator.py) is not available. " 
            "This indicates an issue with the translator module itself. Aborting 13_discopy_jax_eval.py step."
        )
        return 1

    logger.info(f"Starting pipeline step: {Path(__file__).name} - GNN to DisCoPy JAX Evaluation")
    logger.info(f"Reading GNN files from: {args.gnn_input_dir.resolve()}")
    logger.info(f"JAX seed for initialization: {args.jax_seed}")
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not found. Tensor visualizations will be limited to raw data output where applicable.")

    jax_eval_step_output_dir = args.output_dir.resolve() / DEFAULT_OUTPUT_SUBDIR
    try:
        jax_eval_step_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"JAX evaluation outputs will be saved in: {jax_eval_step_output_dir}")
    except OSError as e:
        logger.error(f"Failed to create JAX evaluation output directory {jax_eval_step_output_dir}: {e}")
        return 1

    if not args.gnn_input_dir.is_dir():
        logger.error(f"GNN input directory not found: {args.gnn_input_dir.resolve()}")
        return 1

    glob_pattern = "**/*.md" if args.recursive else "*.md"
    gnn_files = list(args.gnn_input_dir.glob(glob_pattern))
    if not args.recursive:
        gnn_files.extend(list(args.gnn_input_dir.glob("*.gnn.md")))
    gnn_files = sorted(list(set(gnn_files)))

    if not gnn_files:
        logger.warning(f"No GNN files found in {args.gnn_input_dir.resolve()} with pattern '{glob_pattern}'. No evaluations will be performed.")
        return 0

    logger.info(f"Found {len(gnn_files)} GNN files to process for JAX evaluation.")
    
    processed_count = 0
    success_count = 0

    for gnn_file in gnn_files:
        try:
            relative_path = gnn_file.relative_to(args.gnn_input_dir)
            file_specific_output_subdir = jax_eval_step_output_dir / relative_path.parent
        except ValueError:
            file_specific_output_subdir = jax_eval_step_output_dir
        
        file_specific_output_subdir.mkdir(parents=True, exist_ok=True)

        if process_gnn_file_for_jax_eval(gnn_file, file_specific_output_subdir, args.jax_seed, verbose_logging=args.verbose):
            success_count += 1
        processed_count += 1
    
    logger.info(f"Finished processing {processed_count} GNN files. {success_count} JAX evaluations and visualizations generated successfully.")
    
    if processed_count > 0 and success_count < processed_count:
        log_step_warning(logger, f"Some GNN files failed JAX evaluation: {processed_count - success_count} failures out of {processed_count} files")
        return 2 # Partial success / warnings
    elif processed_count > 0 and success_count == processed_count:
        log_step_success(logger, f"All {processed_count} GNN files successfully processed for JAX evaluation")
        return 0 # Full success
    elif processed_count == 0:
        log_step_warning(logger, "No GNN files found to process")
        return 0
    
    return 0

if __name__ == "__main__":
    cli_args = parse_arguments()

    # Enhanced logging setup for standalone execution
    if UTILS_AVAILABLE and cli_args.verbose:
        logger = setup_step_logging("13_discopy_jax_eval", verbose=True)
        logger.info(f"Initialized enhanced logging for step: 13_discopy_jax_eval | STRUCTURED_DATA: {{'step_name': '13_discopy_jax_eval', 'verbose': {cli_args.verbose}, 'structured_logging': True}}")
    else:
        _log_level_standalone = logging.DEBUG if cli_args.verbose else logging.INFO
        logging.basicConfig(level=_log_level_standalone, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if not UTILS_AVAILABLE:
            logger.warning("Enhanced logging not available, using basic logging setup")

    # Set script logger level based on verbose flag
    if cli_args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Script logger '__main__' level set to DEBUG.")
    else:
        logger.setLevel(logging.INFO)

    # Quieten noisy libraries if run standalone
    logging.getLogger('matplotlib').setLevel(logging.WARNING) # If matplotlib is very verbose
    if _JAX_CORE_AVAILABLE_FROM_TRANSLATOR: # Check core JAX availability from translator
        pass # JAX logging is usually not too verbose by default, but can be configured if needed
    
    exit_code = main_discopy_jax_eval_step(cli_args)
    
    if exit_code == 0:
        log_step_success(logger, "Step 13_discopy_jax_eval completed successfully.")
    else:
        log_step_error(logger, f"Step 13_discopy_jax_eval completed with exit code {exit_code}.")
    
    sys.exit(exit_code) 