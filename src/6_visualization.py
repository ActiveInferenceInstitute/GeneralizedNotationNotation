#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 6: Visualization

This script generates visualizations for GNN files by invoking the visualization module:
- Processes .md files in the target directory.
- Creates visual representations of the GNN models.
- Saves visualizations to a dedicated subdirectory within the main output directory.

Usage:
    python 6_visualization.py [options]
    
Options:
    Same as main.py
"""

import sys
from pathlib import Path
import logging # Import logging

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# Attempt to import the cli main function from the visualization module
try:
    from visualization import cli as visualization_cli
except ImportError:
    # Add src to path if running in a context where it's not found
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from visualization import cli as visualization_cli
    except ImportError as e:
        logger.error(f"Error: Could not import visualization.cli: {e}") # Changed from print
        logger.error("Ensure the visualization module is correctly installed or accessible in PYTHONPATH.") # Changed from print
        visualization_cli = None

def run_visualization(target_dir: str, 
                        pipeline_output_dir: str, # Main output dir for the whole pipeline 
                        recursive: bool = False, 
                        verbose: bool = False):
    """Generate visualizations for GNN files using the visualization module."""
    # Set logging level for this script's logger
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Set logging level for noisy libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # Configure logging level for the visualization module based on pipeline verbosity
    viz_module_logger = logging.getLogger("visualization") # Get the parent logger for the module
    if verbose:
        viz_module_logger.setLevel(logging.INFO) # Allow its INFO messages if pipeline is verbose
    else:
        viz_module_logger.setLevel(logging.WARNING) # Silence its INFO messages by default

    if not visualization_cli:
        logger.error("❌🎨 Visualization CLI module not loaded. Cannot proceed.") # Changed from print
        return False # Indicate failure

    viz_step_output_dir = Path(pipeline_output_dir) / "gnn_examples_visualization"
    
    logger.info("🖼️ Preparing to generate GNN visualizations...") # Was print if verbose
    logger.debug(f"  🎯 Target GNN files in: {Path(target_dir).resolve()}") # Was print if verbose
    logger.debug(f"  ել Output visualizations will be in: {viz_step_output_dir.resolve()}") # Was print if verbose
    if recursive:
        logger.debug("  🔄 Recursive mode: Enabled") # Was print if verbose
    else:
        logger.debug("  ➡️ Recursive mode: Disabled") # Was print if verbose

    # Determine project root for relative paths in sub-module reports
    project_root = Path(__file__).resolve().parent.parent

    cli_args = [
        target_dir, 
        "--output-dir", str(viz_step_output_dir),
        "--project-root", str(project_root) # Pass project root to the CLI
    ]
    
    if recursive:
        cli_args.append("--recursive")
        
    logger.debug(f"  🐍 Invoking GNN Visualization module (visualization.cli.main)") # Was print if verbose
    logger.debug(f"     Arguments: {' '.join(cli_args)}") # Was print if verbose
    
    try:
        exit_code = visualization_cli.main(cli_args)
        
        if exit_code == 0:
            logger.info("✅ GNN Visualization module completed successfully.") # Was print if verbose
            logger.debug(f"  🖼️ Visualizations should be available in: {viz_step_output_dir.resolve()}") # Was print if verbose
            # Check if the directory was created and if it has content
            if viz_step_output_dir.exists() and any(viz_step_output_dir.iterdir()):
                num_items = len(list(viz_step_output_dir.glob('**/*'))) # Counts files and dirs
                logger.debug(f"  📊 Found {num_items} items (files/directories) in the output directory.") # Was print if verbose
            else: # This case was only logged if verbose before, now always if dir is empty
                logger.warning(f"⚠️ Output directory {viz_step_output_dir.resolve()} is empty or was not created as expected by the visualization module.") # Was print if verbose
            return True
        else:
            logger.error(f"❌🎨 GNN Visualization module (visualization.cli.main) reported errors (exit code: {exit_code}).") # Changed from print
            return False
            
    except Exception as e:
        logger.error(f"❌🎨 An unexpected error occurred while running the GNN Visualization module: {e}", exc_info=verbose) # Changed from print, added exc_info=verbose
        # if verbose: # Handled by exc_info=verbose
        #     import traceback
        #     traceback.print_exc()
        return False

def main(args):
    """Main function for the visualization step (Step 6).

    This function serves as the entry point when 6_visualization.py is called.
    It logs the start of the step and invokes the `run_visualization` function
    with the parsed arguments.

    Args:
        args (argparse.Namespace): 
            Parsed command-line arguments from `main.py` or standalone execution.
            Expected attributes include: target_dir, output_dir, recursive, verbose.
    """
    # Set this script's logger level based on pipeline's args.verbose
    # This is typically handled by main.py for child modules, and run_visualization also sets levels.
    # if args.verbose:
    #     logger.setLevel(logging.DEBUG)
    # else:
    #     logger.setLevel(logging.INFO)

    logger.info(f"▶️ Starting Step 6: Visualization ({Path(__file__).name})") 
    logger.debug(f"  Parsing options:") # Was print if verbose
    logger.debug(f"    Target directory/file: {args.target_dir}") # Was print if verbose
    logger.debug(f"    Pipeline output directory: {args.output_dir}") # Was print if verbose
    logger.debug(f"    Recursive: {args.recursive}") # Was print if verbose
    logger.debug(f"    Verbose: {args.verbose}") # Was print if verbose

    if not run_visualization(args.target_dir, 
                             args.output_dir, 
                             args.recursive if hasattr(args, 'recursive') else False, 
                             args.verbose):
        logger.error(f"❌ Step 6: Visualization ({Path(__file__).name}) FAILED.") # Changed from print
        return 1
    
    logger.info(f"✅ Step 6: Visualization ({Path(__file__).name}) - COMPLETED") # Was print if verbose
    return 0

if __name__ == '__main__':
    import argparse # Ensure imported
    # This script is called by main.py with arguments.
    # Set up argparse to receive these arguments.
    parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 6: Visualization")
    parser.add_argument(
        "--target-dir",
        type=str, 
        required=True, 
        help="Target GNN file or directory for visualization (passed by main.py)."
    )
    parser.add_argument(
        "--output-dir",
        type=str, 
        required=True, 
        help="Main pipeline output directory (passed by main.py)."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process subdirectories (passed by main.py if set)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for this script (passed by main.py if set)."
    )
    # Add --project-root, as run_visualization passes it to visualization_cli
    # main.py doesn't explicitly pass --project-root to 6_visualization.py currently.
    # However, run_visualization calculates it. For consistency if run standalone and wanting to mimic pipeline call:
    # We can let run_visualization calculate it, or expect it here if main.py were to pass it.
    # For now, stick to what main.py explicitly passes to this script via command line.

    parsed_script_args = parser.parse_args() # Parses arguments from sys.argv, set by main.py

    script_logger_instance = logging.getLogger(__name__) # Use the module-level logger
    log_level_to_set = logging.DEBUG if parsed_script_args.verbose else logging.INFO
    
    # Configure logging for this script instance if it's being run directly
    # and doesn't seem to have handlers (e.g. from a parent orchestrator like main.py)
    if not script_logger_instance.hasHandlers() or not any(isinstance(h, logging.StreamHandler) for h in script_logger_instance.handlers):
        logging.basicConfig(
            level=log_level_to_set,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
            force=True # Override any root config
        )
        script_logger_instance.info(f"Configured basic logging for {__name__} (standalone run).")
    else:
        script_logger_instance.setLevel(log_level_to_set)

    # Adjust visualization module logger level for standalone verbose runs
    if parsed_script_args.verbose:
        viz_module_logger_standalone = logging.getLogger("visualization")
        viz_module_logger_standalone.setLevel(logging.INFO) # Match run_visualization logic

    # Call the script's main logic function with the parsed arguments
    exit_code = main(parsed_script_args)
    sys.exit(exit_code) 