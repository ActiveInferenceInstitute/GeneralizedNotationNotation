#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 4: Type Checking

This script runs type checking on GNN files by invoking the gnn_type_checker module:
- Processes .md files in the target directory.
- Generates type checking reports (and optionally resource estimation reports).
- Saves reports to a dedicated subdirectory within the main output directory.

Usage:
    python 4_gnn_type_checker.py [options]
    
Options:
    Same as main.py
"""

import sys
from pathlib import Path
import logging # Added logging
import os
import argparse # Added missing import
from gnn_type_checker.resource_estimator import GNNResourceEstimator # Moved import

# Configure basic logging
# The level will be set more specifically in run_type_checker based on verbosity
# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # BasicConfig should be handled by main.py
logger = logging.getLogger(__name__) # Create a logger for this module

# Attempt to import the cli main function from the gnn_type_checker module
try:
    from gnn_type_checker import cli as gnn_type_checker_cli
except ImportError:
    # Add src to path if running in a context where it's not found (e.g. direct execution from src/)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from gnn_type_checker import cli as gnn_type_checker_cli
    except ImportError as e:
        # Use logger for error messages
        logger.error(f"Error: Could not import gnn_type_checker.cli: {e}")
        logger.error("Ensure the gnn_type_checker module is correctly installed or accessible in PYTHONPATH.")
        gnn_type_checker_cli = None


def run_type_checker(target_dir: str, 
                     pipeline_output_dir: str, # Main output dir for the whole pipeline
                     recursive: bool = False, 
                     strict: bool = False, 
                     estimate_resources: bool = False, 
                     verbose: bool = False): # verbose is for this script, cli handles its own verbosity
    """
    Run type checking on GNN files in the target directory using the gnn_type_checker module.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    type_checker_module_logger = logging.getLogger("gnn_type_checker")
    if verbose:
        type_checker_module_logger.setLevel(logging.INFO)
    else:
        type_checker_module_logger.setLevel(logging.WARNING)

    if not gnn_type_checker_cli:
        logger.error("❌ GNN Type Checker CLI module not loaded. Cannot proceed.")
        return False # Indicate failure

    # Define the specific output directory for this step's artifacts
    type_checker_step_output_dir = Path(pipeline_output_dir) / "gnn_type_check"
    type_checker_step_output_dir.mkdir(parents=True, exist_ok=True)
    
    # The main markdown report filename for the type checker
    report_filename = "type_check_report.md" # Hardcoded back

    # Determine project root for relative paths in sub-module reports
    project_root = Path(__file__).resolve().parent.parent

    # Prepare arguments for the gnn_type_checker.cli.main() function
    cli_args = [
        str(target_dir), # input_path, ensure it's a string
        "--output-dir", str(type_checker_step_output_dir),
        "--report-file", report_filename, # This filename will be created inside type_checker_step_output_dir
        "--project-root", str(project_root) # Pass project root to the CLI
    ]
    
    if recursive:
        cli_args.append("--recursive")
    if strict:
        cli_args.append("--strict")
    if estimate_resources:
        cli_args.append("--estimate-resources")
    # The gnn_type_checker.cli.main itself doesn't have a global --verbose flag in its parser
    # It prints its own progress/summary. If verbose for this script is on, we print extra info.

    if verbose:
        # Use logger for informational messages
        logger.info(f"  🐍 Invoking GNN Type Checker module with arguments: {' '.join(cli_args)}")
        logger.info(f"  ℹ️ Target GNN files in: {target_dir}")
        logger.info(f"  ℹ️ Type checker outputs will be in: {type_checker_step_output_dir}")
        logger.info(f"  📝 Main type check report will be: {type_checker_step_output_dir / report_filename}")

    try:
        # Call the main function of the type checker's CLI
        # The cli.main function is expected to return 0 for success, 1 for errors.
        type_checker_cli_exit_code = gnn_type_checker_cli.main(cli_args)
        
        type_checker_successful = (type_checker_cli_exit_code == 0)

        if type_checker_successful:
            if verbose:
                logger.info(f"✅ GNN Type Checker module completed successfully.")
        else:
            # Use logger for error messages
            logger.error(f"❌ GNN Type Checker module reported errors (exit code: {type_checker_cli_exit_code}).")
            logger.error(f"   Check logs and reports in {type_checker_step_output_dir} for details.")
            # Resource estimation will still be attempted by the CLI if the flag was passed,
            # even if type checking reported errors, as per cli.py logic.

        # The GNN Type Checker CLI now handles resource estimation internally if the flag is passed.
        # No need to call it separately here.
            
        return type_checker_successful # Success of the step is based on type checking
            
    except Exception as e:
        # Use logger for critical/exception messages
        logger.critical(f"❌ An unexpected error occurred while running the GNN Type Checker module: {e}", exc_info=verbose) # exc_info=True will log traceback if verbose
        # if verbose: # exc_info=verbose handles this
        #     import traceback
        #     traceback.print_exc()
        return False

def main(args):
    """Main function for the type checking step (Step 4).

    Orchestrates the GNN type checking process by invoking 'run_type_checker'.
    This function is typically called by the main pipeline (`main.py`) with
    a pre-populated args object, or by the standalone execution block which
    parses command-line arguments.

    Args:
        args (argparse.Namespace): 
            Parsed command-line arguments. Expected attributes include:
            target_dir, output_dir, recursive, strict, estimate_resources, verbose.
    """
    # The logger level for this module (__name__) and for "gnn_type_checker"
    # should be set by run_type_checker based on args.verbose, or if run standalone,
    # by the __main__ block. Direct setLevel here is redundant if called by main.py.

    logger.info(f"▶️ Starting Step 4: Type Checking ({Path(__file__).name})...") 
    
    if not run_type_checker(
        args.target_dir, 
        args.output_dir, 
        recursive=args.recursive,
        strict=args.strict if hasattr(args, 'strict') else False,
        estimate_resources=args.estimate_resources if hasattr(args, 'estimate_resources') else False,
        verbose=args.verbose # Pass verbose to run_type_checker to control its specific logging
    ):
        logger.error("❌ Step 4: Type Checking failed.") 
        return 1
    
    if args.verbose:
        logger.info("✅ Step 4: Type Checking complete.") 
    return 0

if __name__ == '__main__':
    # Setup basic logging for standalone execution
    # Adapting the logging setup from other pipeline scripts
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__) # Get a logger for this script when run as main

    parser = argparse.ArgumentParser(description="GNN Type Checker (Standalone)")
    project_root = Path(__file__).resolve().parent.parent
    default_target_dir = project_root / "src" / "gnn" / "examples"
    default_output_dir = project_root / "output"

    parser.add_argument("--target-dir", type=Path, default=default_target_dir,
                        help=f"Target directory for GNN files (default: {default_target_dir.relative_to(project_root)})")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir,
                        help=f"Main output directory (default: {default_output_dir.relative_to(project_root)})")
    parser.add_argument("--recursive", default=True, action=argparse.BooleanOptionalAction,
                        help="Recursively search for GNN files. Enabled by default. Use --no-recursive to disable.")
    parser.add_argument("--strict", action="store_true",
                        help="Enable strict type checking mode.")
    parser.add_argument("--estimate-resources", default=True, action=argparse.BooleanOptionalAction,
                        help="Estimate computational resources (default: True). Use --no-estimate-resources to disable.")
    parser.add_argument("--verbose", default=False, action=argparse.BooleanOptionalAction, # Default to False for standalone, True if called by main.py with verbose
                        help="Enable verbose (DEBUG level) logging. Use --no-verbose to ensure INFO level if LOG_LEVEL is not set.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG
        logger.setLevel(logging.DEBUG) # Set this script's logger to DEBUG
        # If gnn_type_checker_logic has its own logger, set it to DEBUG too
        try:
            from gnn_type_checker import gnn_type_checker_logic
            logging.getLogger(gnn_type_checker_logic.__name__).setLevel(logging.DEBUG)
        except (ImportError, AttributeError):
            logger.debug("Could not set verbose logging for gnn_type_checker_logic module.")
        logger.debug("Verbose logging enabled.")
    
    # Quiet noisy libraries if run standalone
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    sys.exit(main(args)) 