#!/usr/bin/env python3

"""
GNN Processing Pipeline - Step 4: Type Checking and Validation

This script performs comprehensive type checking and validation of GNN files:
- Validates GNN syntax and structure
- Checks type consistency
- Estimates computational resource requirements
- Generates validation reports

Usage:
    python 4_gnn_type_checker.py [options]
    (Typically called by main.py)
"""

import argparse
import sys
from pathlib import Path

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
logger = setup_step_logging("4_gnn_type_checker", verbose=False)

# Attempt to import the cli main function from the gnn_type_checker module
try:
    from gnn_type_checker import cli as gnn_type_checker_cli
except ImportError:
    # Add src to path if running in a context where it's not found (e.g. direct execution from src/)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from gnn_type_checker import cli as gnn_type_checker_cli
    except ImportError as e:
        log_step_error(logger, f"Could not import gnn_type_checker.cli: {e}")
        logger.error("Ensure the gnn_type_checker module is correctly installed or accessible in PYTHONPATH.")
        gnn_type_checker_cli = None


def run_type_checker(target_dir: str, 
                     pipeline_output_dir: str, # Main output dir for the whole pipeline
                     recursive: bool = False, 
                     strict: bool = False, 
                     estimate_resources: bool = False, 
                     verbose: bool = False):
    """
    Run type checking on GNN files in the target directory using the gnn_type_checker module.
    """
    log_step_start(logger, f"Running type checking on {target_dir}")

    if not gnn_type_checker_cli:
        log_step_error(logger, "GNN Type Checker CLI module not loaded. Cannot proceed.")
        return False # Indicate failure

    # Define the specific output directory for this step's artifacts
    type_checker_step_output_dir = Path(pipeline_output_dir) / "gnn_type_check"
    type_checker_step_output_dir.mkdir(parents=True, exist_ok=True)
    
    # The main markdown report filename for the type checker
    report_filename = "type_check_report.md"

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

    if verbose:
        logger.info(f"Invoking GNN Type Checker module with arguments: {' '.join(cli_args)}")
        logger.info(f"Target GNN files in: {target_dir}")
        logger.info(f"Type checker outputs will be in: {type_checker_step_output_dir}")
        logger.info(f"Main type check report will be: {type_checker_step_output_dir / report_filename}")

    try:
        # Call the main function of the type checker's CLI
        type_checker_cli_exit_code = gnn_type_checker_cli.main(cli_args)
        
        type_checker_successful = (type_checker_cli_exit_code == 0)

        if type_checker_successful:
            log_step_success(logger, "GNN Type Checker module completed successfully")
        else:
            log_step_error(logger, f"GNN Type Checker module reported errors (exit code: {type_checker_cli_exit_code})")
            logger.error(f"Check logs and reports in {type_checker_step_output_dir} for details.")

        return type_checker_successful
            
    except Exception as e:
        log_step_error(logger, f"Unexpected error occurred while running the GNN Type Checker module: {e}")
        if verbose:
            logger.exception("Full traceback:")
        return False

def main(args):
    """
    Main function for Step 4: Type Checking pipeline.
    
    Args:
        args: argparse.Namespace with target_dir, output_dir, recursive, verbose, etc.
    """
    # Update logger verbosity based on args
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
    
    log_step_start(logger, "Starting Step 4: Type Checking")
    
    # Validate and create type checker output directory
    base_output_dir = Path(args.output_dir)
    if not validate_output_directory(base_output_dir, "gnn_type_check"):
        log_step_error(logger, "Failed to create type check output directory")
        return 1
    
    # Call the type checking function
    try:
        result = run_type_checker(
            target_dir=str(args.target_dir),
            pipeline_output_dir=str(args.output_dir),
            recursive=args.recursive,
            strict=getattr(args, 'strict', False),
            estimate_resources=getattr(args, 'estimate_resources', False),
            verbose=args.verbose
        )
        
        if result:
            log_step_success(logger, "Type checking completed successfully")
            return 0
        else:
            log_step_warning(logger, "Type checking completed with warnings")
            return 0
            
    except Exception as e:
        log_step_error(logger, f"Type checking failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Type check and validate GNN files")
    
    # Define defaults for standalone execution
    script_file_path = Path(__file__).resolve()
    project_root = script_file_path.parent.parent
    default_target_dir = project_root / "src" / "gnn" / "examples"
    default_output_dir = project_root / "output"

    parser.add_argument("--target-dir", type=Path, default=default_target_dir,
                       help="Target directory containing GNN files")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir,
                       help="Main pipeline output directory")
    parser.add_argument("--recursive", action='store_true',
                       help="Search for GNN files recursively")
    parser.add_argument("--strict", action='store_true',
                       help="Enable strict type checking")
    parser.add_argument("--estimate-resources", action='store_true',
                       help="Enable resource estimation")
    parser.add_argument("--verbose", action='store_true',
                       help="Enable verbose output")

    parsed_args = parser.parse_args()

    # Update logger for standalone execution
    if parsed_args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    sys.exit(main(parsed_args)) 