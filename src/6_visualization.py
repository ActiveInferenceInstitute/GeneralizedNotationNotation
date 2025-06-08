#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 6: Visualization

This script generates visualization outputs based on the pipeline state:
- Graph visualizations of GNN models
- Statistical charts and summary plots
- Visual documentation of pipeline outputs

Usage:
    python 6_visualization.py [options]
    (Typically called by main.py)
"""

import argparse
import os
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
logger = setup_step_logging("6_visualization", verbose=False)

# Attempt to import the cli main function from the visualization module
try:
    from visualization import cli as visualization_cli
except ImportError:
    # Add src to path if running in a context where it's not found
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from visualization import cli as visualization_cli
    except ImportError as e:
        log_step_error(logger, f"Could not import visualization.cli: {e}")
        logger.error("Ensure the visualization module is correctly installed or accessible in PYTHONPATH.")
        visualization_cli = None

def run_visualization(target_dir: str, 
                        pipeline_output_dir: str,
                        recursive: bool = False, 
                        verbose: bool = False):
    """Generate visualizations for GNN files using the visualization module."""
    
    log_step_start(logger, f"Generating visualizations for {target_dir}")

    if not visualization_cli:
        log_step_error(logger, "Visualization CLI module not loaded. Cannot proceed.")
        return False

    viz_step_output_dir = Path(pipeline_output_dir) / "visualization"
    
    logger.info("Preparing to generate GNN visualizations...")
    logger.debug(f"Target GNN files in: {Path(target_dir).resolve()}")
    logger.debug(f"Output visualizations will be in: {viz_step_output_dir.resolve()}")
    logger.debug(f"Recursive mode: {'Enabled' if recursive else 'Disabled'}")

    # Determine project root for relative paths in sub-module reports
    project_root = Path(__file__).resolve().parent.parent

    cli_args = [
        target_dir, 
        "--output-dir", str(viz_step_output_dir),
        "--project-root", str(project_root)
    ]
    
    if recursive:
        cli_args.append("--recursive")
        
    logger.debug(f"Invoking GNN Visualization module (visualization.cli.main)")
    logger.debug(f"Arguments: {' '.join(cli_args)}")
    
    try:
        exit_code = visualization_cli.main(cli_args)
        
        if exit_code == 0:
            logger.info("GNN Visualization module completed successfully.")
            logger.debug(f"Visualizations should be available in: {viz_step_output_dir.resolve()}")
            
            # Check if the directory was created and if it has content
            if viz_step_output_dir.exists() and any(viz_step_output_dir.iterdir()):
                num_items = len(list(viz_step_output_dir.glob('**/*')))
                logger.debug(f"Found {num_items} items (files/directories) in the output directory.")
                log_step_success(logger, f"Successfully generated visualizations with {num_items} output items")
            else:
                log_step_warning(logger, f"Output directory {viz_step_output_dir.resolve()} is empty or was not created as expected by the visualization module.")
            return True
        else:
            log_step_error(logger, f"GNN Visualization module reported errors (exit code: {exit_code})")
            return False
            
    except Exception as e:
        log_step_error(logger, f"Unexpected error occurred while running the GNN Visualization module: {e}")
        if verbose:
            logger.exception("Full traceback:")
        return False

def main(args):
    """
    Main function for Step 6: Visualization pipeline.
    
    Args:
        args: argparse.Namespace with target_dir, output_dir, recursive, verbose
    """
    # Update logger verbosity based on args
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
    
    log_step_start(logger, "Starting Step 6: Visualization")
    
    # Validate and create visualization output directory
    base_output_dir = Path(args.output_dir)
    if not validate_output_directory(base_output_dir, "visualization"):
        log_step_error(logger, "Failed to create visualization output directory")
        return 1
    
    # Call the visualization function
    try:
        result = run_visualization(
            target_dir=str(args.target_dir),
            pipeline_output_dir=str(args.output_dir),
            recursive=args.recursive,
            verbose=args.verbose
        )
        
        if result:
            log_step_success(logger, "Visualization generation completed successfully")
            return 0
        else:
            log_step_warning(logger, "Visualization generation completed with warnings")
            return 0
            
    except Exception as e:
        log_step_error(logger, f"Visualization generation failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate visualizations for GNN pipeline")
    
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
    parser.add_argument("--verbose", action='store_true',
                       help="Enable verbose output")

    parsed_args = parser.parse_args()

    # Update logger for standalone execution
    if parsed_args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    sys.exit(main(parsed_args)) 