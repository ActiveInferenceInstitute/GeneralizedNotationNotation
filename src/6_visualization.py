#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 6: Visualization

This script generates visualizations of GNN models and their components.

Usage:
    python 6_visualization.py [options]
    (Typically called by main.py)
"""

import sys
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

from visualization.visualizer import generate_visualizations

# Initialize logger for this step
logger = setup_step_logging("6_visualization", verbose=False)

# Import visualization functionality
# try:
#     from visualization import GNNVisualizer
#     from visualization.matrix_visualizer import MatrixVisualizer
#     from visualization.ontology_visualizer import OntologyVisualizer
#     logger.debug("Successfully imported visualization modules")
#     visualizer = GNNVisualizer  # Create alias for backwards compatibility
# except ImportError as e:
#     log_step_error(logger, f"Could not import visualization modules: {e}")
#     GNNVisualizer = None
#     visualizer = None
#     MatrixVisualizer = None
#     OntologyVisualizer = None

def main(parsed_args: argparse.Namespace):
    """Main function for visualization generation."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("6_visualization.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Graph visualization generation')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Generate visualizations
    success = generate_visualizations(
        logger=logger,
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        recursive=getattr(parsed_args, 'recursive', False)
    )
    
    if success:
        log_step_success(logger, "Visualization generation completed successfully")
        return 0
    else:
        log_step_error(logger, "Visualization generation failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("6_visualization")
    else:
        # Fallback argument parsing
        parser = argparse.ArgumentParser(description="Graph visualization generation")
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