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
from utils.pipeline_template import create_standardized_pipeline_script

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

def process_visualization_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized visualization processing function with consistent signature.
    
    Args:
        target_dir: Directory containing GNN files to visualize
        output_dir: Output directory for visualizations
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
        
        # Call the existing generate_visualizations function with updated signature
        success = generate_visualizations(
            logger=logger,
            target_dir=target_dir,
            output_dir=output_dir,
            recursive=recursive,
            verbose=verbose
        )
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Visualization processing failed: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "6_visualization.py",
    process_visualization_standardized,
    "Graph visualization generation"
)

if __name__ == '__main__':
    sys.exit(run_script()) 