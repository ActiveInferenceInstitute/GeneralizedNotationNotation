#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 9: Render

This script renders GNN specifications to target simulation environments.

Usage:
    python 9_render.py [options]
    (Typically called by main.py)
"""

import sys
import logging
from pathlib import Path
from typing import Dict
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

from gnn.parsers.markdown_parser import MarkdownGNNParser 
from render.renderer import render_gnn_files
from utils.pipeline_template import create_standardized_pipeline_script

# Initialize logger for this step
logger = setup_step_logging("9_render", verbose=False)

# Import rendering functionality
try:
    from render.render import render_gnn_spec
    from render.pymdp.pymdp_renderer import render_gnn_to_pymdp
    from render.rxinfer import render_gnn_to_rxinfer_toml
    logger.debug("Successfully imported rendering modules")
    RENDER_AVAILABLE = True
except ImportError as e:
    log_step_error(logger, f"Could not import rendering modules: {e}")
    render_gnn_spec = None
    render_gnn_to_pymdp = None
    render_gnn_to_rxinfer_toml = None
    RENDER_AVAILABLE = False

def process_rendering_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized rendering processing function with consistent signature.
    
    Args:
        target_dir: Directory containing GNN files to render
        output_dir: Output directory for rendered code
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
        
        # Validate input directory
        if not target_dir.exists():
            log_step_error(logger, f"Input directory does not exist: {target_dir}")
            return False
        
        # Call the existing render_gnn_files function with updated signature
        success = render_gnn_files(
            target_dir=target_dir,
            output_dir=output_dir,
            recursive=recursive,
            logger=logger,
            verbose=verbose
        )
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Rendering processing failed: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "9_render.py",
    process_rendering_standardized,
    "Code generation for simulation environments"
)

if __name__ == '__main__':
    sys.exit(run_script()) 