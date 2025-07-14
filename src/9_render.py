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

def render_gnn_files(input_dir: Path, output_dir: Path, recursive: bool = False):
    """Render GNN files to simulation environments."""
    log_step_start(logger, "Rendering GNN files to simulation environments")
    
    if not RENDER_AVAILABLE:
        log_step_error(logger, "Rendering modules not available")
        return False
    
    # Use centralized output directory configuration
    render_output_dir = get_output_dir_for_script("9_render.py", output_dir)
    render_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find GNN files
    pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(input_dir.glob(pattern))
    
    if not gnn_files:
        log_step_warning(logger, f"No GNN files found in {input_dir} using pattern '{pattern}'")
        return False

    logger.info(f"Found {len(gnn_files)} GNN files to render")
    
    successful_renders = 0
    failed_renders = 0
    
    # Define rendering targets - now includes DisCoPy and ActiveInference.jl
    render_targets = [
        ("pymdp", "pymdp"),
        ("rxinfer_toml", "rxinfer"),
        ("discopy_combined", "discopy"),  # Use combined to get both diagram and JAX evaluation
        ("activeinference_combined", "activeinference_jl")  # Use combined to get multiple scripts and analysis suite
    ]
    
    try:
        # Use performance tracking for rendering operations
        with performance_tracker.track_operation("render_all_gnn_files"):
            for gnn_file in gnn_files:
                try:
                    # Parse GNN file (simplified - would normally use proper parser)
                    gnn_spec = {
                        "name": gnn_file.stem,
                        "source_file": str(gnn_file)
                    }
                    
                    # Render to each target format
                    for target_format, output_subdir in render_targets:
                        try:
                            with performance_tracker.track_operation(f"render_{target_format}_{gnn_file.name}"):
                                success, message, artifacts = render_gnn_spec(
                                    gnn_spec, 
                                    target_format, 
                                    render_output_dir / output_subdir
                                )
                                
                            if success:
                                logger.info(f"{target_format} render successful for {gnn_file.name}: {message}")
                                successful_renders += 1
                            else:
                                logger.warning(f"{target_format} render failed for {gnn_file.name}: {message}")
                                failed_renders += 1
                                
                        except Exception as e:
                            log_step_warning(logger, f"{target_format} rendering failed for {gnn_file.name}: {e}")
                            failed_renders += 1
                        
                except Exception as e:
                    log_step_error(logger, f"Failed to process {gnn_file.name}: {e}")
                    failed_renders += 1
        
        # Log results summary
        total_attempts = successful_renders + failed_renders
        logger.info(f"Rendering complete: {successful_renders}/{total_attempts} renders successful")
        
        if successful_renders > 0:
            log_step_success(logger, f"Rendering completed with {successful_renders} successful renders")
            return True
        else:
            log_step_error(logger, "All rendering attempts failed")
            return False
        
    except Exception as e:
        log_step_error(logger, f"Rendering failed: {e}")
        return False

def main(parsed_args: argparse.Namespace):
    """Main function for rendering operations."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("9_render.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Code generation for simulation environments')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get input and output directories
    input_dir = getattr(parsed_args, 'target_dir', None)
    if input_dir is None:
        input_dir = Path("src/gnn/examples")
    elif isinstance(input_dir, str):
        input_dir = Path(input_dir)
        
    output_dir = Path(getattr(parsed_args, 'output_dir', 'output'))
    recursive = getattr(parsed_args, 'recursive', True)
    
    # Validate input directory
    if input_dir is None or not input_dir.exists():
        log_step_error(logger, f"Input directory does not exist: {input_dir}")
        return 1
    
    # Render GNN files
    success = render_gnn_files(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive
    )
    
    if success:
        log_step_success(logger, "Rendering completed successfully")
        return 0
    else:
        log_step_error(logger, "Rendering failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("9_render")
    else:
        # Fallback argument parsing
        parser = argparse.ArgumentParser(description="Code generation for simulation environments")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--recursive", action="store_true",
                          help="Search recursively in subdirectories")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 