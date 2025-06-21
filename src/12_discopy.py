#!/usr/bin/env python3
"""
Pipeline Step 12: GNN to DisCoPy Diagram Transformation

This script takes GNN model specifications as input, translates them into
DisCoPy diagrams, and saves visualizations of these diagrams.
"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Ensure src directory is in Python path for relative imports
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent # Assuming src/12_discopy.py

# Ensure project_root is at the beginning of sys.path for `import src.xxx`
if str(project_root) in sys.path:
    sys.path.remove(str(project_root))
sys.path.insert(0, str(project_root))

# Ensure src is also in sys.path, potentially for other types of imports or checks,
# but after project_root to prioritize `src.` pattern from root.
if str(project_root / "src") in sys.path:
    sys.path.remove(str(project_root / "src"))
sys.path.insert(1, str(project_root / "src"))

# Import streamlined utilities and translator
try:
    from utils import (
        setup_step_logging,
        log_step_start,
        log_step_success, 
        log_step_warning,
        log_step_error,
        validate_output_directory,
        EnhancedArgumentParser,
        PipelineLogger,
        UTILS_AVAILABLE
    )
    
    # Initialize logger for this step  
    logger = setup_step_logging("12_discopy", verbose=False)  # Will be updated based on args
    
    # Import translator function
    from src.discopy_translator_module.translator import gnn_file_to_discopy_diagram
    
except ImportError as e:
    # Fallback to basic logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import necessary modules for 12_discopy.py: {e}")
    UTILS_AVAILABLE = False
    
    # Define placeholders if imports fail
    gnn_file_to_discopy_diagram = None

DEFAULT_OUTPUT_SUBDIR = "discopy_gnn"

def process_gnn_file_for_discopy(gnn_file_path: Path, discopy_output_dir: Path, verbose_translator: bool):
    """
    Processes a single GNN file:
    1. Translates it to a DisCoPy diagram using the translator module.
    2. Saves a visualization of the diagram (e.g., as a PNG image).
    """
    logger.info(f"Processing GNN file for DisCoPy: {gnn_file_path.name}")
    if not gnn_file_to_discopy_diagram:
        log_step_error(logger, "DisCoPy GNN translator not available. Skipping file.")
        return False

    try:
        diagram = gnn_file_to_discopy_diagram(gnn_file_path, verbose=verbose_translator)

        if diagram is None:
            log_step_warning(logger, f"No DisCoPy diagram could be generated for {gnn_file_path.name}. Skipping visualization.")
            return False

        # Save diagram visualization
        output_diagram_image_path = discopy_output_dir / (gnn_file_path.stem + "_diagram.png")
        
        # Ensure the immediate parent directory for the image exists
        output_diagram_image_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Attempting to draw diagram to: {output_diagram_image_path}")
        diagram.draw(path=str(output_diagram_image_path), show_types=True, figsize=(10, 6))
        logger.info(f"Saved DisCoPy diagram visualization to: {output_diagram_image_path}")

        return True

    except ImportError as e_draw:
        if "matplotlib" in str(e_draw).lower():
            log_step_warning(logger, f"Matplotlib not found, cannot draw diagram for {gnn_file_path.name}. Skipping visualization. Error: {e_draw}")
        else:
            log_step_error(logger, f"Missing import for drawing diagram for {gnn_file_path.name}: {e_draw}")
        return False
    except Exception as e:
        log_step_error(logger, f"Failed to process GNN file {gnn_file_path.name} for DisCoPy: {e}")
        return False

def main(args) -> int:
    """Main execution function for the 12_discopy.py pipeline step."""
    
    # Update logger verbosity based on args
    if UTILS_AVAILABLE and hasattr(args, 'verbose') and args.verbose:
        PipelineLogger.set_verbosity(True)

    if not gnn_file_to_discopy_diagram:
        log_step_error(logger, "Core DisCoPy GNN translator is not available. Aborting 12_discopy.py step.")
        return 1

    log_step_start(logger, f"Starting pipeline step: {Path(__file__).name} - GNN to DisCoPy Transformation")
    logger.info(f"Reading GNN files from: {args.gnn_input_dir.resolve()}")
    
    # Define the specific output directory for this step's artifacts
    discopy_step_output_dir = args.output_dir.resolve() / DEFAULT_OUTPUT_SUBDIR
    try:
        discopy_step_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DisCoPy outputs will be saved in: {discopy_step_output_dir}")
    except OSError as e:
        log_step_error(logger, f"Failed to create DisCoPy output directory {discopy_step_output_dir}: {e}")
        return 1

    if not args.gnn_input_dir.is_dir():
        log_step_error(logger, f"GNN input directory not found: {args.gnn_input_dir.resolve()}")
        return 1

    # Discover GNN files
    glob_pattern = "**/*.md" if args.recursive else "*.md"
    gnn_files = list(args.gnn_input_dir.glob(glob_pattern))
    if not args.recursive:
        gnn_files.extend(list(args.gnn_input_dir.glob("*.gnn.md")))
    gnn_files = sorted(list(set(gnn_files)))

    if not gnn_files:
        log_step_warning(logger, f"No GNN files found in {args.gnn_input_dir.resolve()} with pattern '{glob_pattern}'. No diagrams will be generated.")
        return 0

    logger.info(f"Found {len(gnn_files)} GNN files to process.")
    
    processed_count = 0
    success_count = 0

    for gnn_file in gnn_files:
        # Determine output subdirectory structure based on relative path from gnn_input_dir
        try:
            relative_path = gnn_file.relative_to(args.gnn_input_dir)
            file_specific_output_subdir = discopy_step_output_dir / relative_path.parent
        except ValueError:
            file_specific_output_subdir = discopy_step_output_dir
        
        file_specific_output_subdir.mkdir(parents=True, exist_ok=True)

        if process_gnn_file_for_discopy(gnn_file, file_specific_output_subdir, verbose_translator=args.verbose):
            success_count += 1
        processed_count += 1
    
    logger.info(f"Finished processing {processed_count} GNN files. {success_count} diagrams generated successfully.")
    
    if processed_count > 0 and success_count < processed_count:
        log_step_warning(logger, f"Some GNN files failed to produce DisCoPy diagrams: {success_count}/{processed_count} successful")
        return 2 # Partial success / warnings
    elif processed_count > 0 and success_count == processed_count:
        log_step_success(logger, f"All {processed_count} processed GNN files yielded DisCoPy diagrams successfully.")
        return 0
    elif processed_count == 0:
        return 0
    
    return 0

if __name__ == "__main__":
    # Enhanced argument parsing with fallback
    if UTILS_AVAILABLE:
        try:
            cli_args = EnhancedArgumentParser.parse_step_arguments("12_discopy")
            # Handle argument mapping if needed by the enhanced parser
            if not hasattr(cli_args, 'gnn_input_dir') or cli_args.gnn_input_dir is None:
                if hasattr(cli_args, 'discopy_gnn_input_dir') and cli_args.discopy_gnn_input_dir is not None:
                    cli_args.gnn_input_dir = cli_args.discopy_gnn_input_dir
                elif hasattr(cli_args, 'target_dir') and cli_args.target_dir is not None:
                    cli_args.gnn_input_dir = cli_args.target_dir
            # Ensure recursive attribute exists
            if not hasattr(cli_args, 'recursive'):
                cli_args.recursive = True
        except Exception as e:
            log_step_error(logger, f"Failed to parse arguments with enhanced parser: {e}")
            # Fallback to basic parser
            import argparse
            parser = argparse.ArgumentParser(description="Transforms GNN models to DisCoPy diagrams and saves visualizations.")
            
            # Input directory arguments (with precedence handling)
            parser.add_argument(
                "--gnn-input-dir",
                type=Path,
                help="Directory containing GNN files (e.g., .gnn.md, .md) to process."
            )
            parser.add_argument(
                "--target-dir",
                type=Path,
                help="Alternative name for the directory containing GNN files (compatible with main.py)."
            )
            parser.add_argument(
                "--discopy-gnn-input-dir",
                type=Path,
                help="Directory containing GNN files for DisCoPy processing (pipeline compatibility)."
            )
            
            parser.add_argument(
                "--output-dir", # This is the main pipeline output directory
                type=Path,
                required=True,
                help=f"Main pipeline output directory. DisCoPy diagrams will be saved in a '{DEFAULT_OUTPUT_SUBDIR}' subdirectory here."
            )
            parser.add_argument(
                "--recursive",
                action="store_true",
                default=True,
                help="Recursively search for GNN files in the gnn-input-dir. Default: True."
            )
            parser.add_argument(
                "--verbose",
                action="store_true",
                default=False,
                help="Enable verbose (DEBUG level) logging for this script and the translator. Default: False."
            )
            
            cli_args = parser.parse_args()
            
            # Handle argument mapping with precedence: discopy_gnn_input_dir > gnn_input_dir > target_dir
            if cli_args.gnn_input_dir is None:
                if cli_args.discopy_gnn_input_dir is not None:
                    cli_args.gnn_input_dir = cli_args.discopy_gnn_input_dir
                elif cli_args.target_dir is not None:
                    cli_args.gnn_input_dir = cli_args.target_dir
                else:
                    parser.error("One of --gnn-input-dir, --target-dir, or --discopy-gnn-input-dir is required")
    else:
        # Fallback to basic parser
        import argparse
        parser = argparse.ArgumentParser(description="Transforms GNN models to DisCoPy diagrams and saves visualizations.")
        
        # Input directory arguments (with precedence handling)
        parser.add_argument(
            "--gnn-input-dir",
            type=Path,
            help="Directory containing GNN files (e.g., .gnn.md, .md) to process."
        )
        parser.add_argument(
            "--target-dir",
            type=Path,
            help="Alternative name for the directory containing GNN files (compatible with main.py)."
        )
        parser.add_argument(
            "--discopy-gnn-input-dir",
            type=Path,
            help="Directory containing GNN files for DisCoPy processing (pipeline compatibility)."
        )
        
        parser.add_argument(
            "--output-dir", # This is the main pipeline output directory
            type=Path,
            required=True,
            help=f"Main pipeline output directory. DisCoPy diagrams will be saved in a '{DEFAULT_OUTPUT_SUBDIR}' subdirectory here."
        )
        parser.add_argument(
            "--recursive",
            action="store_true",
            default=True,
            help="Recursively search for GNN files in the gnn-input-dir. Default: True."
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Enable verbose (DEBUG level) logging for this script and the translator. Default: False."
        )
        
        cli_args = parser.parse_args()
        
        # Handle argument mapping with precedence: discopy_gnn_input_dir > gnn_input_dir > target_dir
        if cli_args.gnn_input_dir is None:
            if cli_args.discopy_gnn_input_dir is not None:
                cli_args.gnn_input_dir = cli_args.discopy_gnn_input_dir
            elif cli_args.target_dir is not None:
                cli_args.gnn_input_dir = cli_args.target_dir
            else:
                parser.error("One of --gnn-input-dir, --target-dir, or --discopy-gnn-input-dir is required")

    # Update logger for standalone execution
    if cli_args.verbose:
        PipelineLogger.set_verbosity(True)

    # Call the main logic function for this step
    exit_code = main(cli_args)
    sys.exit(exit_code) 