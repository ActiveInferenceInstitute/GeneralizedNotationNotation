#!/usr/bin/env python3
"""
Pipeline Step 12: GNN to DisCoPy Diagram Transformation

This script takes GNN model specifications as input, translates them into
DisCoPy diagrams, and saves visualizations of these diagrams.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src directory is in Python path for relative imports
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent # Assuming src/12_discopy.py
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "src") not in sys.path: # Also ensure src is in path for src.discopy etc.
    sys.path.insert(0, str(project_root / "src"))

from discopy.drawing import equation_to_gif # type: ignore

try:
    from src.discopy.translator import gnn_file_to_discopy_diagram
    from src.utils.logging_utils import setup_standalone_logging
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_init_fail = logging.getLogger(__name__)
    logger_init_fail.error(f"Failed to import necessary modules for 12_discopy.py: {e}. Ensure translator.py and logging_utils.py are accessible.")
    # Define placeholders if imports fail to allow script to load but fail gracefully
    gnn_file_to_discopy_diagram = None 
    setup_standalone_logging = None

logger = logging.getLogger(__name__) # GNN_Pipeline.12_discopy or __main__

DEFAULT_OUTPUT_SUBDIR = "discopy_diagrams"

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the GNN to DisCoPy script."""
    parser = argparse.ArgumentParser(description="Transforms GNN models to DisCoPy diagrams and saves visualizations.")
    parser.add_argument(
        "--gnn-input-dir",
        type=Path,
        required=True,
        help="Directory containing GNN files (e.g., .gnn.md, .md) to process."
    )
    parser.add_argument(
        "--output-dir", # This is the main pipeline output directory
        type=Path,
        required=True,
        help=f"Main pipeline output directory. DisCoPy diagrams will be saved in a '{DEFAULT_OUTPUT_SUBDIR}' subdirectory here."
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively search for GNN files in the gnn-input-dir. Default: True."
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose (DEBUG level) logging for this script and the translator. Default: False."
    )
    return parser.parse_args()

def process_gnn_file_for_discopy(gnn_file_path: Path, discopy_output_dir: Path, verbose_translator: bool):
    """
    Processes a single GNN file:
    1. Translates it to a DisCoPy diagram using the translator module.
    2. Saves a visualization of the diagram (e.g., as a PNG image).
    """
    logger.info(f"Processing GNN file for DisCoPy: {gnn_file_path.name}")
    if not gnn_file_to_discopy_diagram:
        logger.error("DisCoPy GNN translator not available. Skipping file.")
        return False

    try:
        diagram = gnn_file_to_discopy_diagram(gnn_file_path, verbose=verbose_translator)

        if diagram is None:
            logger.warning(f"No DisCoPy diagram could be generated for {gnn_file_path.name}. Skipping visualization.")
            return False # Indicate that no diagram was made

        # Save diagram visualization
        output_diagram_image_path = discopy_output_dir / (gnn_file_path.stem + "_diagram.png")
        
        # Ensure the immediate parent directory for the image exists (might be a subdir if gnn_file_path had parents)
        output_diagram_image_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Attempting to draw diagram to: {output_diagram_image_path}")
        diagram.draw(path=str(output_diagram_image_path), show_types=True, figsize=(10, 6))
        logger.info(f"Saved DisCoPy diagram visualization to: {output_diagram_image_path}")
        
        # Additionally, try to save the equation (text representation)
        # equation_output_path = discopy_output_dir / (gnn_file_path.stem + "_equation.txt")
        # with open(equation_output_path, 'w', encoding='utf-8') as f_eq:
        #     f_eq.write(str(diagram))
        # logger.info(f"Saved DisCoPy diagram equation to: {equation_output_path}")

        return True # Successfully processed and visualized

    except ImportError as e_draw:
        if "matplotlib" in str(e_draw).lower():
            logger.warning(f"Matplotlib not found, cannot draw diagram for {gnn_file_path.name}. Skipping visualization. Error: {e_draw}")
        else:
            logger.error(f"Missing import for drawing diagram for {gnn_file_path.name}: {e_draw}")
        return False
    except Exception as e:
        logger.error(f"Failed to process GNN file {gnn_file_path.name} for DisCoPy: {e}", exc_info=True)
        return False

def main_discopy_step(args: argparse.Namespace) -> int:
    """Main execution function for the 12_discopy.py pipeline step."""
    # Logger level for this script is set by main.py or standalone __main__ block.
    # The verbose flag here is passed to the translator and process_gnn_file_for_discopy.
    
    if not gnn_file_to_discopy_diagram: # Check if imports succeeded
        logger.critical("Core DisCoPy GNN translator is not available. Aborting 12_discopy.py step.")
        return 1 # Critical failure

    logger.info(f"Starting pipeline step: {Path(__file__).name} - GNN to DisCoPy Transformation")
    logger.info(f"Reading GNN files from: {args.gnn_input_dir.resolve()}")
    
    # Define the specific output directory for this step's artifacts
    discopy_step_output_dir = args.output_dir.resolve() / DEFAULT_OUTPUT_SUBDIR
    try:
        discopy_step_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DisCoPy outputs will be saved in: {discopy_step_output_dir}")
    except OSError as e:
        logger.error(f"Failed to create DisCoPy output directory {discopy_step_output_dir}: {e}")
        return 1 # Cannot proceed without output directory

    if not args.gnn_input_dir.is_dir():
        logger.error(f"GNN input directory not found: {args.gnn_input_dir.resolve()}")
        return 1

    # Discover GNN files (typically .md or .gnn.md)
    glob_pattern = "**/*.md" if args.recursive else "*.md"
    # Also consider .gnn.md specifically
    # More robust discovery might be needed if GNN files have varied extensions.
    gnn_files = list(args.gnn_input_dir.glob(glob_pattern))
    # Add specific .gnn.md files if recursive is False and they weren't caught by *.md
    if not args.recursive:
        gnn_files.extend(list(args.gnn_input_dir.glob("*.gnn.md")))
    gnn_files = sorted(list(set(gnn_files))) # Deduplicate and sort

    if not gnn_files:
        logger.warning(f"No GNN files found in {args.gnn_input_dir.resolve()} with pattern '{glob_pattern}'. No diagrams will be generated.")
        return 0 # Not an error, just no work to do

    logger.info(f"Found {len(gnn_files)} GNN files to process.")
    
    processed_count = 0
    success_count = 0

    for gnn_file in gnn_files:
        # Determine output subdirectory structure based on relative path from gnn_input_dir
        try:
            relative_path = gnn_file.relative_to(args.gnn_input_dir)
            file_specific_output_subdir = discopy_step_output_dir / relative_path.parent
        except ValueError: # Should not happen if gnn_file is from gnn_input_dir.glob
            file_specific_output_subdir = discopy_step_output_dir
        
        file_specific_output_subdir.mkdir(parents=True, exist_ok=True)

        if process_gnn_file_for_discopy(gnn_file, file_specific_output_subdir, verbose_translator=args.verbose):
            success_count += 1
        processed_count += 1
    
    logger.info(f"Finished processing {processed_count} GNN files. {success_count} diagrams generated successfully.")
    
    if processed_count > 0 and success_count < processed_count:
        logger.warning("Some GNN files failed to produce DisCoPy diagrams.")
        return 2 # Partial success / warnings
    elif processed_count > 0 and success_count == processed_count:
        logger.info("All processed GNN files yielded DisCoPy diagrams successfully.")
        return 0 # Full success
    elif processed_count == 0: # Should be caught by the earlier check, but for safety
        return 0 # No files to process is not an error
    
    return 0 # Default success if no other condition met

if __name__ == "__main__":
    cli_args = parse_arguments()

    # Setup logging for standalone execution using the utility function
    if setup_standalone_logging:
        log_level_to_set = logging.DEBUG if cli_args.verbose else logging.INFO
        setup_standalone_logging(level=log_level_to_set, logger_name=__name__)
    else:
        # Fallback basic config if utility function couldn't be imported
        _log_level_standalone = logging.DEBUG if cli_args.verbose else logging.INFO
        logging.basicConfig(level=_log_level_standalone, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.getLogger(__name__).warning("Using fallback basic logging due to missing setup_standalone_logging utility.")

    # Quieten noisy libraries if run standalone
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Call the main logic function for this step
    exit_code = main_discopy_step(cli_args)
    sys.exit(exit_code) 