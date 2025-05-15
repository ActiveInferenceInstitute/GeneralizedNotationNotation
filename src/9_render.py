"""
Pipeline step for rendering GNN specifications.

This script calls the main rendering logic defined in the src/render/render.py module.
"""

import argparse
import logging
import sys
from pathlib import Path
import glob # Added for file searching
import os

# Ensure the GNN_Pipeline logger is used if this script is run in that context.
# If run standalone, it will use the root logger configured by its own main.
logger = logging.getLogger(__name__) # If part of GNN_Pipeline, __name__ will be '9_render'
                                     # If standalone, __name__ will be '__main__'


def main(args: argparse.Namespace) -> int:
    """
    Entry point for the GNN rendering pipeline step (Step 9).

    This function imports and calls the main function from the `render.py` module
    (expected in `src/render/`). It identifies GNN specification files
    (typically JSON exports from Step 5) in a subdirectory of `args.output_dir`
    (conventionally `args.output_dir/gnn_exports/`) and attempts to render them
    into specified simulator formats (e.g., pymdp, rxinfer).

    Args:
        args (argparse.Namespace):
            Parsed command-line arguments. Expected attributes include:
            output_dir (PathLike): Main pipeline output directory.
            recursive (bool): Whether to search for GNN spec files recursively.
            verbose (bool): Flag for verbose logging.

    Returns:
        int: 0 for success, 1 for failure.
    """
    # Set logging level for noisy libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    current_script_path = Path(__file__).resolve()
    # Add 'src' to sys.path to allow 'from render import render'
    # This assumes '9_render.py' is in 'src/' and 'render.py' is in 'src/render/'
    src_dir = current_script_path.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from render import render as render_module
        logger.info("Successfully imported render module.")
    except ImportError as e:
        logger.error(f"Failed to import the render module from {src_dir / 'render'}: {e}")
        logger.error("Please ensure 'render.py' exists in the 'src/render/' directory.")
        logger.error(f"Current sys.path: {sys.path}")
        return 1

    logger.info(f"Executing render step with arguments from main pipeline: {args}")

    # The render step should process GNN files exported by a previous pipeline step (e.g., 5_export.py).
    # These exported GNN specifications (expected to be *.json or *.gnn files)
    # are typically found in a subdirectory of the main pipeline output directory.
    # We'll use "gnn_exports" as the conventional name for this subdirectory.
    gnn_export_subdir_name = "gnn_exports"
    base_target_dir = Path(args.output_dir).resolve() / gnn_export_subdir_name
    logger.info(f"Render step will target GNN specifications from: {base_target_dir}")

    # Specific output subdirectory for this rendering step
    step_output_dir_name = "gnn_rendered_simulators" # Changed from "gnn_renders" for clarity
    base_output_dir = Path(args.output_dir).resolve() / step_output_dir_name
    
    supported_formats = ["pymdp", "rxinfer"]
    overall_success = True
    files_processed_count = 0

    # Determine glob pattern for GNN files
    # Common patterns could be *.gnn.json or simply *.json if the directory only contains GNN specs.
    # For now, let's assume *.json as a common case for GNN specs.
    # Users can place their GNN JSON files directly in gnn/examples or subdirectories.
    glob_pattern = "*.json" 
    
    if not base_target_dir.is_dir():
        logger.error(f"Target directory for GNN specs not found or is not a directory: {base_target_dir}")
        return 1

    logger.info(f"Searching for GNN specification files ({glob_pattern}) in {base_target_dir} (recursive: {args.recursive})")

    if args.recursive:
        gnn_files = list(base_target_dir.rglob(glob_pattern))
    else:
        gnn_files = list(base_target_dir.glob(glob_pattern))

    if not gnn_files:
        logger.warning(f"No GNN specification files ({glob_pattern}) found in {base_target_dir}.")
        # Still return 0 as this isn't an error of the script itself, but no work to do.
        return 0

    logger.info(f"Found {len(gnn_files)} GNN specification files to process.")

    for gnn_file_path in gnn_files:
        files_processed_count +=1
        logger.info(f"Processing GNN specification: {gnn_file_path}")
        
        relative_path_from_base_target = gnn_file_path.relative_to(base_target_dir)

        for target_format in supported_formats:
            logger.info(f"  Rendering to format: {target_format}")

            # Construct specific output directory for this file and format
            # e.g., ../output/gnn_rendered_simulators/pymdp/subdir_if_any/
            render_output_subdir = base_output_dir / target_format / relative_path_from_base_target.parent
            render_output_subdir.mkdir(parents=True, exist_ok=True)

            # Determine output filename for the render_module
            # render_module.main will add _rendered.ext if --output_filename is not given
            # or use the provided name. We can just pass the stem.
            output_file_stem = gnn_file_path.stem
            if output_file_stem.endswith(".gnn"): # e.g., my_model.gnn -> my_model
                output_file_stem = Path(output_file_stem).stem

            render_cli_args = [
                str(gnn_file_path),
                str(render_output_subdir),
                target_format,
                "--output_filename", output_file_stem # Pass the stem, render.py handles adding _rendered.ext
            ]

            if args.verbose: # Pass verbose flag from main pipeline args
                render_cli_args.append("--verbose")
            
            logger.debug(f"    Calling render_module.main with args: {render_cli_args}")
            
            try:
                render_result = render_module.main(cli_args=render_cli_args)
                if render_result == 0:
                    logger.info(f"    Successfully rendered {gnn_file_path.name} to {target_format} in {render_output_subdir}")
                else:
                    logger.error(f"    Failed to render {gnn_file_path.name} to {target_format}. Exit code: {render_result}")
                    overall_success = False
            except Exception as e:
                logger.error(f"    Exception during rendering {gnn_file_path.name} to {target_format}: {e}", exc_info=True)
                overall_success = False
                
    if files_processed_count > 0 and overall_success:
        logger.info(f"Render step completed successfully for {files_processed_count} GNN file(s).")
    elif files_processed_count > 0 and not overall_success:
        logger.error(f"Render step completed for {files_processed_count} GNN file(s), but some renderings failed.")
    elif files_processed_count == 0: # This case is already handled if gnn_files is empty, but for completeness
        logger.info("Render step completed. No GNN files were processed.")
        
    return 0 if overall_success else 1

if __name__ == "__main__":
    # Basic configuration for running this script standalone
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level_for_standalone = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level_for_standalone,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True # Override any existing root logger configuration for standalone run
    )

    # Define arguments that 9_render.py's main() function expects when called by main.py
    # or for a similar standalone test.
    # The main() function of 9_render.py itself scans a directory derived from output_dir.
    parser = argparse.ArgumentParser(description="GNN Rendering Step - Standalone Directory Processing Mode")
    parser.add_argument(
        "--output-dir", 
        default="../output", # This script's main() derives scan path from this
        type=Path,
        help="Main pipeline output directory. Render step scans 'output_dir/gnn_exports'. Default: ../output"
    )
    parser.add_argument(
        "--recursive", 
        action="store_true", 
        default=False, # Default for standalone, main.py passes it if true.
        help="Recursively scan for GNN spec files in 'output_dir/gnn_exports'."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        default=(log_level_for_standalone == logging.DEBUG),
        help="Enable verbose logging for this script."
    )
    # Note: The main() function of 9_render.py does not take individual gnn_spec_file, target_format, etc.
    # It orchestrates calls to render_module.main() for files it finds.

    cli_args = parser.parse_args()

    # If verbose is set by arg, ensure logger level is DEBUG for standalone run
    if cli_args.verbose:
        current_script_logger = logging.getLogger(__name__) # __name__ will be __main__ here
        current_script_logger.setLevel(logging.DEBUG)
        # Also set root logger and handlers if basicConfig was used
        if logging.getLogger().handlers: # Check if basicConfig has run
             logging.getLogger().setLevel(logging.DEBUG)
             for handler in logging.getLogger().handlers:
                 handler.setLevel(logging.DEBUG)
        current_script_logger.debug("Verbose logging enabled for standalone run of 9_render.py.")

    # Call the script's main function, which processes a directory.
    exit_code = main(cli_args) 
    sys.exit(exit_code) 