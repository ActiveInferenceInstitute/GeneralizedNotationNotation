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

# Attempt to import the new logging utility
try:
    from utils.logging_utils import setup_standalone_logging
except ImportError:
    # Fallback for standalone execution or if src is not directly in path
    current_script_path_for_util = Path(__file__).resolve()
    project_root_for_util = current_script_path_for_util.parent.parent
    # Try adding project root, then src, to sys.path for utils
    paths_to_try = [str(project_root_for_util), str(project_root_for_util / "src")]
    original_sys_path = list(sys.path) # Store original path
    for p_try in paths_to_try:
        if p_try not in sys.path:
            sys.path.insert(0, p_try)
    try:
        from utils.logging_utils import setup_standalone_logging
    except ImportError:
        setup_standalone_logging = None
        _temp_logger_name = __name__ if __name__ != "__main__" else "src.9_render_import_warning"
        _temp_logger = logging.getLogger(_temp_logger_name)
        if not _temp_logger.hasHandlers():
            if not logging.getLogger().hasHandlers():
                logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
            else:
                 _temp_logger.addHandler(logging.StreamHandler(sys.stderr))
                 _temp_logger.propagate = False
        _temp_logger.warning(
            "Could not import setup_standalone_logging from utils.logging_utils. Standalone logging might be basic."
        )
    finally:
        sys.path = original_sys_path # Restore original sys.path

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
    # log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    # log_level_for_standalone = getattr(logging, log_level_str, logging.INFO)
    # logging.basicConfig(
    #     level=log_level_for_standalone,
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     stream=sys.stdout,
    #     force=True # Override any existing root logger configuration for standalone run
    # )

    # Define arguments that 9_render.py's main() function expects when called by main.py
    # or for a similar standalone test.
    # The main() function of 9_render.py itself scans a directory derived from output_dir.
    parser = argparse.ArgumentParser(description="GNN Rendering Step - Standalone Directory Processing Mode")
    
    # Determine project root for default paths, assuming script is in src/
    script_file_path = Path(__file__).resolve()
    project_root_for_defaults = script_file_path.parent.parent # src/ -> project_root
    default_output_dir_standalone = project_root_for_defaults / "output"

    parser.add_argument(
        "--output-dir", 
        default=default_output_dir_standalone, 
        type=Path,
        help=f"Main pipeline output directory. Render step scans 'output_dir/gnn_exports'. Default: {default_output_dir_standalone.relative_to(project_root_for_defaults) if default_output_dir_standalone.is_relative_to(project_root_for_defaults) else default_output_dir_standalone}"
    )
    parser.add_argument(
        "--recursive", 
        action=argparse.BooleanOptionalAction, 
        default=False, 
        help="Recursively scan for GNN spec files in 'output_dir/gnn_exports'."
    )
    parser.add_argument(
        "--verbose", 
        action=argparse.BooleanOptionalAction, 
        default=False,
        help="Enable verbose logging for this script."
    )
    # Note: The main() function of 9_render.py does not take individual gnn_spec_file, target_format, etc.
    # It orchestrates calls to render_module.main() for files it finds.

    cli_args = parser.parse_args()

    # Setup logging for standalone execution
    log_level_to_set = logging.DEBUG if cli_args.verbose else logging.INFO
    if setup_standalone_logging:
        setup_standalone_logging(level=log_level_to_set, logger_name=__name__)
    else:
        if not logging.getLogger().hasHandlers(): # Check root handlers
            logging.basicConfig(
                level=log_level_to_set,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                # datefmt="%Y-%m-%d %H:%M:%S", # Use default datefmt
                stream=sys.stdout
            )
        # Ensure this script's logger (which is __main__ here) level is set even in fallback
        logging.getLogger(__name__).setLevel(log_level_to_set) 
        logging.getLogger(__name__).warning("Using fallback basic logging due to missing setup_standalone_logging utility.")

    # Quieten noisy libraries if run standalone
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    # (PIL is not directly used here, but matplotlib might use it)

    # Call the script's main function, which processes a directory.
    exit_code = main(cli_args) 
    sys.exit(exit_code) 