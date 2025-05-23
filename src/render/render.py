"""
Main module for the GNN rendering step.

This module contains the core logic for transforming GNN specifications
into executable formats for target simulation environments like PyMDP and RxInfer.jl.
"""

import argparse
import logging
import json # For parsing GNN spec file
from pathlib import Path
import sys
from typing import Optional, Tuple, List, Dict, Any

# Import the new modular renderer
from .pymdp_renderer import render_gnn_to_pymdp
from .rxinfer import render_gnn_to_rxinfer_jl, placeholder_gnn_parser # Placeholder for actual parser

# Attempt to import the logging utility for standalone execution
try:
    from utils.logging_utils import setup_standalone_logging
except ImportError:
    # Fallback for standalone execution or if src is not directly in path
    _current_script_path_for_util = Path(__file__).resolve()
    _project_root_for_util = _current_script_path_for_util.parent.parent
    _paths_to_try_util = [str(_project_root_for_util), str(_project_root_for_util / "src")]
    _original_sys_path_util = list(sys.path)
    for _p_try_util in _paths_to_try_util:
        if _p_try_util not in sys.path:
            sys.path.insert(0, _p_try_util)
    try:
        from utils.logging_utils import setup_standalone_logging
    except ImportError:
        setup_standalone_logging = None # Define as None if import fails
    finally:
        # Restore sys.path if it was modified, to prevent side effects
        # This is important if this module might be imported after standalone path manipulation
        if _current_script_path_for_util:
             sys.path = _original_sys_path_util

logger = logging.getLogger(__name__)

RENDERER_MAPPING = {
    "pymdp": render_gnn_to_pymdp,
    "rxinfer": render_gnn_to_rxinfer_jl
}

def render_gnn_spec(
    gnn_spec: Dict[str, Any], 
    output_script_path: Path, 
    target_format: str,
    render_options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Renders a GNN specification to the specified target format.

    Args:
        gnn_spec: The parsed GNN specification as a Python dictionary.
        output_script_path: Path to save the rendered script.
        target_format: The target rendering format ("pymdp" or "rxinfer").
        render_options: Optional dictionary of options for the specific renderer.

    Returns:
        A tuple: (success: bool, message: str, artifact_uris: List[str])
    """
    render_options = render_options or {}

    if target_format.lower() == "pymdp":
        logger.info(f"Rendering GNN spec to PyMDP format at {output_script_path}...")
        # Using the modular PyMDP renderer from pymdp_renderer.py
        return render_gnn_to_pymdp(gnn_spec, output_script_path, options=render_options)
    elif target_format.lower() == "rxinfer":
        logger.info(f"Rendering GNN spec to RxInfer.jl format at {output_script_path}...")
        # Note: render_gnn_to_rxinfer_jl will be refactored similarly in the future
        return render_gnn_to_rxinfer_jl(gnn_spec, output_script_path, options=render_options)
    else:
        msg = f"Unsupported target format: {target_format}. Supported formats: pymdp, rxinfer."
        logger.error(msg)
        return False, msg, []

def main(cli_args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Render GNN specifications to executable formats.")
    parser.add_argument("gnn_spec_file", type=Path, help="Path to the GNN specification file (JSON format).")
    parser.add_argument("output_dir", type=Path, help="Directory to save the rendered output script.")
    parser.add_argument("target_format", type=str.lower, choices=["pymdp", "rxinfer"], 
                        help="Target format for rendering (pymdp or rxinfer).")
    parser.add_argument("--output_filename", type=str, default=None, 
                        help="Optional custom name for the output file (extension will be added automatically).")
    # Add other generic options if needed, e.g., verbosity for this script's logger
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for the renderer.")

    args = parser.parse_args(cli_args)

    # Set logger level based on verbose flag, effective if main() is called directly
    # or if standalone and setup_standalone_logging hasn't already set this specific logger.
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled for render.py.")
    else:
        # If not verbose, and not configured by a parent, it might default to WARNING or INFO.
        # Ensure it's at least INFO if no parent logger set it higher or if standalone.
        if logger.level == logging.NOTSET or logger.level > logging.INFO: # check if unconfigured or too high
            logger.setLevel(logging.INFO)

    logger.info(f"Starting GNN rendering process for {args.gnn_spec_file} to {args.target_format}")

    if not args.gnn_spec_file.is_file():
        logger.error(f"GNN specification file not found: {args.gnn_spec_file}")
        return 1

    try:
        with open(args.gnn_spec_file, 'r', encoding='utf-8') as f:
            gnn_spec = json.load(f)
        logger.debug(f"src/render/render.py: Loaded gnn_spec from JSON ({args.gnn_spec_file}): {{gnn_spec}}")
        logger.info(f"Successfully loaded GNN specification from {args.gnn_spec_file}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from GNN specification file {args.gnn_spec_file}: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to read GNN specification file {args.gnn_spec_file}: {e}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output directory exists: {args.output_dir}")

    # Determine output filename
    if args.output_filename:
        base_filename = args.output_filename
        # Remove existing extension if user provided one, to avoid doubling up
        base_filename = Path(base_filename).stem 
    else:
        base_filename = args.gnn_spec_file.stem
        if base_filename.endswith('.gnn'): # e.g., my_model.gnn.json -> my_model
            base_filename = Path(base_filename).stem

    file_extension = ".py" if args.target_format == "pymdp" else ".jl"
    output_script_name = f"{base_filename}_rendered{file_extension}"
    output_script_path = args.output_dir / output_script_name

    # Create render_options with proper typing to avoid linter errors
    render_options: Dict[str, Any] = {}
    
    if args.target_format == "pymdp":
        render_options["include_example_usage"] = True # Default for PyMDP
    elif args.target_format == "rxinfer":
        render_options["include_inference_script"] = True # Default for RxInfer
        render_options["data_bindings"] = {} # Example, user might need to specify this
        render_options["inference_iterations"] = 50
        render_options["calculate_free_energy"] = False

    logger.info(f"Output will be saved to: {output_script_path}")

    success, message, artifacts = render_gnn_spec(
        gnn_spec,
        output_script_path,
        args.target_format,
        render_options
    )

    if success:
        logger.info(f"Rendering successful: {message}")
        if artifacts:
            logger.info(f"Generated artifacts: {artifacts}")
        return 0
    else:
        logger.error(f"Rendering failed: {message}")
        return 1

if __name__ == "__main__":
    # When run as a script, cli_args is None, so sys.argv will be used by main().
    
    # For standalone execution, we need to parse verbosity to set up logging correctly *before* calling main().
    # Create a temporary parser just to get the verbose flag for initial logging setup.
    temp_parser = argparse.ArgumentParser(add_help=False) # Don't show help for this temporary parse
    temp_parser.add_argument("--verbose", action="store_true")
    # Use parse_known_args to avoid errors if other args are present (they'll be parsed by main's parser)
    known_args, _ = temp_parser.parse_known_args()
    
    log_level_for_standalone = logging.DEBUG if known_args.verbose else logging.INFO

    if setup_standalone_logging:
        # logger_name=None will configure the root logger.
        # If we want this script's logger (__main__) to also be set, pass logger_name=__name__.
        setup_standalone_logging(level=log_level_for_standalone, logger_name=__name__) 
    else:
        # Fallback basic config if utility function couldn't be imported
        # Ensure this only runs if no handlers are configured on the root logger yet.
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=log_level_for_standalone, 
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                stream=sys.stdout
            )
        # Ensure this script's logger (__main__) level is set even in fallback
        logging.getLogger(__name__).setLevel(log_level_for_standalone)
        if not setup_standalone_logging: # Log warning only if it was actually None
            logging.getLogger(__name__).warning(
                "Using fallback basicConfig for logging due to missing setup_standalone_logging utility."
            )

    # Now that logging is configured based on potential --verbose, call main().
    # main() will then parse all arguments again and also set its logger level.
    # This is slightly redundant but ensures correct behavior in all invocation scenarios.
    sys.exit(main()) 