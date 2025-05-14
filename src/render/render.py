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
from typing import Optional, Tuple, List

# Assuming pymdp.py and rxinfer.py are in the same directory
from .pymdp import render_gnn_to_pymdp, placeholder_gnn_parser_pymdp # Placeholder for actual parser
from .rxinfer import render_gnn_to_rxinfer_jl, placeholder_gnn_parser # Placeholder for actual parser

logger = logging.getLogger(__name__)

RENDERER_MAPPING = {
    "pymdp": render_gnn_to_pymdp,
    "rxinfer": render_gnn_to_rxinfer_jl
}

def render_gnn_spec(
    gnn_spec: dict, 
    output_script_path: Path, 
    target_format: str,
    render_options: Optional[dict] = None
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
        # Note: render_gnn_to_pymdp will be refactored to take gnn_spec (dict) directly
        # and output_script_path. The gnn_parser argument will be removed from it.
        # For now, we pass a dummy parser, but this part of render_gnn_to_pymdp will be unused.
        return render_gnn_to_pymdp(gnn_spec, output_script_path, options=render_options)
    elif target_format.lower() == "rxinfer":
        logger.info(f"Rendering GNN spec to RxInfer.jl format at {output_script_path}...")
        # Note: render_gnn_to_rxinfer_jl will be refactored similarly.
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

    logger.info(f"Starting GNN rendering process for {args.gnn_spec_file} to {args.target_format}")

    if not args.gnn_spec_file.is_file():
        logger.error(f"GNN specification file not found: {args.gnn_spec_file}")
        return 1

    try:
        with open(args.gnn_spec_file, 'r', encoding='utf-8') as f:
            gnn_spec = json.load(f)
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

    # Placeholder for render_options; could be extended via more CLI args or a config file
    render_options = {}
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
    # Basic logging setup for standalone execution.
    # This will be effective only if this script is run directly.
    # If imported, the importing module (e.g., main.py) should handle basicConfig.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sys.exit(main()) 