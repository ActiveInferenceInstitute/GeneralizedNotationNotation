"""
Main rendering module for GNN specifications.

This module coordinates the rendering of GNN specifications to various
target platforms, including RxInfer.jl and PyMDP.
"""

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import renderers
from .pymdp_renderer import render_gnn_to_pymdp
from .rxinfer import render_gnn_to_rxinfer_toml

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def render_gnn_spec(
    gnn_spec: Dict[str, Any],
    target: str,
    output_directory: Union[str, Path],
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to a target format.
    
    Args:
        gnn_spec: Dictionary containing the GNN specification
        target: Target platform ("pymdp", "rxinfer_toml", etc.)
        output_directory: Directory for output files
        options: Additional options for the renderer
        
    Returns:
        Tuple of (success flag, message, list of artifact URIs)
    """
    output_directory = Path(output_directory) if isinstance(output_directory, str) else output_directory
    options = options or {}
    
    # Create output directory if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)
    
    if target.lower() == "pymdp":
        # Render to PyMDP
        model_name = gnn_spec.get("name", "model")
        script_path = output_directory / f"{model_name}_pymdp.py"
        return render_gnn_to_pymdp(gnn_spec, script_path, options)
        
    elif target.lower() == "rxinfer_toml":
        # Render to RxInfer TOML config
        model_name = gnn_spec.get("name", "model")
        config_path = output_directory / f"{model_name}_config.toml"
        return render_gnn_to_rxinfer_toml(gnn_spec, config_path, options)
        
    else:
        error_msg = f"Unsupported target platform: {target}"
        logger.error(error_msg)
        return False, error_msg, []

def main(cli_args=None):
    """Command-line entry point for the renderer."""
    parser = argparse.ArgumentParser(description="Render GNN specifications to various target platforms")
    parser.add_argument("gnn_file", help="Path to the GNN specification file")
    parser.add_argument("output_dir", help="Output directory for rendered files")
    parser.add_argument("target", choices=["pymdp", "rxinfer_toml"], default="pymdp", help="Target platform")
    parser.add_argument("--output_filename", help="Base filename for the output (without extension)")
    parser.add_argument("--debug", "--verbose", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args(cli_args)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    gnn_file_path = Path(args.gnn_file)
    if not gnn_file_path.exists():
        logger.error(f"GNN file not found: {gnn_file_path}")
        return 1
    
    # Parse the GNN file (placeholder - would normally use a proper parser)
    try:
        # Import dynamically to avoid circular imports
        from ..gnn.parser import parse_gnn_file
        gnn_spec = parse_gnn_file(gnn_file_path)
    except (ImportError, ModuleNotFoundError):
        logger.warning("Could not import GNN parser, using JSON loader as fallback")
        import json
        try:
            with open(gnn_file_path, "r", encoding="utf-8") as f:
                gnn_spec = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse GNN file: {gnn_file_path}")
            return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use custom filename if provided, otherwise use model name from spec
    if args.output_filename:
        base_filename = args.output_filename
    else:
        base_filename = gnn_spec.get("name", gnn_file_path.stem)
    
    # Determine output file path based on target and base filename
    if args.target == "pymdp":
        output_path = output_dir / f"{base_filename}_pymdp.py"
    elif args.target == "rxinfer_toml":
        output_path = output_dir / f"{base_filename}_config.toml"
    
    # Render the specification
    success, message, artifacts = render_gnn_spec(
        gnn_spec, 
        args.target, 
        output_dir,
        {"output_filename": base_filename}
    )
    
    if success:
        print(f"Successfully rendered to {args.target}: {message}")
        print(f"Output artifacts: {', '.join(artifacts)}")
        return 0
    else:
        print(f"Error rendering to {args.target}: {message}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 