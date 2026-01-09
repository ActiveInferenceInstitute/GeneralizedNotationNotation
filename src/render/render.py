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

# Import renderers with proper error handling
try:
    from .pymdp import render_gnn_to_pymdp
    PYMDP_AVAILABLE = True
except ImportError as e:
    logging.warning(
        f"PyMDP renderer not available - this is normal if PyMDP is not installed. "
        f"Details: {e}. To enable PyMDP rendering: uv pip install inferactively-pymdp. "
        f"Alternative frameworks available: RxInfer.jl, ActiveInference.jl, DisCoPy, JAX."
    )
    render_gnn_to_pymdp = None
    PYMDP_AVAILABLE = False

try:
    from .rxinfer import render_gnn_to_rxinfer_toml
    RXINFER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RxInfer renderer not available: {e}")
    render_gnn_to_rxinfer_toml = None
    RXINFER_AVAILABLE = False

try:
    from .discopy import render_gnn_to_discopy, render_gnn_to_discopy_jax
    DISCOPY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"DisCoPy renderer not available: {e}")
    render_gnn_to_discopy = None
    render_gnn_to_discopy_jax = None
    DISCOPY_AVAILABLE = False

try:
    from .activeinference_jl import render_gnn_to_activeinference_jl, render_gnn_to_activeinference_combined
    ACTIVEINFERENCE_JL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ActiveInference.jl renderer not available: {e}")
    render_gnn_to_activeinference_jl = None
    render_gnn_to_activeinference_combined = None
    ACTIVEINFERENCE_JL_AVAILABLE = False

try:
    from .jax import render_gnn_to_jax, render_gnn_to_jax_pomdp, render_gnn_to_jax_combined
    JAX_AVAILABLE = True
except ImportError as e:
    logging.warning(f"JAX renderer not available: {e}")
    render_gnn_to_jax = None
    render_gnn_to_jax_pomdp = None
    render_gnn_to_jax_combined = None
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from .processor import render_gnn_spec

def main(cli_args=None):
    """Command-line entry point for the renderer."""
    parser = argparse.ArgumentParser(description="Render GNN specifications to various target platforms")
    parser.add_argument("gnn_file", help="Path to the GNN specification file")
    parser.add_argument("output_dir", help="Output directory for rendered files")
    parser.add_argument("target", choices=["pymdp", "rxinfer_toml", "discopy", "discopy_jax", "discopy_combined", "activeinference_jl", "activeinference_combined", "jax", "jax_pomdp"], 
                       default="pymdp", help="Target platform")
    parser.add_argument("--output_filename", help="Base filename for the output (without extension)")
    parser.add_argument("--debug", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--jax-seed", type=int, default=0, help="Seed for JAX PRNG (for discopy_jax targets)")
    
    args = parser.parse_args(cli_args)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    gnn_file_path = Path(args.gnn_file)
    if not gnn_file_path.exists():
        logger.error(f"GNN file not found: {gnn_file_path}")
        return 1
    
    # Parse the GNN file using proper parsing
    try:
        # Try to import and use the GNN parser
        from gnn.parser import parse_gnn_file
        gnn_spec = parse_gnn_file(gnn_file_path)
        logger.info(f"Successfully parsed GNN file using parser: {gnn_file_path}")
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"Could not import GNN parser: {e}")
        # Fall back to JSON or markdown parsing
        if gnn_file_path.suffix.lower() == '.json':
            import json
            try:
                with open(gnn_file_path, "r", encoding="utf-8") as f:
                    gnn_spec = json.load(f)
                logger.info(f"Successfully parsed JSON GNN file: {gnn_file_path}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON GNN file: {gnn_file_path} - {e}")
                return 1
        else:
            # Try markdown parsing
            try:
                from .pymdp.pymdp_renderer import parse_gnn_markdown
                with open(gnn_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                gnn_spec = parse_gnn_markdown(content, gnn_file_path)
                if not gnn_spec:
                    logger.error(f"Failed to parse markdown GNN file: {gnn_file_path}")
                    return 1
                logger.info(f"Successfully parsed markdown GNN file: {gnn_file_path}")
            except Exception as e:
                logger.error(f"Failed to parse GNN file: {gnn_file_path} - {e}")
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