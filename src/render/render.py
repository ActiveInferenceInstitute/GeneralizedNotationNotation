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
        f"Details: {e}. To enable PyMDP rendering: pip install pymdp. "
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
        target: Target platform ("pymdp", "rxinfer_toml", "discopy", "discopy_jax", "discopy_combined", etc.)
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
        if not PYMDP_AVAILABLE or render_gnn_to_pymdp is None:
            return False, (
                "PyMDP renderer not available. "
                "To enable: pip install pymdp. "
                "Try alternative frameworks: rxinfer, activeinference_jl, discopy, or jax."
            ), []
        # Render to PyMDP
        model_name = gnn_spec.get("name", "model")
        script_path = output_directory / f"{model_name}_pymdp.py"
        return render_gnn_to_pymdp(gnn_spec, script_path, options)
        
    elif target.lower() == "rxinfer_toml":
        if not RXINFER_AVAILABLE or render_gnn_to_rxinfer_toml is None:
            return False, "RxInfer renderer not available", []
        # Render to RxInfer TOML config
        model_name = gnn_spec.get("name", "model")
        config_path = output_directory / f"{model_name}_config.toml"
        return render_gnn_to_rxinfer_toml(gnn_spec, config_path, options)
        
    elif target.lower() == "discopy":
        if not DISCOPY_AVAILABLE or render_gnn_to_discopy is None:
            return False, "DisCoPy renderer not available", []
        # Render to DisCoPy categorical diagram
        model_name = gnn_spec.get("name", "model")
        script_path = output_directory / f"{model_name}_discopy.py"
        return render_gnn_to_discopy(gnn_spec, script_path, options)
        
    elif target.lower() == "discopy_jax":
        if not DISCOPY_AVAILABLE or render_gnn_to_discopy_jax is None:
            return False, "DisCoPy JAX renderer not available", []
        # Render to DisCoPy matrix diagram with JAX evaluation
        model_name = gnn_spec.get("name", "model")
        script_path = output_directory / f"{model_name}_discopy_jax.py"
        return render_gnn_to_discopy_jax(gnn_spec, script_path, options)
        
    elif target.lower() == "discopy_combined":
        if not DISCOPY_AVAILABLE:
            return False, "DisCoPy renderer not available", []
        # Render to both DisCoPy diagram and JAX evaluation
        model_name = gnn_spec.get("name", "model")
        script_path = output_directory / f"{model_name}_discopy_combined.py"
        
        # Use the combined renderer from discopy module
        try:
            from .discopy.discopy_renderer import render_gnn_to_discopy_combined
            return render_gnn_to_discopy_combined(gnn_spec, script_path, options)
        except ImportError:
            return False, "DisCoPy combined renderer not available", []
        
    elif target.lower() == "activeinference_jl":
        if not ACTIVEINFERENCE_JL_AVAILABLE or render_gnn_to_activeinference_jl is None:
            return False, "ActiveInference.jl renderer not available", []
        # Render to ActiveInference.jl script
        model_name = gnn_spec.get("name", "model")
        script_path = output_directory / f"{model_name}_activeinference.jl"
        return render_gnn_to_activeinference_jl(gnn_spec, script_path, options)
        
    elif target.lower() == "activeinference_combined":
        if not ACTIVEINFERENCE_JL_AVAILABLE or render_gnn_to_activeinference_combined is None:
            return False, "ActiveInference.jl combined renderer not available", []
        # Render to multiple ActiveInference.jl scripts with analysis suite
        return render_gnn_to_activeinference_combined(gnn_spec, output_directory, options)
        
    elif target == "jax":
        if not JAX_AVAILABLE or render_gnn_to_jax is None:
            return False, "JAX renderer not available", []
        model_name = gnn_spec.get("name", "model")
        script_path = output_directory / f"{model_name}_jax.py"
        return render_gnn_to_jax(gnn_spec, script_path, options)
        
    elif target == "jax_pomdp":
        if not JAX_AVAILABLE or render_gnn_to_jax_pomdp is None:
            return False, "JAX POMDP renderer not available", []
        model_name = gnn_spec.get("name", "model")
        script_path = output_directory / f"{model_name}_jax_pomdp.py"
        return render_gnn_to_jax_pomdp(gnn_spec, script_path, options)
        
    else:
        error_msg = f"Unsupported target platform: {target}"
        logger.error(error_msg)
        return False, error_msg, []

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
        from ..gnn.parser import parse_gnn_file
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