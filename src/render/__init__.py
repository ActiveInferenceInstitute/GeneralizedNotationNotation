#!/usr/bin/env python3
"""
Render Module

This module provides rendering capabilities for GNN specifications to various
target languages and simulation environments.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List

def process_render(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process render step - generate simulation code from GNN files.
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for rendered code
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if successful, False otherwise
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        from .render import render_gnn_files
        
        return render_gnn_files(
            target_dir=target_dir,
            output_dir=output_dir,
            logger=logger,
            verbose=verbose,
            **kwargs
        )
        
    except Exception as e:
        logger.error(f"Render processing failed: {e}")
        return False

def render_gnn_spec(
    gnn_spec: Dict[str, Any],
    target: str,
    output_directory: Union[str, Path],
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render GNN specification to target format.
    
    Args:
        gnn_spec: Parsed GNN specification
        target: Target format ('pymdp', 'rxinfer', 'activeinference_jl', 'jax', 'discopy')
        output_directory: Output directory
        options: Optional rendering options
        
    Returns:
        Tuple of (success, message, warnings)
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if target == 'pymdp':
        from .pymdp.pymdp_renderer import render_gnn_to_pymdp
        
        model_name = gnn_spec.get('model_name', 'GNN_Model')
        output_file = output_dir / f"{model_name}_pymdp_simulation.py"
        
        return render_gnn_to_pymdp(gnn_spec, output_file, options)
        
    elif target == 'rxinfer':
        from .rxinfer.toml_generator import render_gnn_to_rxinfer_toml
        
        model_name = gnn_spec.get('model_name', 'GNN_Model')
        output_file = output_dir / f"{model_name}_rxinfer.jl"
        
        return render_gnn_to_rxinfer_toml(gnn_spec, output_file, options)
        
    elif target == 'activeinference_jl':
        from .activeinference_jl.activeinference_renderer import render_gnn_to_activeinference_jl
        
        model_name = gnn_spec.get('model_name', 'GNN_Model')
        output_file = output_dir / f"{model_name}_activeinference.jl"
        
        return render_gnn_to_activeinference_jl(gnn_spec, output_file, options)
        
    elif target == 'jax':
        from .jax.jax_renderer import render_gnn_to_jax
        
        model_name = gnn_spec.get('model_name', 'GNN_Model')
        output_file = output_dir / f"{model_name}_jax.py"
        
        return render_gnn_to_jax(gnn_spec, output_file, options)
        
    elif target == 'discopy':
        from .discopy.discopy_renderer import render_gnn_to_discopy
        
        model_name = gnn_spec.get('model_name', 'GNN_Model')
        output_file = output_dir / f"{model_name}_discopy.py"
        
        return render_gnn_to_discopy(gnn_spec, output_file, options)
        
    else:
        return False, f"Unsupported target: {target}", []

__all__ = [
    'process_render',
    'render_gnn_spec'
]
