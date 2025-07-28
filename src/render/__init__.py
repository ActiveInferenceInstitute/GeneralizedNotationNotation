#!/usr/bin/env python3
"""
Render Module

This module provides rendering capabilities for GNN specifications to various
target languages and simulation environments.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List

# Import all the missing render functions from subdirectories
try:
    from .pymdp.pymdp_renderer import PyMdpRenderer, render_gnn_to_pymdp_impl
    from .pymdp.pymdp_converter import GnnToPyMdpConverter, convert_gnn_to_pymdp
    from .activeinference_jl.activeinference_jl_renderer import ActiveInferenceJlRenderer, render_gnn_to_activeinference_jl_impl
    from .rxinfer.rxinfer_renderer import RxInferRenderer, render_gnn_to_rxinfer_impl  
    from .discopy.discopy_renderer import DiscopyCategoryRenderer, render_gnn_to_discopy_impl
except ImportError:
    # Fallback classes and functions
    class PyMdpRenderer:
        def __init__(self): pass
    class GnnToPyMdpConverter:
        def __init__(self): pass
    class ActiveInferenceJlRenderer:
        def __init__(self): pass
    class RxInferRenderer:
        def __init__(self): pass
    class DiscopyCategoryRenderer:
        def __init__(self): pass
    
    def render_gnn_to_pymdp_impl(*args, **kwargs):
        return {"error": "PyMDP renderer not available"}
    def render_gnn_to_activeinference_jl_impl(*args, **kwargs):
        return {"error": "ActiveInference.jl renderer not available"}
    def render_gnn_to_rxinfer_impl(*args, **kwargs):
        return {"error": "RxInfer renderer not available"}
    def render_gnn_to_discopy_impl(*args, **kwargs):
        return {"error": "DisCoPy renderer not available"}
    def convert_gnn_to_pymdp(*args, **kwargs):
        return {"error": "PyMDP converter not available"}

# Export the missing functions that scripts are looking for
def render_gnn_to_pymdp(*args, **kwargs):
    """Render GNN to PyMDP simulation code."""
    return render_gnn_to_pymdp_impl(*args, **kwargs)

def render_gnn_to_activeinference_jl(*args, **kwargs):
    """Render GNN to ActiveInference.jl simulation code."""
    return render_gnn_to_activeinference_jl_impl(*args, **kwargs)

def render_gnn_to_rxinfer(*args, **kwargs):
    """Render GNN to RxInfer.jl simulation code."""
    return render_gnn_to_rxinfer_impl(*args, **kwargs)

def render_gnn_to_discopy(*args, **kwargs):
    """Render GNN to DisCoPy categorical diagram."""
    return render_gnn_to_discopy_impl(*args, **kwargs)

def pymdp_renderer(*args, **kwargs):
    """Legacy function name for PyMDP rendering."""
    return render_gnn_to_pymdp(*args, **kwargs)

def activeinference_jl_renderer(*args, **kwargs):
    """Legacy function name for ActiveInference.jl rendering."""
    return render_gnn_to_activeinference_jl(*args, **kwargs)

def rxinfer_renderer(*args, **kwargs):
    """Legacy function name for RxInfer rendering."""
    return render_gnn_to_rxinfer(*args, **kwargs)

def discopy_renderer(*args, **kwargs):
    """Legacy function name for DisCoPy rendering."""
    return render_gnn_to_discopy(*args, **kwargs)

def pymdp_converter(*args, **kwargs):
    """Legacy function name for PyMDP conversion."""
    return convert_gnn_to_pymdp(*args, **kwargs)

# Add to __all__ for proper exports
__all__ = [
    'PyMdpRenderer', 'GnnToPyMdpConverter', 'ActiveInferenceJlRenderer',
    'RxInferRenderer', 'DiscopyCategoryRenderer',
    'render_gnn_to_pymdp', 'render_gnn_to_activeinference_jl', 
    'render_gnn_to_rxinfer', 'render_gnn_to_discopy',
    'pymdp_renderer', 'activeinference_jl_renderer', 'rxinfer_renderer',
    'discopy_renderer', 'pymdp_converter', 'process_render'
]

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

def main():
    """Main entry point for the render module."""
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 3:
        print("Usage: render <gnn_file> <output_dir> [target]")
        return 1
    
    gnn_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    target = sys.argv[3] if len(sys.argv) > 3 else "pymdp"
    
    if not gnn_file.exists():
        print(f"Error: GNN file {gnn_file} not found")
        return 1
    
    try:
        # Read GNN file
        with open(gnn_file, 'r') as f:
            gnn_content = f.read()
        
        # Parse GNN content (simplified)
        gnn_spec = {
            "model_name": gnn_file.stem,
            "content": gnn_content
        }
        
        # Render
        success, message, warnings = render_gnn_spec(gnn_spec, target, output_dir)
        
        if success:
            print(f"Successfully rendered to {target}")
            return 0
        else:
            print(f"Error: {message}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1

def get_module_info() -> Dict[str, Any]:
    """Get comprehensive information about the render module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'available_targets': [],
        'supported_formats': []
    }
    
    # Available targets
    info['available_targets'].extend([
        'pymdp', 'rxinfer', 'activeinference_jl', 'jax', 'discopy'
    ])
    
    # Supported formats
    info['supported_formats'].extend([
        'Python', 'Julia', 'JAX', 'DisCoPy'
    ])
    
    return info

def get_available_renderers() -> Dict[str, Dict[str, Any]]:
    """Get available renderers and their capabilities."""
    return {
        'pymdp': {
            'function': 'render_gnn_to_pymdp',
            'description': 'PyMDP simulation code generator',
            'output_format': 'Python',
            'available': True,
            'features': ['discrete_state', 'discrete_action', 'belief_state']
        },
        'rxinfer': {
            'function': 'render_gnn_to_rxinfer',
            'description': 'RxInfer.jl inference code generator',
            'output_format': 'Julia',
            'available': True,
            'features': ['probabilistic_inference', 'message_passing']
        },
        'activeinference_jl': {
            'function': 'render_gnn_to_activeinference_jl',
            'description': 'ActiveInference.jl simulation code generator',
            'output_format': 'Julia',
            'available': True,
            'features': ['active_inference', 'free_energy']
        },
        'jax': {
            'function': 'render_gnn_to_jax',
            'description': 'JAX-based simulation code generator',
            'output_format': 'Python',
            'available': True,
            'features': ['automatic_differentiation', 'gpu_acceleration']
        },
        'discopy': {
            'function': 'render_gnn_to_discopy',
            'description': 'DisCoPy categorical diagram generator',
            'output_format': 'Python',
            'available': True,
            'features': ['categorical_diagrams', 'tensor_networks']
        }
    }

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

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "Rendering for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'pymdp_rendering': True,
    'rxinfer_rendering': True,
    'activeinference_jl_rendering': True,
    'discopy_rendering': True,
    'jax_rendering': True,
    'mcp_integration': True
}

__all__ = [
    'process_render',
    'render_gnn_spec',
    'main',
    'get_module_info',
    'get_available_renderers',
    'FEATURES',
    '__version__'
]
