"""
Renderers for GNN specifications to various target languages and frameworks.

This package contains modules for rendering GNN specifications to:
- RxInfer.jl
- PyMDP
- DisCoPy
- ActiveInference.jl
- JAX
- Other simulators
"""

# Import renderers here to make them available at package level
from .render import render_gnn_spec, main

# Target-specific renderers with fallback for missing dependencies
try:
    from .rxinfer import render_gnn_to_rxinfer_toml
    RXINFER_AVAILABLE = True
except ImportError:
    render_gnn_to_rxinfer_toml = None
    RXINFER_AVAILABLE = False

try:
    from .pymdp.pymdp_renderer import render_gnn_to_pymdp
    PYMDP_AVAILABLE = True
except ImportError:
    render_gnn_to_pymdp = None
    PYMDP_AVAILABLE = False

try:
    from .discopy import render_gnn_to_discopy, render_gnn_to_discopy_jax
    render_gnn_to_discopy_combined = None  # This function doesn't exist yet
    DISCOPY_AVAILABLE = True
except ImportError:
    render_gnn_to_discopy = None
    render_gnn_to_discopy_jax = None
    render_gnn_to_discopy_combined = None
    DISCOPY_AVAILABLE = False

try:
    from .activeinference_jl import render_gnn_to_activeinference_jl, render_gnn_to_activeinference_combined
    ACTIVEINFERENCE_JL_AVAILABLE = True
except ImportError:
    render_gnn_to_activeinference_jl = None
    render_gnn_to_activeinference_combined = None
    ACTIVEINFERENCE_JL_AVAILABLE = False

try:
    from .jax import render_gnn_to_jax, render_gnn_to_jax_pomdp, render_gnn_to_jax_combined
    JAX_AVAILABLE = True
except ImportError:
    render_gnn_to_jax = None
    render_gnn_to_jax_pomdp = None
    render_gnn_to_jax_combined = None
    JAX_AVAILABLE = False

# MCP integration
try:
    from .mcp import (
        register_tools,
        handle_render_gnn_spec,
        handle_list_render_targets,
        RenderGnnInput,
        RenderGnnOutput,
        ListRenderTargetsInput,
        ListRenderTargetsOutput
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "GNN specification renderers for multiple target languages"

# Feature availability flags
FEATURES = {
    'pymdp_rendering': PYMDP_AVAILABLE,
    'rxinfer_rendering': RXINFER_AVAILABLE,
    'discopy_rendering': DISCOPY_AVAILABLE,
    'activeinference_jl_rendering': ACTIVEINFERENCE_JL_AVAILABLE,
    'jax_rendering': JAX_AVAILABLE,
    'mcp_integration': MCP_AVAILABLE
}

# Main API functions
__all__ = [
    # Core render function
    'render_gnn_spec',
    'main',
    
    # Target-specific renderers (test-compatible names)
    'render_pymdp_code',
    'render_rxinfer_code', 
    'render_discopy_code',
    'render_jax_code',
    'generate_render_report',
    
    # Target-specific renderers (actual names)
    'render_gnn_to_pymdp',
    'render_gnn_to_rxinfer_toml',
    'render_gnn_to_discopy',
    'render_gnn_to_discopy_jax',
    'render_gnn_to_discopy_combined',
    'render_gnn_to_activeinference_jl',
    'render_gnn_to_activeinference_combined',
    'render_gnn_to_jax',
    'render_gnn_to_jax_pomdp',
    'render_gnn_to_jax_combined',
    
    # MCP integration (if available)
    'register_tools',
    'handle_render_gnn_spec',
    'handle_list_render_targets',
    'RenderGnnInput',
    'RenderGnnOutput',
    'ListRenderTargetsInput',
    'ListRenderTargetsOutput',
    
    # Metadata
    'FEATURES',
    '__version__'
]

# Add conditional exports based on availability
if not PYMDP_AVAILABLE:
    __all__.remove('render_gnn_to_pymdp')

if not RXINFER_AVAILABLE:
    __all__.remove('render_gnn_to_rxinfer_toml')

if not DISCOPY_AVAILABLE:
    __all__.remove('render_gnn_to_discopy')
    __all__.remove('render_gnn_to_discopy_jax')
    __all__.remove('render_gnn_to_discopy_combined')

if not ACTIVEINFERENCE_JL_AVAILABLE:
    __all__.remove('render_gnn_to_activeinference_jl')
    __all__.remove('render_gnn_to_activeinference_combined')

if not JAX_AVAILABLE:
    __all__.remove('render_gnn_to_jax')
    __all__.remove('render_gnn_to_jax_pomdp')
    __all__.remove('render_gnn_to_jax_combined')

if not MCP_AVAILABLE:
    __all__.remove('register_tools')
    __all__.remove('handle_render_gnn_spec')
    __all__.remove('handle_list_render_targets')
    __all__.remove('RenderGnnInput')
    __all__.remove('RenderGnnOutput')
    __all__.remove('ListRenderTargetsInput')
    __all__.remove('ListRenderTargetsOutput')


def get_module_info():
    """Get comprehensive information about the render module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'available_targets': [],
        'supported_formats': {}
    }
    
    # Available targets
    if PYMDP_AVAILABLE:
        info['available_targets'].append('pymdp')
        info['supported_formats']['pymdp'] = 'Python PyMDP agent and environment'
    
    if RXINFER_AVAILABLE:
        info['available_targets'].append('rxinfer_toml')
        info['supported_formats']['rxinfer_toml'] = 'RxInfer.jl TOML configuration'
    
    if DISCOPY_AVAILABLE:
        info['available_targets'].extend(['discopy', 'discopy_jax'])
        info['supported_formats']['discopy'] = 'DisCoPy categorical diagrams'
        info['supported_formats']['discopy_jax'] = 'DisCoPy with JAX evaluation'
    
    if ACTIVEINFERENCE_JL_AVAILABLE:
        info['available_targets'].extend(['activeinference_jl', 'activeinference_combined'])
        info['supported_formats']['activeinference_jl'] = 'ActiveInference.jl models'
        info['supported_formats']['activeinference_combined'] = 'Combined ActiveInference.jl renderer'
    
    if JAX_AVAILABLE:
        info['available_targets'].extend(['jax', 'jax_pomdp', 'jax_combined'])
        info['supported_formats']['jax'] = 'JAX-based implementations'
        info['supported_formats']['jax_pomdp'] = 'JAX POMDP implementations'
        info['supported_formats']['jax_combined'] = 'Combined JAX renderer'
    
    return info


def get_available_renderers() -> dict:
    """Get information about all available renderers."""
    renderers = {}
    
    if PYMDP_AVAILABLE:
        renderers['pymdp'] = {
            'function': render_gnn_to_pymdp,
            'description': 'Render GNN to PyMDP Python code',
            'output_format': 'python'
        }
    
    if RXINFER_AVAILABLE:
        renderers['rxinfer_toml'] = {
            'function': render_gnn_to_rxinfer_toml,
            'description': 'Render GNN to RxInfer.jl TOML configuration',
            'output_format': 'toml'
        }
    
    if DISCOPY_AVAILABLE:
        renderers['discopy'] = {
            'function': render_gnn_to_discopy,
            'description': 'Render GNN to DisCoPy categorical diagrams',
            'output_format': 'python'
        }
        renderers['discopy_jax'] = {
            'function': render_gnn_to_discopy_jax,
            'description': 'Render GNN to DisCoPy with JAX evaluation',
            'output_format': 'python'
        }
    
    if ACTIVEINFERENCE_JL_AVAILABLE:
        renderers['activeinference_jl'] = {
            'function': render_gnn_to_activeinference_jl,
            'description': 'Render GNN to ActiveInference.jl models',
            'output_format': 'julia'
        }
        renderers['activeinference_combined'] = {
            'function': render_gnn_to_activeinference_combined,
            'description': 'Combined ActiveInference.jl renderer',
            'output_format': 'julia'
        }
    
    if JAX_AVAILABLE:
        renderers['jax'] = {
            'function': render_gnn_to_jax,
            'description': 'Render GNN to JAX implementations',
            'output_format': 'python'
        }
        renderers['jax_pomdp'] = {
            'function': render_gnn_to_jax_pomdp,
            'description': 'Render GNN to JAX POMDP implementations',
            'output_format': 'python'
        }
        renderers['jax_combined'] = {
            'function': render_gnn_to_jax_combined,
            'description': 'Combined JAX renderer',
            'output_format': 'python'
        }
    
    return renderers


# Test-compatible function aliases
def render_pymdp_code(gnn_file_path, output_dir=None):
    """Render GNN to PyMDP code (test-compatible alias)."""
    if not PYMDP_AVAILABLE:
        raise ImportError("PyMDP renderer not available")
    return render_gnn_to_pymdp(gnn_file_path, output_dir)

def render_rxinfer_code(gnn_file_path, output_dir=None):
    """Render GNN to RxInfer code (test-compatible alias)."""
    if not RXINFER_AVAILABLE:
        raise ImportError("RxInfer renderer not available")
    return render_gnn_to_rxinfer_toml(gnn_file_path, output_dir)

def render_discopy_code(gnn_file_path, output_dir=None):
    """Render GNN to DisCoPy code (test-compatible alias)."""
    if not DISCOPY_AVAILABLE:
        raise ImportError("DisCoPy renderer not available")
    return render_gnn_to_discopy(gnn_file_path, output_dir)

def render_jax_code(gnn_file_path, output_dir=None):
    """Render GNN to JAX code (test-compatible alias)."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX renderer not available")
    return render_gnn_to_jax(gnn_file_path, output_dir)

def generate_render_report(render_results, output_path=None):
    """Generate a report from render results (test-compatible alias)."""
    import json
    from datetime import datetime
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": render_results,
        "summary": {
            "total_renders": len(render_results) if isinstance(render_results, list) else 1,
            "successful_renders": sum(1 for r in render_results if r.get('success', False)) if isinstance(render_results, list) else (1 if render_results.get('success', False) else 0)
        }
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report 