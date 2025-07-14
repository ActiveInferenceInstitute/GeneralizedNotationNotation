"""
Renderers for GNN specifications to various target languages and frameworks.

This package contains modules for rendering GNN specifications to:
- RxInfer.jl
- PyMDP
- DisCoPy
- ActiveInference.jl
- Other simulators
"""

# Import renderers here to make them available at package level
from .render import render_gnn_spec, main

# Target-specific renderers with fallback for missing dependencies
try:
    from .rxinfer import render_gnn_to_rxinfer_toml
except ImportError:
    render_gnn_to_rxinfer_toml = None

try:
    from .pymdp.pymdp_renderer import render_gnn_to_pymdp
except ImportError:
    render_gnn_to_pymdp = None

try:
    from .discopy import render_gnn_to_discopy, render_gnn_to_discopy_jax, render_gnn_to_discopy_combined
except ImportError:
    render_gnn_to_discopy = None
    render_gnn_to_discopy_jax = None
    render_gnn_to_discopy_combined = None

try:
    from .activeinference_jl import render_gnn_to_activeinference_jl, render_gnn_to_activeinference_combined
except ImportError:
    render_gnn_to_activeinference_jl = None
    render_gnn_to_activeinference_combined = None

try:
    from .jax import render_gnn_to_jax, render_gnn_to_jax_pomdp, render_gnn_to_jax_combined
except ImportError:
    render_gnn_to_jax = None
    render_gnn_to_jax_pomdp = None
    render_gnn_to_jax_combined = None

__all__ = [
    'render_gnn_spec',  # Main render function
    'main',             # CLI entry point
    'render_gnn_to_pymdp',  # PyMDP-specific renderer
    'render_gnn_to_rxinfer_toml',  # RxInfer-specific renderer
    'render_gnn_to_discopy',  # DisCoPy diagram renderer
    'render_gnn_to_discopy_jax',  # DisCoPy JAX evaluation renderer
    'render_gnn_to_discopy_combined',  # Combined DisCoPy renderer
    'render_gnn_to_activeinference_jl',  # ActiveInference.jl renderer
    'render_gnn_to_activeinference_combined'  # Combined ActiveInference.jl renderer
]
__all__ += [
    'render_gnn_to_jax',
    'render_gnn_to_jax_pomdp',
    'render_gnn_to_jax_combined'
] 