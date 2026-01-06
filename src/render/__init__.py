#!/usr/bin/env python3
"""
Render Module

This module provides rendering capabilities for GNN specifications to various
target languages and simulation environments.
"""

from .processor import (
    process_render,
    render_gnn_spec,
    get_module_info,
    get_available_renderers
)

# Import POMDP processing capabilities if available
try:
    from .pomdp_processor import (
        POMDPRenderProcessor,
        process_pomdp_for_frameworks
    )
    POMDP_PROCESSING_AVAILABLE = True
except ImportError:
    POMDP_PROCESSING_AVAILABLE = False

from .generators import (
    generate_pymdp_code,
    generate_rxinfer_code,
    generate_rxinfer_fallback_code,
    generate_activeinference_jl_code,
    generate_activeinference_jl_fallback_code,
    generate_discopy_code,
    generate_discopy_fallback_code,
    create_active_inference_diagram
)

# Import specific renderers (used by tests)
try:
    from .pymdp import render_gnn_to_pymdp
except ImportError:
    render_gnn_to_pymdp = None

try:
    from .rxinfer import render_gnn_to_rxinfer, render_gnn_to_rxinfer_toml
except ImportError:
    render_gnn_to_rxinfer = None
    render_gnn_to_rxinfer_toml = None

try:
    from .discopy import render_gnn_to_discopy
except ImportError:
    render_gnn_to_discopy = None

try:
    from .activeinference_jl import render_gnn_to_activeinference_jl
except ImportError:
    render_gnn_to_activeinference_jl = None

__all__ = [
    # Core functions
    'process_render',
    'render_gnn_spec',
    'get_module_info',
    'get_available_renderers',
    
    # Generator functions
    'generate_pymdp_code',
    'generate_rxinfer_code',
    'generate_rxinfer_fallback_code',
    'generate_activeinference_jl_code',
    'generate_activeinference_jl_fallback_code',
    'generate_discopy_code',
    'generate_discopy_fallback_code',
    'create_active_inference_diagram',
    
    # Specific renderer functions (may be None if submodule unavailable)
    'render_gnn_to_pymdp',
    'render_gnn_to_rxinfer',
    'render_gnn_to_rxinfer_toml',
    'render_gnn_to_discopy',
    'render_gnn_to_activeinference_jl'
] + (['POMDPRenderProcessor', 'process_pomdp_for_frameworks'] if POMDP_PROCESSING_AVAILABLE else [])



__version__ = "2.0.0"
FEATURES = {
    "pymdp_rendering": True, 
    "rxinfer_rendering": True, 
    "activeinference_jl_rendering": True, 
    "discopy_rendering": True, 
    "jax_rendering": True, 
    "mcp_integration": True,
    "pomdp_processing": POMDP_PROCESSING_AVAILABLE,
    "state_space_extraction": POMDP_PROCESSING_AVAILABLE,
    "modular_injection": POMDP_PROCESSING_AVAILABLE,
    "framework_specific_outputs": True,
    "structured_documentation": True
}

from .render import main  # expose CLI entry as attribute for tests
