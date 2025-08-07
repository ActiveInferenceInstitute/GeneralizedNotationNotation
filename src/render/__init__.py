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

# Import legacy wrapper functions
from .legacy import (
    render_gnn_to_pymdp,
    render_gnn_to_activeinference_jl,
    render_gnn_to_rxinfer,
    render_gnn_to_discopy,
    pymdp_renderer,
    activeinference_jl_renderer,
    rxinfer_renderer,
    discopy_renderer,
    pymdp_converter
)

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
    
    # Legacy wrappers
    'render_gnn_to_pymdp',
    'render_gnn_to_activeinference_jl',
    'render_gnn_to_rxinfer',
    'render_gnn_to_discopy',
    'pymdp_renderer',
    'activeinference_jl_renderer',
    'rxinfer_renderer',
    'discopy_renderer',
    'pymdp_converter'
]
