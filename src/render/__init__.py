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
    'create_active_inference_diagram'
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
