#!/usr/bin/env python3
"""
Render Module

This module provides rendering capabilities for GNN specifications to various
target languages and simulation environments.
"""

# Phase 6: render submodules are in-tree; unconditional imports.
from .processor import (
    get_available_renderers,
    get_module_info,
    process_render,
    render_gnn_spec,
)
from .pomdp_processor import POMDPRenderProcessor, process_pomdp_for_frameworks
from .generators import (
    generate_activeinference_jl_code,
    generate_discopy_code,
    generate_pymdp_code,
    generate_rxinfer_code,
)
from .pymdp import render_gnn_to_pymdp
from .rxinfer import render_gnn_to_rxinfer, render_gnn_to_rxinfer_toml
from .discopy import render_gnn_to_discopy
from .activeinference_jl import render_gnn_to_activeinference_jl
from .pytorch import render_gnn_to_pytorch
from .numpyro import render_gnn_to_numpyro
from .pymdp.pymdp_renderer import PyMDPRenderer


class JAXRenderer:
    """Facade over ``render_gnn_to_jax`` exposed as a class for callers that
    want polymorphic dispatch. The real rendering work is in
    ``render/jax/jax_renderer.py`` — this class forwards ``render`` to it."""

    def render(self, spec) -> str:
        from .jax.jax_renderer import render_gnn_to_jax
        result = render_gnn_to_jax(spec)
        return result if isinstance(result, str) else str(result)


def get_supported_frameworks():
    """Return list of supported rendering frameworks.

    Returns:
        List of framework names that can be used for rendering.
    """
    return ['pymdp', 'rxinfer', 'activeinference_jl', 'jax', 'discopy', 'pytorch', 'numpyro']


def validate_render(result, framework=None):
    """Validate render output.

    Args:
        result: The render result to validate.
        framework: Optional framework name for framework-specific validation.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If validation fails.
    """
    if result is None:
        raise ValueError("Render result is None")
    if isinstance(result, str) and len(result) == 0:
        raise ValueError("Render result is empty string")
    return True


__all__ = [
    # Core functions
    'process_render',
    'render_gnn_spec',
    'get_module_info',
    'get_available_renderers',

    # Generator functions
    'generate_pymdp_code',
    'generate_rxinfer_code',
    'generate_activeinference_jl_code',
    'generate_discopy_code',

    # Specific renderer functions
    'render_gnn_to_pymdp',
    'render_gnn_to_rxinfer',
    'render_gnn_to_rxinfer_toml',
    'render_gnn_to_discopy',
    'render_gnn_to_activeinference_jl',
    'render_gnn_to_pytorch',
    'render_gnn_to_numpyro',

    # Renderer classes
    'PyMDPRenderer',
    'JAXRenderer',

    # POMDP processing
    'POMDPRenderProcessor',
    'process_pomdp_for_frameworks',

    # Utility functions
    'get_supported_frameworks',
    'validate_render',
]


__version__ = "1.6.0"
FEATURES = {
    "pymdp_rendering": True,
    "rxinfer_rendering": True,
    "activeinference_jl_rendering": True,
    "discopy_rendering": True,
    "jax_rendering": True,
    "pytorch_rendering": True,
    "numpyro_rendering": True,
    "mcp_integration": True,
    "pomdp_processing": True,
    "state_space_extraction": True,
    "modular_injection": True,
    "framework_specific_outputs": True,
    "structured_documentation": True
}

from .render import main  # expose CLI entry as attribute for tests
