"""
JAX Templates for GNN Rendering

This module contains comprehensive JAX templates for POMDPs and other Active Inference models.
Templates are designed for maximum performance with JIT compilation, vmap, and pmap.
"""

from .pomdp_template import POMDP_TEMPLATE
from .general_template import GENERAL_TEMPLATE
from .combined_template import COMBINED_TEMPLATE

__all__ = [
    'POMDP_TEMPLATE',
    'GENERAL_TEMPLATE', 
    'COMBINED_TEMPLATE'
] 