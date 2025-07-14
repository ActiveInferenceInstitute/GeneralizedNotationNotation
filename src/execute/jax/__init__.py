"""
JAX Executor for GNN Processing Pipeline

This module provides comprehensive execution capabilities for JAX POMDP scripts
generated from GNN specifications, including device selection, performance monitoring,
and benchmarking.

@Web: https://github.com/google/jax
@Web: https://optax.readthedocs.io
@Web: https://flax.readthedocs.io
@Web: https://pfjax.readthedocs.io
"""

from .jax_runner import (
    run_jax_scripts,
    execute_jax_script,
    find_jax_scripts,
    is_jax_available,
    get_jax_device_info
)

__all__ = [
    'run_jax_scripts',
    'execute_jax_script', 
    'find_jax_scripts',
    'is_jax_available',
    'get_jax_device_info'
] 