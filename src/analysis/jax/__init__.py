"""
JAX Analysis Module

Per-framework analysis and visualization for JAX Active Inference simulations.
"""

from .analyzer import (
    generate_analysis_from_logs,
    create_jax_visualizations,
    extract_simulation_data,
)

__all__ = [
    "generate_analysis_from_logs",
    "create_jax_visualizations",
    "extract_simulation_data",
]
