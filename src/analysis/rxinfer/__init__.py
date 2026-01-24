"""
RxInfer.jl Analysis Module

Per-framework analysis and visualization for RxInfer.jl simulations.
"""

from .analyzer import (
    generate_analysis_from_logs,
    create_rxinfer_visualizations,
    extract_simulation_data,
)

__all__ = [
    "generate_analysis_from_logs",
    "create_rxinfer_visualizations",
    "extract_simulation_data",
]
