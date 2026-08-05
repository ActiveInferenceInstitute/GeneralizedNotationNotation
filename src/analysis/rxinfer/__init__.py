"""
RxInfer.jl Analysis Module

Per-framework analysis and visualization for RxInfer.jl simulations.
Includes animated HTML visualizations for belief/state/VFE evolution.
"""

from typing import Any

from .analyzer import (
    create_rxinfer_visualizations,
    extract_simulation_data,
    generate_analysis_from_logs,
)
from .animator import generate_animated_html

__all__: list[Any] = [
    "generate_analysis_from_logs",
    "create_rxinfer_visualizations",
    "extract_simulation_data",
    "generate_animated_html",
]
