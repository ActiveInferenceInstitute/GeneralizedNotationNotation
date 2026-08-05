"""
RxInfer.jl Analysis Module

Per-framework analysis and visualization for RxInfer.jl simulations.
Includes animated HTML visualizations, GIF animations, and interactive dashboard.
"""

from typing import Any

from .analyzer import (
    create_rxinfer_visualizations,
    extract_simulation_data,
    generate_analysis_from_logs,
)
from .animator import generate_animated_html
from .dashboard import generate_dashboard
from .gif_animator import generate_gif_animation

__all__: list[Any] = [
    "generate_analysis_from_logs",
    "create_rxinfer_visualizations",
    "extract_simulation_data",
    "generate_animated_html",
    "generate_gif_animation",
    "generate_dashboard",
]
