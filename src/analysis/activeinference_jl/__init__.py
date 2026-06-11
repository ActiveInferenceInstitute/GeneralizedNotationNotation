"""
ActiveInference.jl Analysis Module

Per-framework analysis for ActiveInference.jl simulations.
"""

from typing import Any

from .analyzer import (
    generate_analysis_from_logs,
)

__all__: list[Any] = [
    "generate_analysis_from_logs",
]
