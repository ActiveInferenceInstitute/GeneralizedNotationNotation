"""
ActiveInference.jl Analysis Module

Per-framework analysis for ActiveInference.jl simulations.
"""

from .analyzer import (
    generate_analysis_from_logs,
)

__all__ = [
    "generate_analysis_from_logs",
]
