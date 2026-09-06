"""
ActiveInference.jl execution module for running rendered ActiveInference.jl scripts.

This module contains the ActiveInference.jl script executor and analysis suite
for the GNN Processing Pipeline.
"""

from typing import Any

from .activeinference_runner import (
    execute_activeinference_script,
    find_activeinference_scripts,
    is_julia_available,
    run_activeinference_analysis,
)

__all__: list[Any] = [
    "run_activeinference_analysis",
    "find_activeinference_scripts",
    "execute_activeinference_script",
    "is_julia_available",
]
