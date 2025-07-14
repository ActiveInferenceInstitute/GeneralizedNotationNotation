"""
ActiveInference.jl execution module for running rendered ActiveInference.jl scripts.

This module contains the ActiveInference.jl script executor and analysis suite 
for the GNN Processing Pipeline.
"""

from .activeinference_runner import (
    run_activeinference_analysis,
    run_analysis_suite,
    find_activeinference_scripts,
    execute_activeinference_script,
    is_julia_available
)

__all__ = [
    'run_activeinference_analysis',
    'run_analysis_suite',
    'find_activeinference_scripts',
    'execute_activeinference_script',
    'is_julia_available'
] 