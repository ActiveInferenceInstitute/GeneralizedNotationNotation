#!/usr/bin/env python3
"""
PyMDP Execution Module

This module provides PyMDP simulation execution capabilities for the GNN pipeline.
It includes utilities for running PyMDP simulations configured from GNN specifications.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

from .pymdp_simulation import PyMDPSimulation
from .pymdp_utils import (
    convert_numpy_for_json,
    safe_json_dump,
    safe_pickle_dump,
    clean_trace_for_serialization,
    save_simulation_results,
    generate_simulation_summary,
    create_output_directory_with_timestamp,
    format_duration,
    extract_gnn_dimensions,
    validate_gnn_pomdp_structure
)
from .pymdp_visualizer import PyMDPVisualizer, create_visualizer, save_all_visualizations
from .execute_pymdp import (
    execute_pymdp_simulation,
    execute_from_gnn_file,
    batch_execute_pymdp,
    execute_pymdp_step
)

__all__ = [
    'PyMDPSimulation',
    'PyMDPVisualizer',
    'create_visualizer', 
    'save_all_visualizations',
    'convert_numpy_for_json',
    'safe_json_dump',
    'safe_pickle_dump',
    'clean_trace_for_serialization',
    'save_simulation_results',
    'generate_simulation_summary',
    'create_output_directory_with_timestamp',
    'format_duration',
    'extract_gnn_dimensions',
    'validate_gnn_pomdp_structure',
    'execute_pymdp_simulation',
    'execute_from_gnn_file',
    'batch_execute_pymdp',
    'execute_pymdp_step'
]

# Legacy support - the execute_pymdp_simulation function is now imported 
# from the execute_pymdp module instead of being defined here

def get_module_info() -> Dict[str, Any]:
    """Get information about the PyMDP execution module."""
    return {
        'module_name': 'PyMDP Execution',
        'version': '1.0.0',
        'description': 'PyMDP simulation execution for GNN pipeline',
        'capabilities': [
            'GNN-to-PyMDP parameter extraction',
            'Configurable POMDP simulations', 
            'Active Inference agent execution',
            'Comprehensive result visualization',
            'Pipeline integration support'
        ],
        'dependencies': ['pymdp', 'numpy', 'matplotlib', 'seaborn'],
        'supported_gnn_formats': ['markdown', 'json', 'yaml']
    } 