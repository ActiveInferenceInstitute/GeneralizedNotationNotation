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
    format_duration
)
from .pymdp_visualizer import PyMDPVisualizer, create_visualizer, save_all_visualizations

__all__ = [
    'PyMDPSimulation',
    'PyMDPVisualizer',
    'convert_numpy_for_json',
    'safe_json_dump',
    'safe_pickle_dump',
    'clean_trace_for_serialization',
    'save_simulation_results',
    'generate_simulation_summary',
    'create_output_directory_with_timestamp',
    'format_duration',
    'create_visualizer',
    'save_all_visualizations'
]

def execute_pymdp_simulation(
    gnn_config: Dict[str, Any],
    output_dir: Path,
    **kwargs
) -> Tuple[Dict[str, Any], Path]:
    """
    Execute a PyMDP simulation configured from GNN specifications.
    
    Args:
        gnn_config: Parsed GNN configuration containing POMDP parameters
        output_dir: Directory to save simulation results
        **kwargs: Additional simulation parameters
        
    Returns:
        Tuple of (results_dict, output_path)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting PyMDP simulation with GNN config: {gnn_config.get('model_name', 'unknown')}")
    
    # Create PyMDP simulation instance
    simulation = PyMDPSimulation(gnn_config=gnn_config, **kwargs)
    
    # Run simulation
    results = simulation.run_simulation()
    
    # Save results
    output_path = save_simulation_results(results, output_dir)
    
    # Generate visualizations
    visualizer = create_visualizer(results)
    viz_path = save_all_visualizations(visualizer, output_dir)
    
    logger.info(f"PyMDP simulation completed. Results saved to: {output_path}")
    
    return results, output_path 