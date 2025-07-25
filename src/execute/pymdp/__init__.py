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
    'execute_pymdp_simulation',
    'execute_pymdp_simulation_from_gnn',
    'configure_from_gnn_spec',
    'create_visualizer',
    'save_all_visualizations',
    'convert_numpy_for_json',
    'safe_json_dump',
    'safe_pickle_dump',
    'clean_trace_for_serialization',
    'save_simulation_results',
    'generate_simulation_summary',
    'create_output_directory_with_timestamp',
    'format_duration'
]

def execute_pymdp_simulation(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute PyMDP simulation from GNN specification.
    
    Args:
        gnn_spec: Parsed GNN specification dictionary
        output_dir: Output directory for results
        config_overrides: Optional configuration overrides
        
    Returns:
        Tuple of (success, results_dict)
    """
    try:
        # Create simulation with GNN configuration
        simulation = PyMDPSimulation(gnn_config=gnn_spec)
        
        # Apply configuration overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(simulation, key):
                    setattr(simulation, key, value)
        
        # Create PyMDP model
        agent, matrices = simulation.create_pymdp_model()
        
        if agent is None:
            return False, {'error': 'Failed to create PyMDP model'}
        
        # Run simulation
        results = simulation.run_simulation(output_dir)
        
        if results:
            return True, results
        else:
            return False, {'error': 'Simulation failed to produce results'}
            
    except Exception as e:
        logging.error(f"PyMDP simulation failed: {e}")
        return False, {'error': str(e)}


def execute_pymdp_simulation_from_gnn(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute PyMDP simulation from full GNN specification with matrix extraction.
    
    This function specifically handles GNN specifications with InitialParameterization
    sections and extracts the A, B, C, D, E matrices for authentic Active Inference simulation.
    
    Args:
        gnn_spec: Complete GNN specification dictionary
        output_dir: Output directory for results  
        config_overrides: Optional configuration overrides
        
    Returns:
        Tuple of (success, results_dict)
    """
    try:
        # Use the PyMDP converter to properly extract GNN matrices
        from ...render.pymdp.pymdp_converter import GnnToPyMdpConverter
        
        # Convert GNN specification to PyMDP configuration
        converter = GnnToPyMdpConverter(gnn_spec)
        pymdp_config = converter.convert()
        
        if not pymdp_config["success"]:
            logging.warning(f"GNN conversion failed: {pymdp_config.get('error')}")
            # Fall back to standard execution
            return execute_pymdp_simulation(gnn_spec, output_dir, config_overrides)
        
        # Create enhanced simulation with extracted matrices
        simulation = PyMDPSimulation()
        
        # Configure simulation with extracted matrices
        if pymdp_config.get("matrices"):
            simulation.gnn_matrices = pymdp_config["matrices"]
            logging.info(f"Using extracted GNN matrices: {list(pymdp_config['matrices'].keys())}")
        
        # Configure dimensions from GNN
        if pymdp_config.get("dimensions"):
            dims = pymdp_config["dimensions"]
            simulation.num_states = dims.get("num_states", 3)
            simulation.num_observations = dims.get("num_observations", 3)
            simulation.num_actions = dims.get("num_actions", 3)
            
        # Apply configuration overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(simulation, key):
                    setattr(simulation, key, value)
        
        # Store full GNN spec for reference
        simulation.gnn_spec = gnn_spec
        simulation.model_name = gnn_spec.get('model_name', 'GNN_Model')
        
        # Create PyMDP model with GNN matrices
        agent, matrices = simulation.create_pymdp_model_from_gnn()
        
        if agent is None:
            logging.warning("Failed to create model from GNN matrices, using fallback")
            return execute_pymdp_simulation(gnn_spec, output_dir, config_overrides)
        
        # Run simulation
        results = simulation.run_simulation(output_dir)
        
        if results:
            # Add GNN-specific information to results
            results['gnn_matrices_used'] = list(pymdp_config.get("matrices", {}).keys())
            results['gnn_model_name'] = gnn_spec.get('model_name')
            return True, results
        else:
            return False, {'error': 'Simulation failed to produce results'}
            
    except Exception as e:
        logging.error(f"GNN-based PyMDP simulation failed: {e}")
        # Fall back to standard execution
        return execute_pymdp_simulation(gnn_spec, output_dir, config_overrides)


def configure_from_gnn_spec(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract configuration parameters from GNN specification.
    
    Args:
        gnn_spec: GNN specification dictionary
        
    Returns:
        Configuration dictionary for PyMDP simulation
    """
    config = {}
    
    # Extract model parameters
    if 'model_parameters' in gnn_spec:
        model_params = gnn_spec['model_parameters']
        config['num_states'] = model_params.get('num_hidden_states', 3)
        config['num_observations'] = model_params.get('num_obs', 3)
        config['num_actions'] = model_params.get('num_actions', 3)
    
    # Extract from variables if available
    if 'variables' in gnn_spec:
        for var in gnn_spec['variables']:
            if var.get('name') == 'A' and 'dimensions' in var:
                dims = var['dimensions']
                if len(dims) >= 2:
                    config['num_observations'] = dims[0]
                    config['num_states'] = dims[1]
            elif var.get('name') == 'B' and 'dimensions' in var:
                dims = var['dimensions']
                if len(dims) >= 3:
                    config['num_actions'] = dims[2]
    
    # Extract model metadata
    config['model_name'] = gnn_spec.get('model_name', 'GNN_Model')
    config['model_annotation'] = gnn_spec.get('annotation', '')
    
    return config 