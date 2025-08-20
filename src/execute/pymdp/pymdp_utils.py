#!/usr/bin/env python3
"""
PyMDP Utilities Module

Utility functions for PyMDP simulations including JSON serialization,
numpy array handling, and data conversion utilities.

Features:
- Comprehensive numpy type serialization for JSON
- Recursive data structure conversion
- Safe file operations
- Data validation utilities
- Pipeline integration support

Author: GNN PyMDP Integration
Date: 2024
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import logging


def convert_numpy_for_json(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to JSON-serializable types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.complexfloating):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_for_json(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj


def safe_json_dump(data: Any, file_path: Path, indent: int = 2) -> bool:
    """
    Safely dump data to JSON file with numpy type conversion.
    
    Args:
        data: Data to serialize
        file_path: Path to save JSON file
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert numpy types
        serializable_data = convert_numpy_for_json(data)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON file
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=indent, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def safe_pickle_dump(data: Any, file_path: Path) -> bool:
    """
    Safely dump data to pickle file.
    
    Args:
        data: Data to serialize
        file_path: Path to save pickle file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to save pickle to {file_path}: {e}")
        return False


def validate_trace_data(trace: Dict[str, Any]) -> bool:
    """
    Validate trace data structure.
    
    Args:
        trace: Trace dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['episode', 'true_states', 'observations', 'actions', 
                    'rewards', 'beliefs']
    
    # Optional keys that may be present in enhanced traces
    optional_keys = ['policies', 'expected_free_energies', 'variational_free_energies', 'positions']
    
    for key in required_keys:
        if key not in trace:
            logging.warning(f"Missing required key in trace: {key}")
            return False
    
    # Check that lists have consistent lengths (except episode)
    list_keys = [k for k in required_keys if k != 'episode']
    lengths = [len(trace[k]) for k in list_keys if isinstance(trace[k], list)]
    
    if lengths and len(set(lengths)) > 1:
        logging.warning(f"Inconsistent trace lengths: {dict(zip(list_keys, lengths))}")
        return False
    
    return True


def clean_trace_for_serialization(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean trace data for JSON serialization.
    
    Args:
        trace: Raw trace data
        
    Returns:
        Cleaned trace data ready for JSON serialization
    """
    cleaned_trace = {}
    
    for key, value in trace.items():
        if isinstance(value, list):
            cleaned_list = []
            for item in value:
                if item is None:
                    cleaned_list.append(None)
                elif isinstance(item, np.ndarray):
                    cleaned_list.append(item.tolist())
                elif isinstance(item, (np.integer, np.floating)):
                    cleaned_list.append(convert_numpy_for_json(item))
                else:
                    cleaned_list.append(convert_numpy_for_json(item))
            cleaned_trace[key] = cleaned_list
        else:
            cleaned_trace[key] = convert_numpy_for_json(value)
    
    return cleaned_trace


def save_simulation_results(traces: List[Dict], metrics: Dict[str, List], 
                           config: Dict, model_matrices: Optional[Dict],
                           output_dir: Path) -> Dict[str, bool]:
    """
    Save all simulation results with proper error handling.
    
    Args:
        traces: List of episode traces
        metrics: Performance metrics
        config: Simulation configuration
        model_matrices: PyMDP model matrices (optional)
        output_dir: Output directory
        
    Returns:
        Dictionary indicating success/failure of each save operation
    """
    results = {}
    
    # Save configuration
    results['config'] = safe_json_dump(config, output_dir / 'simulation_config.json')
    
    # Save performance metrics
    results['metrics'] = safe_json_dump(metrics, output_dir / 'performance_metrics.json')
    
    # Save traces as pickle (full data)
    results['traces_pickle'] = safe_pickle_dump(traces, output_dir / 'simulation_traces.pkl')
    
    # Save traces as JSON (cleaned data)
    cleaned_traces = [clean_trace_for_serialization(trace) for trace in traces]
    results['traces_json'] = safe_json_dump(cleaned_traces, output_dir / 'simulation_traces.json')
    
    # Save model matrices if provided
    if model_matrices:
        results['matrices'] = safe_pickle_dump(model_matrices, output_dir / 'model_matrices.pkl')
    else:
        results['matrices'] = True  # Not applicable
    
    return results


def calculate_episode_statistics(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate statistics for a single episode.
    
    Args:
        trace: Episode trace data
        
    Returns:
        Dictionary of episode statistics
    """
    stats = {
        'episode': trace.get('episode', 0),
        'total_reward': 0.0,
        'episode_length': 0,
        'mean_belief_entropy': 0.0,
        'final_state': None,
        'success': False
    }
    
    # Calculate total reward
    if 'rewards' in trace and trace['rewards']:
        stats['total_reward'] = float(np.sum(trace['rewards']))
    
    # Episode length
    if 'actions' in trace and trace['actions']:
        stats['episode_length'] = len(trace['actions'])
    
    # Mean belief entropy
    if 'beliefs' in trace and trace['beliefs']:
        entropies = []
        for belief in trace['beliefs']:
            if belief is not None and len(belief) > 0:
                # Calculate entropy: -sum(p * log(p))
                belief_array = np.array(belief)
                belief_array = belief_array + 1e-16  # Avoid log(0)
                entropy = -np.sum(belief_array * np.log(belief_array))
                entropies.append(entropy)
        
        if entropies:
            stats['mean_belief_entropy'] = float(np.mean(entropies))
    
    # Final state
    if 'true_states' in trace and trace['true_states']:
        stats['final_state'] = trace['true_states'][-1]
    
    # Check success (threshold-based on total reward)
    if 'rewards' in trace and trace['rewards']:
        stats['success'] = stats['total_reward'] > 0.0
    
    return stats


def generate_simulation_summary(all_traces: List[Dict], 
                               performance_metrics: Dict[str, List]) -> Dict[str, Any]:
    """
    Generate a comprehensive simulation summary.
    
    Args:
        all_traces: List of all episode traces
        performance_metrics: Performance metrics dictionary
        
    Returns:
        Summary dictionary
    """
    summary = {
        'total_episodes': len(all_traces),
        'successful_episodes': 0,
        'total_steps': 0,
        'average_reward': 0.0,
        'average_episode_length': 0.0,
        'success_rate': 0.0,
        'episode_statistics': []
    }
    
    # Calculate per-episode statistics
    for trace in all_traces:
        episode_stats = calculate_episode_statistics(trace)
        summary['episode_statistics'].append(episode_stats)
        
        if episode_stats['success']:
            summary['successful_episodes'] += 1
        
        summary['total_steps'] += episode_stats['episode_length']
    
    # Calculate averages
    if summary['total_episodes'] > 0:
        summary['success_rate'] = summary['successful_episodes'] / summary['total_episodes']
        summary['average_episode_length'] = summary['total_steps'] / summary['total_episodes']
        
        if 'episode_rewards' in performance_metrics and performance_metrics['episode_rewards']:
            summary['average_reward'] = float(np.mean(performance_metrics['episode_rewards']))
    
    return summary


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def create_output_directory_with_timestamp(base_dir: Union[str, Path], 
                                          prefix: str = "simulation") -> Path:
    """
    Create output directory with timestamp.
    
    Args:
        base_dir: Base directory for output
        prefix: Prefix for directory name
        
    Returns:
        Path to created directory
    """
    from datetime import datetime
    
    # Accept str paths as well as Path
    base_dir = Path(base_dir) if not isinstance(base_dir, Path) else base_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"{prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def parse_gnn_matrix_string(matrix_str: str) -> np.ndarray:
    """
    Parse matrix string from GNN initial parameterization format.
    
    Args:
        matrix_str: String representation of matrix from GNN file
        
    Returns:
        Numpy array representation of the matrix
    """
    try:
        # Remove comments and extra whitespace
        cleaned = matrix_str.strip()
        
        # Handle GNN tuple format: {(row1), (row2), ...}
        if cleaned.startswith('{') and cleaned.endswith('}'):
            cleaned = cleaned[1:-1]  # Remove outer braces
        
        # Parse individual rows
        rows = []
        if '(' in cleaned:
            # Parse tuple format
            import re
            tuple_pattern = r'\((.*?)\)'
            tuple_matches = re.findall(tuple_pattern, cleaned)
            
            for tuple_str in tuple_matches:
                row_values = [float(x.strip()) for x in tuple_str.split(',')]
                rows.append(row_values)
        else:
            # Parse simple comma-separated format
            if ',' in cleaned:
                values = [float(x.strip()) for x in cleaned.split(',')]
                rows = [values]  # Single row
        
        if rows:
            return np.array(rows)
        else:
            # Fallback parsing
            return np.array(eval(matrix_str))
            
    except Exception as e:
        logging.warning(f"Failed to parse GNN matrix string: {e}")
        # Return default identity matrix
        return np.eye(3)


def parse_gnn_vector_string(vector_str: str) -> np.ndarray:
    """
    Parse vector string from GNN initial parameterization format.
    
    Args:
        vector_str: String representation of vector from GNN file
        
    Returns:
        Numpy array representation of the vector
    """
    try:
        # Remove comments and extra whitespace
        cleaned = vector_str.strip()
        
        # Handle GNN tuple format: {(val1, val2, ...)}
        if cleaned.startswith('{') and cleaned.endswith('}'):
            cleaned = cleaned[1:-1]  # Remove outer braces
        
        if cleaned.startswith('(') and cleaned.endswith(')'):
            cleaned = cleaned[1:-1]  # Remove outer parentheses
        
        # Parse comma-separated values
        values = [float(x.strip()) for x in cleaned.split(',')]
        return np.array(values)
        
    except Exception as e:
        logging.warning(f"Failed to parse GNN vector string: {e}")
        # Return default vector
        return np.array([0.0, 0.0, 1.0])


def extract_gnn_dimensions(gnn_spec: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract POMDP dimensions from GNN specification.
    
    Args:
        gnn_spec: Parsed GNN specification
        
    Returns:
        Dictionary containing extracted dimensions
    """
    dimensions = {
        'num_states': 3,
        'num_observations': 3,
        'num_actions': 3
    }
    
    # Try model parameters first
    model_params = gnn_spec.get('model_parameters', {})
    if model_params:
        dimensions['num_states'] = int(model_params.get('num_hidden_states', 3))
        dimensions['num_observations'] = int(model_params.get('num_obs', 3))
        dimensions['num_actions'] = int(model_params.get('num_actions', 3))
        return dimensions
    
    # Try to infer from state space variables
    variables = gnn_spec.get('variables', [])
    for var in variables:
        var_name = var.get('name', '').upper()
        dimensions_list = var.get('dimensions', [])
        
        if var_name == 'A' and len(dimensions_list) >= 2:
            dimensions['num_observations'] = dimensions_list[0]
            dimensions['num_states'] = dimensions_list[1]
        elif var_name == 'B' and len(dimensions_list) >= 3:
            dimensions['num_states'] = dimensions_list[0]  # next states
            dimensions['num_actions'] = dimensions_list[2]
        elif var_name == 'C' and len(dimensions_list) >= 1:
            dimensions['num_observations'] = dimensions_list[0]
        elif var_name == 'D' and len(dimensions_list) >= 1:
            dimensions['num_states'] = dimensions_list[0]
    
    return dimensions


def validate_gnn_pomdp_structure(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that GNN specification contains valid POMDP structure.
    
    Args:
        gnn_spec: Parsed GNN specification
        
    Returns:
        Validation result dictionary
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'dimensions': {}
    }
    
    # Extract dimensions
    try:
        result['dimensions'] = extract_gnn_dimensions(gnn_spec)
    except Exception as e:
        result['errors'].append(f"Failed to extract dimensions: {e}")
        result['valid'] = False
    
    # Check for required POMDP matrices
    initial_params = gnn_spec.get('initial_parameterization', {})
    required_matrices = ['A', 'B', 'C', 'D']
    
    for matrix_name in required_matrices:
        if matrix_name not in initial_params:
            result['warnings'].append(f"Missing initial parameterization for {matrix_name} matrix")
    
    # Check model name
    if not gnn_spec.get('model_name'):
        result['warnings'].append("No model name specified")
    
    return result


if __name__ == "__main__":
    print("PyMDP Utilities - Helper functions for PyMDP simulations integrated with GNN pipeline")
    print("This module should be imported and used with the main simulation class.") 