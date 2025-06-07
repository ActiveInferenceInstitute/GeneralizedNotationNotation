"""
Module for generating RxInfer.jl configuration files from parsed GNN content.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import toml

logger = logging.getLogger(__name__)

def format_toml_value(value: Any) -> str:
    """
    Format a Python value for TOML configuration.
    
    Args:
        value: The Python value to format
        
    Returns:
        A string representation suitable for TOML
    """
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, (list, tuple)):
        if all(isinstance(x, (int, float)) for x in value):
            # Simple array of numbers
            return "[" + ", ".join(str(x) for x in value) + "]"
        elif all(isinstance(x, str) for x in value):
            # Array of strings
            return "[" + ", ".join([f'"{x}"' for x in value]) + "]"
        elif all(isinstance(x, (list, tuple)) for x in value):
            # Matrix (2D array)
            rows = []
            for row in value:
                if all(isinstance(x, (int, float)) for x in row):
                    rows.append("[" + ", ".join(str(x) for x in row) + "]")
                else:
                    rows.append("[" + ", ".join(format_toml_value(x) for x in row) + "]")
            return "[\n    " + ",\n    ".join(rows) + "\n]"
    
    # Default case, try to convert to string
    return str(value)

def generate_toml_section(section_name: str, parameters: Dict[str, Any], indent_level: int = 0) -> str:
    """
    Generate a TOML section with parameters.
    
    Args:
        section_name: The name of the section
        parameters: Dictionary of parameters for the section
        indent_level: The indentation level for the section
        
    Returns:
        TOML section as a string
    """
    indent = ' ' * (4 * indent_level)
    lines = [f"{indent}[{section_name}]"]
    
    # Sort parameters for consistent output
    for key, value in sorted(parameters.items()):
        if isinstance(value, dict):
            # Nested section
            nested_section = f"{section_name}.{key}"
            lines.append("")
            lines.append(generate_toml_section(nested_section, value, indent_level))
        else:
            # Simple key-value pair
            lines.append(f"{indent}{key} = {format_toml_value(value)}")
    
    return "\n".join(lines)

def generate_rxinfer_config(parsed_gnn: Dict[str, Any], output_path: Path) -> bool:
    """
    Generate a RxInfer.jl configuration file from parsed GNN content.
    
    Args:
        parsed_gnn: Dictionary of parsed GNN content
        output_path: Path to write the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract parameters from parsed GNN
        initial_params = parsed_gnn.get('initial_parameters', {})
        
        # Start with a comment
        toml_lines = ["# Multi-agent Trajectory Planning Configuration", ""]
        
        # Organize parameters into sections
        model_params = {}
        matrices = {}
        priors = {}
        visualization = {}
        environments = {}
        agents = []
        experiments = {}
        
        # Model parameters
        for param in ['dt', 'gamma', 'nr_steps', 'nr_iterations', 'nr_agents', 
                     'softmin_temperature', 'intermediate_steps', 'save_intermediates']:
            if param in initial_params:
                model_params[param] = initial_params[param]
        
        # State space matrices
        for matrix in ['A', 'B', 'C']:
            if matrix in initial_params:
                matrices[matrix] = initial_params[matrix]
        
        # Prior distributions
        for prior in ['initial_state_variance', 'control_variance', 'goal_constraint_variance',
                     'gamma_shape', 'gamma_scale_factor']:
            if prior in initial_params:
                priors[prior] = initial_params[prior]
        
        # Visualization parameters
        for viz in ['x_limits', 'y_limits', 'fps', 'heatmap_resolution', 'plot_width',
                   'plot_height', 'agent_alpha', 'target_alpha', 'color_palette']:
            if viz in initial_params:
                visualization[viz] = initial_params[viz]
        
        # Environment definitions
        # Door environment
        door_env = {
            'description': "Two parallel walls with a gap between them",
            'obstacles': [
                {
                    'center': list(initial_params.get('door_obstacle_center_1', (-40.0, 0.0))),
                    'size': list(initial_params.get('door_obstacle_size_1', (70.0, 5.0)))
                },
                {
                    'center': list(initial_params.get('door_obstacle_center_2', (40.0, 0.0))),
                    'size': list(initial_params.get('door_obstacle_size_2', (70.0, 5.0)))
                }
            ]
        }
        
        # Wall environment
        wall_env = {
            'description': "A single wall obstacle in the center",
            'obstacles': [
                {
                    'center': list(initial_params.get('wall_obstacle_center', (0.0, 0.0))),
                    'size': list(initial_params.get('wall_obstacle_size', (10.0, 5.0)))
                }
            ]
        }
        
        # Combined environment
        combined_env = {
            'description': "A combination of walls and obstacles",
            'obstacles': [
                {
                    'center': list(initial_params.get('combined_obstacle_center_1', (-50.0, 0.0))),
                    'size': list(initial_params.get('combined_obstacle_size_1', (70.0, 2.0)))
                },
                {
                    'center': list(initial_params.get('combined_obstacle_center_2', (50.0, 0.0))),
                    'size': list(initial_params.get('combined_obstacle_size_2', (70.0, 2.0)))
                },
                {
                    'center': list(initial_params.get('combined_obstacle_center_3', (5.0, -1.0))),
                    'size': list(initial_params.get('combined_obstacle_size_3', (3.0, 10.0)))
                }
            ]
        }
        
        environments = {
            'door': door_env,
            'wall': wall_env,
            'combined': combined_env
        }
        
        # Agent configurations
        for i in range(1, 5):  # 4 agents
            agent_prefix = f'agent{i}_'
            if f'{agent_prefix}id' in initial_params:
                agent = {
                    'id': initial_params.get(f'{agent_prefix}id', i),
                    'radius': initial_params.get(f'{agent_prefix}radius', 1.0),
                    'initial_position': list(initial_params.get(f'{agent_prefix}initial_position', (0.0, 0.0))),
                    'target_position': list(initial_params.get(f'{agent_prefix}target_position', (0.0, 0.0)))
                }
                agents.append(agent)
        
        # Experiment configurations
        for param in ['seeds', 'results_dir', 'animation_template', 'control_vis_filename',
                     'obstacle_distance_filename', 'path_uncertainty_filename', 'convergence_filename']:
            key = 'experiment_seeds' if param == 'seeds' else param
            if key in initial_params:
                value = initial_params[key]
                if param == 'seeds' and isinstance(value, tuple):
                    value = list(value)
                experiments[param] = value
        
        # Add matrices to model section
        if matrices:
            model_params['matrices'] = matrices
        
        # Generate TOML sections
        toml_lines.append(generate_toml_section('model', model_params))
        toml_lines.append("")
        toml_lines.append(generate_toml_section('priors', priors))
        toml_lines.append("")
        toml_lines.append(generate_toml_section('visualization', visualization))
        toml_lines.append("")
        
        # Environment sections
        env_lines = ["# Environment definitions"]
        for env_name, env_data in environments.items():
            # Handle special case for obstacles which need to be separate sections
            obstacles = env_data.pop('obstacles', [])
            env_section = generate_toml_section(f'environments.{env_name}', 
                                              {'description': env_data.get('description', '')})
            env_lines.append(env_section)
            env_lines.append("")
            
            for i, obstacle in enumerate(obstacles):
                obstacle_section = f"[[environments.{env_name}.obstacles]]"
                env_lines.append(obstacle_section)
                for k, v in obstacle.items():
                    env_lines.append(f"center = {format_toml_value(v) if k == 'center' else ''}")
                    env_lines.append(f"size = {format_toml_value(v) if k == 'size' else ''}")
                env_lines.append("")
        
        toml_lines.append("\n".join(env_lines))
        
        # Agent configurations
        toml_lines.append("# Agent configurations")
        for agent in agents:
            toml_lines.append("[[agents]]")
            for k, v in agent.items():
                toml_lines.append(f"{k} = {format_toml_value(v)}")
            toml_lines.append("")
        
        # Experiment configurations
        toml_lines.append(generate_toml_section('experiments', experiments))
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(toml_lines))
        
        logger.info(f"Successfully generated RxInfer.jl configuration file: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating RxInfer.jl configuration: {e}", exc_info=True)
        return False

def generate_rxinfer_config_from_spec(gnn_spec: dict, logger: logging.Logger) -> str:
    """
    Generates an RxInfer TOML configuration string from a GNN specification dictionary.
    This is a simplified version that works on an already-parsed dictionary (spec).

    Args:
        gnn_spec (dict): The GNN specification as a Python dictionary.
        logger (logging.Logger): Logger instance.

    Returns:
        str: A string containing the TOML configuration.
    """
    logger.debug("Generating RxInfer TOML config from GNN spec dictionary.")
    
    model_name = gnn_spec.get("ModelName", "Unknown GNN Model")

    # This is a placeholder implementation. A proper implementation would
    # map the gnn_spec dictionary to the TOML structure in a more detailed way,
    # similar to how generate_rxinfer_config works with the gnn_model object.
    
    toml_config = {
        "model": {
            "name": model_name,
            "description": gnn_spec.get("ModelAnnotation", "Configuration generated from GNN specification."),
            "dt": 1.0,
            "gamma": 1.0,
            "nr_steps": 40,
            "nr_iterations": 500
        },
        "priors": {
            "initial_state_variance": 100.0,
            "control_variance": 0.2
        }
    }

    return toml.dumps(toml_config)

if __name__ == "__main__":
    # Simple test if run directly
    import sys
    import json
    
    if len(sys.argv) > 2:
        # Load parsed GNN from a JSON file
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            parsed_gnn = json.load(f)
        
        output_path = Path(sys.argv[2])
        success = generate_rxinfer_config(parsed_gnn, output_path)
        print(f"Configuration generation {'successful' if success else 'failed'}.") 