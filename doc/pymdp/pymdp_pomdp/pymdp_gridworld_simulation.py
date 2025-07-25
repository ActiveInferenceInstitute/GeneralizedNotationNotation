#!/usr/bin/env python3
"""
PyMDP Gridworld POMDP Simulation - Pipeline Integration

A comprehensive PyMDP gridworld simulation that integrates with the GNN pipeline.
This simulation can be configured from GNN specifications and executed through
the pipeline's render and execute steps.

Pipeline Integration:
- Reads configuration from GNN POMDP specifications
- Integrates with src/render/pymdp/ for code generation
- Uses src/execute/pymdp/ for simulation execution
- Supports pipeline output structure and logging

Features:
- Authentic PyMDP Agent implementation
- GNN-driven configuration
- Comprehensive visualization and analysis
- Pipeline-compatible structure

Author: GNN PyMDP Integration
Date: 2024
"""

import numpy as np
import json
import pickle
import time
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for pipeline integration
script_dir = Path(__file__).parent
gnn_root = script_dir.parent.parent.parent
src_dir = gnn_root / "src"
sys.path.insert(0, str(src_dir))

# Pipeline imports
try:
    from gnn.parsers.markdown_parser import MarkdownGNNParser
    from render.pymdp.pymdp_converter import GNNToPyMDPConverter
    from execute.pymdp.pymdp_simulation import PyMDPSimulation
    from execute.pymdp.pymdp_utils import save_simulation_results, create_output_directory_with_timestamp
    from execute.pymdp.pymdp_visualizer import PyMDPVisualizer
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Pipeline modules not available: {e}")
    print("Running in standalone mode with default configuration")
    PIPELINE_AVAILABLE = False

# PyMDP imports
try:
    from pymdp import utils
    from pymdp.agent import Agent
    PYMDP_AVAILABLE = True
except ImportError:
    print("PyMDP not available. Install with: pip install inferactively-pymdp")
    PYMDP_AVAILABLE = False

# =============================================================================
# CONFIGURATION VARIABLES - GNN PIPELINE INTEGRATION
# =============================================================================

# GNN Integration - These parameters can be overridden by GNN specifications
# Default values provided for standalone operation
def load_gnn_config(gnn_config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from GNN specifications or use defaults.
    
    This function integrates with the GNN pipeline (11_render.py -> 12_execute.py)
    to extract POMDP parameters from parsed GNN files, specifically from
    actinf_pomdp_agent.md or other GNN POMDP specifications.
    
    Args:
        gnn_config_path: Path to GNN-derived configuration JSON
        
    Returns:
        Configuration dictionary with POMDP parameters
    """
    # Default configuration (used when no GNN config provided)
    default_config = {
        # Basic environment parameters
        'GRID_SIZE': 4,
        'NUM_STATES': 16,  # GRID_SIZE^2
        'NUM_OBSERVATIONS': 16,  # Full observability case
        'NUM_ACTIONS': 4,  # Up, Down, Left, Right
        'NUM_STEPS': 20,
        
        # Agent goal configuration
        'GOAL_LOCATION': 15,  # Bottom-right corner (index 15 in 4x4 grid)
        'REWARD_MAGNITUDE': 2.0,
        
        # POMDP generative model parameters
        'OBSERVATION_NOISE': 0.1,  # Noise in observation model (A matrix)
        'TRANSITION_NOISE': 0.05,  # Noise in transition model (B matrix)
        'PREFERENCE_PRECISION': 2.0,  # Temperature parameter for preferences
        'PRIOR_PRECISION': 1.0,  # Precision of prior beliefs (D vector)
        
        # Active Inference parameters
        'POLICY_HORIZON': 3,  # Planning horizon for policy inference
        'POLICY_PRECISION': 16.0,  # Precision parameter for policy prior
        'ACTION_PRECISION': 16.0,  # Precision for action selection
        
        # Learning parameters
        'LEARNING_RATE_A': 0.1,  # Learning rate for A matrix updates
        'LEARNING_RATE_B': 0.1,  # Learning rate for B matrix updates
        'ENABLE_LEARNING': True,  # Whether to enable parameter learning
        
        # Visualization and output
        'SAVE_RESULTS': True,
        'GENERATE_PLOTS': True,
        'VERBOSE': True
    }
    
    if gnn_config_path and Path(gnn_config_path).exists():
        try:
            with open(gnn_config_path, 'r') as f:
                gnn_config = json.load(f)
            
            # Merge GNN config with defaults, GNN values take precedence
            config = {**default_config, **gnn_config}
            
            # Validate and derive dependent parameters
            config['NUM_STATES'] = config['GRID_SIZE'] ** 2
            if 'NUM_OBSERVATIONS' not in gnn_config:
                config['NUM_OBSERVATIONS'] = config['NUM_STATES']  # Default to full observability
                
            print(f"✓ Loaded GNN configuration from: {gnn_config_path}")
            return config
            
        except Exception as e:
            print(f"⚠ Warning: Could not load GNN config from {gnn_config_path}: {e}")
            print("  Using default configuration")
            
    return default_config

# Load configuration (can be overridden by pipeline)
CONFIG = load_gnn_config()

# Extract configuration variables for backward compatibility
GRID_SIZE = CONFIG['GRID_SIZE']
NUM_STATES = CONFIG['NUM_STATES'] 
NUM_OBSERVATIONS = CONFIG['NUM_OBSERVATIONS']
NUM_ACTIONS = CONFIG['NUM_ACTIONS']
NUM_STEPS = CONFIG['NUM_STEPS']
GOAL_LOCATION = CONFIG['GOAL_LOCATION']
REWARD_MAGNITUDE = CONFIG['REWARD_MAGNITUDE']

# =============================================================================
# PIPELINE INTEGRATION FUNCTIONS
# =============================================================================

def run_simulation_from_gnn(gnn_config_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Main entry point for running PyMDP simulation from GNN configuration.
    
    This function is designed to be called by the pipeline execution step
    (12_execute.py) after GNN parsing and rendering (11_render.py).
    
    Args:
        gnn_config_path: Path to GNN-derived configuration JSON
        output_dir: Directory for saving results
        
    Returns:
        Dictionary containing simulation results and metadata
    """
    # Load GNN-derived configuration
    global CONFIG
    CONFIG = load_gnn_config(gnn_config_path)
    
    # Update global variables from config
    globals().update({
        'GRID_SIZE': CONFIG['GRID_SIZE'],
        'NUM_STATES': CONFIG['NUM_STATES'],
        'NUM_OBSERVATIONS': CONFIG['NUM_OBSERVATIONS'], 
        'NUM_ACTIONS': CONFIG['NUM_ACTIONS'],
        'NUM_STEPS': CONFIG['NUM_STEPS'],
        'GOAL_LOCATION': CONFIG['GOAL_LOCATION'],
        'REWARD_MAGNITUDE': CONFIG['REWARD_MAGNITUDE']
    })
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(f"output_pymdp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"RUNNING PYMDP SIMULATION FROM GNN CONFIGURATION")
    print(f"{'='*80}")
    print(f"Configuration source: {gnn_config_path}")
    print(f"Output directory: {output_path}")
    print(f"Grid size: {CONFIG['GRID_SIZE']}x{CONFIG['GRID_SIZE']}")
    print(f"Goal location: {CONFIG['GOAL_LOCATION']}")
    print(f"Simulation steps: {CONFIG['NUM_STEPS']}")
    
    # Run the simulation
    try:
        simulation_results = run_simulation()
        
        # Add metadata
        simulation_results['metadata'] = {
            'gnn_config_path': gnn_config_path,
            'output_directory': str(output_path),
            'configuration': CONFIG,
            'execution_timestamp': datetime.now().isoformat(),
            'pipeline_integration': True
        }
        
        # Save results if requested
        if CONFIG.get('SAVE_RESULTS', True):
            results_path = output_path / 'simulation_results.json'
            with open(results_path, 'w') as f:
                json.dump(convert_numpy_for_json(simulation_results), f, indent=2)
            print(f"✓ Results saved to: {results_path}")
        
        # Generate visualizations if requested  
        if CONFIG.get('GENERATE_PLOTS', True):
            visualizer = create_visualizer(
                grid_size=CONFIG['GRID_SIZE'],
                goal_location=CONFIG['GOAL_LOCATION']
            )
            viz_results = save_all_visualizations(
                visualizer, 
                simulation_results,
                output_path
            )
            simulation_results['visualizations'] = viz_results
            print(f"✓ Visualizations saved to: {output_path}")
        
        print(f"\n✓ PyMDP simulation completed successfully!")
        print(f"✓ Total execution time: {simulation_results.get('execution_time', 'N/A')}")
        
        return simulation_results
        
    except Exception as e:
        error_msg = f"PyMDP simulation failed: {str(e)}"
        print(f"✗ {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'metadata': {
                'gnn_config_path': gnn_config_path,
                'execution_timestamp': datetime.now().isoformat(),
                'pipeline_integration': True
            }
        }

def validate_gnn_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate GNN-derived configuration parameters.
    
    Args:
        config: Configuration dictionary from GNN parsing
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Required parameters
    required_params = ['GRID_SIZE', 'NUM_ACTIONS', 'NUM_STEPS']
    for param in required_params:
        if param not in config:
            errors.append(f"Missing required parameter: {param}")
    
    # Parameter validation
    if 'GRID_SIZE' in config:
        if not isinstance(config['GRID_SIZE'], int) or config['GRID_SIZE'] < 2:
            errors.append("GRID_SIZE must be integer >= 2")
            
    if 'NUM_ACTIONS' in config:
        if not isinstance(config['NUM_ACTIONS'], int) or config['NUM_ACTIONS'] < 1:
            errors.append("NUM_ACTIONS must be positive integer")
            
    if 'NUM_STEPS' in config:
        if not isinstance(config['NUM_STEPS'], int) or config['NUM_STEPS'] < 1:
            errors.append("NUM_STEPS must be positive integer")
    
    if 'GOAL_LOCATION' in config and 'GRID_SIZE' in config:
        max_location = config['GRID_SIZE'] ** 2 - 1
        if config['GOAL_LOCATION'] > max_location:
            errors.append(f"GOAL_LOCATION must be <= {max_location} for {config['GRID_SIZE']}x{config['GRID_SIZE']} grid")
    
    return len(errors) == 0, errors

# =============================================================================
# POMDP GENERATIVE MODEL CONFIGURATION (GNN-CONFIGURABLE)
# =============================================================================

def create_pymdp_model_from_config(config: Dict[str, Any]) -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Create PyMDP model matrices configured from GNN specifications.
    
    This function constructs a complete POMDP generative model using real PyMDP
    utilities and configuration extracted from GNN specifications.
    
    Args:
        config: Configuration dictionary from GNN parsing
        
    Returns:
        tuple: (agent, model_matrices) where agent is a PyMDP Agent instance
               and model_matrices contains the A, B, C, D arrays
    
    POMDP Generative Model Components:
    
    A matrix [NUM_OBSERVATIONS, NUM_STATES]: 
        Observation likelihood P(observation | hidden_state)
        - Maps hidden grid positions to sensory observations
        - Configured from GNN observation model specifications
    
    B matrix [NUM_STATES, NUM_STATES, NUM_ACTIONS]:
        Transition dynamics P(next_state | current_state, action)  
        - Each B[:,:,action] slice represents transition probabilities for that action
        - Rows = FROM states, Columns = TO states
        - Configured from GNN state space and action specifications
    
    C vector [NUM_OBSERVATIONS]:
        Prior preferences log P(observation)
        - Higher values = more preferred observations
        - Configured from GNN goal and reward specifications
    
    D vector [NUM_STATES]: 
        Initial state prior P(initial_state)
        - Starting belief over hidden states
        - Configured from GNN initial state specifications
    """
    if not PYMDP_AVAILABLE:
        raise ImportError("PyMDP not available")
        
    grid_size = config.get("grid_size", 4)
    perception_noise = config.get("perception_noise", 0.1)
    goal_pos = config.get("goal_position", [3, 3])
    start_pos = config.get("start_position", [0, 0])
    walls = set(map(tuple, config.get("walls", [])))
    
    # State space: grid positions (flattened)
    num_states = grid_size * grid_size
    num_observations = num_states  # Direct observation of position (noisy)
    num_actions = 4  # up, down, left, right
    
    # A matrix: Observation model P(obs|state) - configured from GNN
    A = utils.obj_array(1)
    A[0] = np.eye(num_observations) * (1 - perception_noise) + \
           (perception_noise / num_observations) * np.ones((num_observations, num_states))
    A[0] = utils.norm_dist(A[0])
    
    # B matrix: Transition model P(next_state|state,action) - configured from GNN
    B = utils.obj_array(1)
    B[0] = np.zeros((num_states, num_states, num_actions))
    
    # Build transition matrices from GNN action specifications
    for s in range(num_states):
        row, col = divmod(s, grid_size)
        
        for action in range(num_actions):
            # Default: stay in same state
            next_row, next_col = row, col
            
            # Apply action if valid
            if action == 0 and row > 0:  # up
                next_row = row - 1
            elif action == 1 and row < grid_size - 1:  # down
                next_row = row + 1
            elif action == 2 and col > 0:  # left
                next_col = col - 1
            elif action == 3 and col < grid_size - 1:  # right
                next_col = col + 1
                
            # Check for walls from GNN specifications
            if (next_row, next_col) in walls:
                next_row, next_col = row, col  # bounce back
                
            next_state = next_row * grid_size + next_col
            B[0][next_state, s, action] = 1.0
    
    # C vector: Preferences - configured from GNN goals and rewards
    C = utils.obj_array(1)
    C[0] = np.zeros(num_observations)
    
    # Set preferences from GNN reward specifications
    goal_state = goal_pos[0] * grid_size + goal_pos[1]
    rewards = config.get("rewards", {})
    C[0][goal_state] = rewards.get("goal", 10.0)  # High preference for goal
    
    # Wall penalties from GNN
    for wall_pos in walls:
        wall_state = wall_pos[0] * grid_size + wall_pos[1]
        if wall_state < num_observations:
            C[0][wall_state] = rewards.get("wall", -1.0)
    
    # D vector: Initial state distribution - configured from GNN
    D = utils.obj_array(1)
    D[0] = np.zeros(num_states)
    start_state = start_pos[0] * grid_size + start_pos[1]
    D[0][start_state] = 1.0
    D[0] = utils.norm_dist(D[0])
    
    # Create PyMDP agent with GNN-configured parameters
    agent_params = {
        'A': A,
        'B': B, 
        'C': C,
        'D': D,
        'use_param_info_gain': config.get("use_param_info_gain", True),
        'use_states_info_gain': config.get("use_states_info_gain", True),
        'lr_pA': config.get("learning_rate", 0.5),
        'lr_pB': config.get("learning_rate", 0.5),
        'alpha': config.get("alpha", 16.0),
        'gamma': config.get("gamma", 16.0),
        'action_precision': config.get("action_precision", 16.0)
    }
    
    agent = Agent(**agent_params)
    
    model_matrices = {
        'A': A[0],
        'B': B[0], 
        'C': C[0],
        'D': D[0],
        'config': config
    }
    
    return agent, model_matrices

# =============================================================================
# PIPELINE INTEGRATION FUNCTIONS
# =============================================================================

def run_pipeline_simulation(gnn_file_path: Optional[Path] = None, 
                           output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run PyMDP simulation configured from GNN pipeline.
    
    Args:
        gnn_file_path: Path to GNN specification file
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing simulation results and metadata
    """
    # Load configuration from GNN
    config = load_gnn_configuration(gnn_file_path)
    
    # Create output directory
    if output_dir is None:
        output_dir = create_output_directory_with_timestamp("pymdp_simulation")
    
    # Create and run simulation
    if PIPELINE_AVAILABLE:
        simulation = PyMDPSimulation(config)
        results = simulation.run()
        
        # Save results using pipeline utilities
        save_simulation_results(results, output_dir)
        
        # Generate visualizations
        visualizer = PyMDPVisualizer(config)
        visualizer.visualize_results(results, output_dir)
        
    else:
        # Fallback to direct implementation
        results = run_standalone_simulation(config, output_dir)
    
    return results

def run_standalone_simulation(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Run simulation in standalone mode without full pipeline.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing simulation results
    """
    if not PYMDP_AVAILABLE:
        raise ImportError("PyMDP not available for standalone simulation")
        
    print(f"Running PyMDP simulation with configuration: {config}")
    
    # Create model from configuration
    agent, model_matrices = create_pymdp_model_from_config(config)
    
    # Run simulation
    num_timesteps = config.get("num_timesteps", 20)
    num_episodes = config.get("num_episodes", 5)
    
    results = {
        'episodes': [],
        'config': config,
        'model_matrices': model_matrices,
        'metadata': {
            'start_time': datetime.now().isoformat(),
            'grid_size': config.get("grid_size", 4),
            'num_episodes': num_episodes,
            'num_timesteps': num_timesteps
        }
    }
    
    for episode in range(num_episodes):
        episode_data = run_episode(agent, model_matrices, num_timesteps)
        results['episodes'].append(episode_data)
        print(f"Completed episode {episode + 1}/{num_episodes}")
    
    results['metadata']['end_time'] = datetime.now().isoformat()
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "simulation_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"Simulation completed. Results saved to {output_dir}")
    return results

def run_episode(agent: Any, model_matrices: Dict[str, np.ndarray], num_timesteps: int) -> Dict[str, Any]:
    """
    Run a single episode of the PyMDP simulation.
    
    Args:
        agent: PyMDP Agent instance
        model_matrices: Model matrices dictionary
        num_timesteps: Number of timesteps to run
        
    Returns:
        Episode data dictionary
    """
    config = model_matrices['config']
    grid_size = config.get("grid_size", 4)
    start_pos = config.get("start_position", [0, 0])
    
    # Initialize episode
    start_state = start_pos[0] * grid_size + start_pos[1]
    current_state = start_state
    
    episode_data = {
        'states': [current_state],
        'observations': [current_state],  # Direct observation for now
        'actions': [],
        'beliefs': [],
        'free_energy': [],
        'timesteps': num_timesteps
    }
    
    # Reset agent
    agent.reset()
    
    for t in range(num_timesteps):
        # Get observation
        observation = [current_state]  # Direct observation
        
        # Agent inference
        qs = agent.infer_states(observation)
        q_pi = agent.infer_policies()
        
        # Sample action
        action = agent.sample_action()
        
        # Environment transition (simulate using B matrix)
        B = model_matrices['B']
        next_state_probs = B[:, current_state, action[0]]
        current_state = np.random.choice(len(next_state_probs), p=next_state_probs)
        
        # Store data
        episode_data['beliefs'].append(qs[0].copy())
        episode_data['actions'].append(action[0])
        episode_data['states'].append(current_state)
        episode_data['observations'].append(current_state)
        
        # Calculate free energy (simplified)
        free_energy = -np.sum(qs[0] * np.log(qs[0] + 1e-16))
        episode_data['free_energy'].append(free_energy)
    
    return episode_data

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function for standalone execution or pipeline integration."""
    
    print("PyMDP Gridworld POMDP Simulation - Pipeline Integration")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Determine execution mode
    if len(sys.argv) > 1:
        gnn_file_path = Path(sys.argv[1])
        print(f"Using GNN file: {gnn_file_path}")
    else:
        gnn_file_path = None
        print("No GNN file specified, using default configuration")
    
    try:
        # Run simulation
        results = run_pipeline_simulation(gnn_file_path)
        
        print("\nSimulation completed successfully!")
        print(f"Number of episodes: {len(results.get('episodes', []))}")
        print(f"Configuration: {results.get('config', {})}")
        
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 