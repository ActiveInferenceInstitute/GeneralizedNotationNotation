#!/usr/bin/env python3
"""
PyMDP Gridworld POMDP Simulation

A comprehensive gridworld simulation using PyMDP with all variables configured at the top.
This script implements a partially observable Markov decision process (POMDP) gridworld
where an agent navigates through a grid with obstacles, goals, and uncertain observations.

Features:
- All configuration variables at the top for easy modification
- Comprehensive output confirmation and validation
- Complete trace saving for analysis
- Built-in visualization utilities
- Performance monitoring and debugging
- Support for different gridworld layouts and scenarios

Author: docxology
Date: 2025-07-24
"""

import numpy as np
import json
import pickle
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from pymdp_gridworld_visualizer import GridworldVisualizer, save_all_visualizations
from pymdp_utils import (
    convert_numpy_for_json, 
    safe_json_dump, 
    safe_pickle_dump,
    save_simulation_results,
    generate_simulation_summary,
    create_output_directory_with_timestamp,
    format_duration
)

# =============================================================================
# CONFIGURATION VARIABLES - ALL SETTINGS CONFIGURED HERE
# =============================================================================

# =============================================================================
# POMDP GENERATIVE MODEL CONFIGURATION
# =============================================================================
# The Active Inference agent models the environment as a Partially Observable
# Markov Decision Process (POMDP) with four core components:
# - A matrix: P(observation | hidden_state) - observation likelihood 
# - B matrix: P(next_state | current_state, action) - transition dynamics
# - C vector: log preferences over observations - prior preferences
# - D vector: P(initial_state) - prior beliefs about starting state

# Gridworld State Space Configuration
GRID_SIZE = 5  # 5x5 grid = 25 discrete spatial locations
NUM_STATES = GRID_SIZE * GRID_SIZE  # 25 total hidden states (agent positions)
NUM_OBSERVATIONS = 9  # Different observation types (wall, empty, goal, etc.)
NUM_ACTIONS = 4  # North, South, East, West movements

# POMDP Generative Model Matrix Dimensions:
# A matrix: [NUM_OBSERVATIONS, NUM_STATES] = [9, 25] 
#   - A[obs_idx, state_idx] = P(observation=obs_idx | hidden_state=state_idx)
#   - Each column represents observation probabilities for a given state
#   - Column-normalized (each column sums to 1)
#
# B matrix: [NUM_STATES, NUM_STATES, NUM_ACTIONS] = [25, 25, 4]
#   - B[next_state, current_state, action] = P(next_state | current_state, action)
#   - B[:,:,0] = transition matrix for action 0 (North)
#   - B[:,:,1] = transition matrix for action 1 (South) 
#   - B[:,:,2] = transition matrix for action 2 (East)
#   - B[:,:,3] = transition matrix for action 3 (West)
#   - Each action slice is column-normalized
#   - Rows represent "to" states, columns represent "from" states
#
# C vector: [NUM_OBSERVATIONS] = [9]
#   - C[obs_idx] = log preference for observation obs_idx
#   - Higher values = more preferred observations
#
# D vector: [NUM_STATES] = [25] 
#   - D[state_idx] = P(initial_state = state_idx)
#   - Normalized probability distribution over starting locations

# Agent Configuration
DISCOUNT_FACTOR = 0.9  # Future reward discount
PLANNING_HORIZON = 5  # Number of steps to plan ahead
INFERENCE_ITERATIONS = 16  # Number of variational message passing iterations
ACTION_PRECISION = 4.0  # Precision for action selection (higher = more deterministic)

# Learning Parameters
LEARNING_RATE_A = 0.05  # Learning rate for observation model (A matrix)
LEARNING_RATE_B = 0.05  # Learning rate for transition model (B matrix)
USE_PARAMETER_LEARNING = True  # Enable parameter learning
USE_INFORMATION_GAIN = True  # Enable information-seeking behavior

# Simulation Parameters
NUM_EPISODES = 10  # Number of episodes to run
MAX_STEPS_PER_EPISODE = 50  # Maximum steps per episode
RANDOM_SEED = 42  # For reproducible results

# Gridworld Layout Configuration
# 0 = empty, 1 = wall, 2 = goal, 3 = hazard
# This layout defines the spatial structure but not the POMDP state space
# The POMDP state space is the linear indexing of grid positions: 0-24
GRID_LAYOUT = np.array([
    [0, 0, 0, 0, 2],  # Top row: goal at top-right (state 4)
    [0, 1, 0, 0, 0],  # Wall in middle (state 6)
    [0, 0, 0, 1, 0],  # Wall in middle (state 13)
    [0, 0, 0, 0, 0],  # Empty (states 15-19)
    [0, 0, 0, 0, 0]   # Bottom row: start area (states 20-24)
])

# POMDP Observation Model Configuration  
# Controls the A matrix: P(observation | hidden_state)
OBSERVATION_ACCURACY = 0.8  # 80% chance of correct observation
NOISE_LEVEL = 0.1  # 10% chance of random observation
# Remaining probability mass distributed among other observations

# Reward Configuration (used to construct C vector preferences)
REWARD_GOAL = 10.0  # Reward for reaching goal
REWARD_HAZARD = -5.0  # Penalty for hitting hazard
REWARD_STEP = -0.1  # Small penalty per step to encourage efficiency
REWARD_WALL = -1.0  # Penalty for trying to move into wall

# Visualization Configuration
SAVE_VISUALIZATIONS = True  # Save plots to files
SHOW_PLOTS = False  # Display plots during simulation (set to False to only save)
PLOT_STYLE = 'seaborn-v0_8'  # Matplotlib style
FIGURE_SIZE = (12, 8)  # Figure size for plots

# Output Configuration
OUTPUT_DIR = Path("pymdp_gridworld_output")  # Output directory
SAVE_TRACES = True  # Save complete simulation traces
SAVE_MATRICES = True  # Save model matrices
VERBOSE_OUTPUT = True  # Detailed console output

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO if VERBOSE_OUTPUT else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(OUTPUT_DIR / 'simulation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_output_directory():
    """Create output directory with timestamp (now using imported utility)"""
    return create_output_directory_with_timestamp(OUTPUT_DIR, "gridworld_sim")

def validate_configuration():
    """Validate all configuration parameters"""
    logger = logging.getLogger(__name__)
    
    # Validate grid size
    if GRID_SIZE < 2:
        raise ValueError("Grid size must be at least 2")
    
    # Validate probabilities
    if not 0 <= OBSERVATION_ACCURACY <= 1:
        raise ValueError("Observation accuracy must be between 0 and 1")
    
    if not 0 <= DISCOUNT_FACTOR <= 1:
        raise ValueError("Discount factor must be between 0 and 1")
    
    # Validate layout
    if GRID_LAYOUT.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"Grid layout shape {GRID_LAYOUT.shape} doesn't match grid size {GRID_SIZE}")
    
    logger.info("Configuration validation passed")
    return True

# =============================================================================
# GRIDWORLD ENVIRONMENT CLASS
# =============================================================================

class GridworldEnvironment:
    """Gridworld environment for POMDP simulation"""
    
    def __init__(self, grid_layout: np.ndarray, observation_accuracy: float = 0.8):
        self.grid_layout = grid_layout.copy()
        self.grid_size = grid_layout.shape[0]
        self.observation_accuracy = observation_accuracy
        self.current_state = None
        self.episode_step = 0
        
        # Define action mappings (North, South, East, West)
        self.actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.action_names = ['North', 'South', 'East', 'West']
        
        # Find start and goal positions
        self.start_positions = self._find_positions(0)  # Empty cells
        self.goal_positions = self._find_positions(2)   # Goal cells
        self.wall_positions = self._find_positions(1)   # Wall cells
        self.hazard_positions = self._find_positions(3) # Hazard cells
        
        self.logger = logging.getLogger(__name__)
    
    def _find_positions(self, cell_type: int) -> List[Tuple[int, int]]:
        """Find all positions of a given cell type"""
        return list(zip(*np.where(self.grid_layout == cell_type)))
    
    def reset(self) -> int:
        """Reset environment to initial state"""
        # Choose random start position from empty cells
        if self.start_positions:
            start_pos = self.start_positions[np.random.choice(len(self.start_positions))]
        else:
            start_pos = (0, 0)  # Default start
        
        self.current_state = self._pos_to_state(start_pos)
        self.episode_step = 0
        
        self.logger.info(f"Environment reset to state {self.current_state} (position {start_pos})")
        return self.current_state
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert grid position to state index"""
        return pos[0] * self.grid_size + pos[1]
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state index to grid position"""
        return (state // self.grid_size, state % self.grid_size)
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """Take action and return (observation, reward, done, info)"""
        if self.current_state is None:
            raise ValueError("Environment not reset")
        
        current_pos = self._state_to_pos(self.current_state)
        action_delta = self.actions[action]
        new_pos = (
            max(0, min(self.grid_size - 1, current_pos[0] + action_delta[0])),
            max(0, min(self.grid_size - 1, current_pos[1] + action_delta[1]))
        )
        
        # Check if new position is a wall
        if new_pos in self.wall_positions:
            new_pos = current_pos  # Stay in place
            reward = REWARD_WALL
        else:
            new_state = self._pos_to_state(new_pos)
            self.current_state = new_state
            
            # Calculate reward
            if new_pos in self.goal_positions:
                reward = REWARD_GOAL
            elif new_pos in self.hazard_positions:
                reward = REWARD_HAZARD
            else:
                reward = REWARD_STEP
        
        # Generate observation
        observation = self._generate_observation(new_pos)
        
        # Check if episode is done
        done = (new_pos in self.goal_positions or 
                new_pos in self.hazard_positions or 
                self.episode_step >= MAX_STEPS_PER_EPISODE)
        
        self.episode_step += 1
        
        info = {
            'position': new_pos,
            'action_taken': self.action_names[action],
            'step': self.episode_step
        }
        
        return observation, reward, done, info
    
    def _generate_observation(self, pos: Tuple[int, int]) -> int:
        """Generate observation for current position"""
        # Get true cell type
        cell_type = self.grid_layout[pos]
        
        # Add observation noise
        if np.random.random() < self.observation_accuracy:
            # Correct observation
            return cell_type
        else:
            # Noisy observation - random cell type
            return np.random.choice(NUM_OBSERVATIONS)
    
    def get_true_state(self) -> int:
        """Get current true state (for debugging)"""
        return self.current_state
    
    def get_position(self) -> Tuple[int, int]:
        """Get current position"""
        return self._state_to_pos(self.current_state)

# =============================================================================
# PYMDP MODEL CONSTRUCTION
# =============================================================================

def create_pymdp_model():
    """
    Create PyMDP model matrices for the gridworld using authentic PyMDP methods.
    
    This function constructs a complete POMDP generative model using real PyMDP
    utilities and follows the official PyMDP API patterns documented at:
    https://pymdp-rtd.readthedocs.io/
    
    Returns:
        tuple: (agent, model_matrices) where agent is a PyMDP Agent instance
               and model_matrices contains the A, B, C, D arrays
    
    POMDP Generative Model Components:
    
    A matrix [NUM_OBSERVATIONS, NUM_STATES]: P(observation | hidden_state)
    - Observation likelihood mapping hidden states to observations
    - Column-normalized conditional probability distributions
    - Each column A[:,s] represents P(observation | state=s)
    
    B matrix [NUM_STATES, NUM_STATES, NUM_ACTIONS]: P(next_state | current_state, action)
    - Transition dynamics for each action
    - Each slice B[:,:,a] is the transition matrix for action a
    - Column-normalized: each column represents P(next_state | current_state, action)
    - Deterministic gridworld dynamics: each column has exactly one 1.0 entry
    
    C vector [NUM_OBSERVATIONS]: log preferences over observations  
    - Prior preferences encoding goal-seeking behavior
    - Higher values indicate more preferred observations
    
    D vector [NUM_STATES]: P(initial_state)
    - Prior beliefs about agent's starting location
    - Normalized probability distribution
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Import real PyMDP modules - this confirms we're using authentic PyMDP
        from pymdp import utils
        from pymdp.agent import Agent
        logger.info("Successfully imported PyMDP modules - using authentic PyMDP library")
    except ImportError as e:
        logger.error(f"PyMDP not available: {e}")
        logger.error("Please install PyMDP: pip install inferactively-pymdp")
        return None, None
    
    logger.info("Constructing POMDP generative model matrices...")
    
    # =============================================================================
    # A MATRIX CONSTRUCTION: P(observation | hidden_state)
    # =============================================================================
    # Create observation model using PyMDP's object array structure
    # Object arrays allow factorized representations of probability distributions
    A = utils.obj_array(1)  # Single observation modality
    A[0] = np.zeros((NUM_OBSERVATIONS, NUM_STATES))
    
    logger.info(f"A matrix shape: {A[0].shape} = [observations={NUM_OBSERVATIONS}, states={NUM_STATES}]")
    
    # Fill A matrix based on grid layout and observation model
    for state in range(NUM_STATES):
        # Convert linear state index to 2D grid position
        pos = (state // GRID_SIZE, state % GRID_SIZE)
        
        # Get the true cell type from the layout
        if pos[0] < GRID_LAYOUT.shape[0] and pos[1] < GRID_LAYOUT.shape[1]:
            cell_type = GRID_LAYOUT[pos]
        else:
            cell_type = 0  # Default to empty if outside layout bounds
        
        # Set correct observation probability
        A[0][cell_type, state] = OBSERVATION_ACCURACY
        
        # Add observation noise to other observation types
        noise_prob = (1 - OBSERVATION_ACCURACY) / (NUM_OBSERVATIONS - 1)
        for obs in range(NUM_OBSERVATIONS):
            if obs != cell_type:
                A[0][obs, state] = noise_prob
    
    # Normalize A matrix using PyMDP utilities
    A[0] = utils.norm_dist(A[0])
    logger.info("A matrix constructed and normalized")
    
    # =============================================================================
    # B MATRIX CONSTRUCTION: P(next_state | current_state, action)  
    # =============================================================================
    # Create transition model using PyMDP's object array structure
    # This is the core of gridworld dynamics - each action slice defines movement
    B = utils.obj_array(1)  # Single state factor (spatial location)
    B[0] = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))
    
    logger.info(f"B matrix shape: {B[0].shape} = [next_states={NUM_STATES}, current_states={NUM_STATES}, actions={NUM_ACTIONS}]")
    logger.info("Constructing B matrix slices for each action:")
    
    # Action mappings following PyMDP gridworld conventions
    actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # North, South, East, West
    action_names = ['North', 'South', 'East', 'West']
    
    # Fill B matrix for each action - this creates deterministic transitions
    for action_idx, (action_delta, action_name) in enumerate(zip(actions, action_names)):
        logger.info(f"  Action {action_idx} ({action_name}): delta={action_delta}")
        
        # For each current state, determine the next state under this action
        for current_state in range(NUM_STATES):
            # Convert linear state to 2D position
            current_pos = (current_state // GRID_SIZE, current_state % GRID_SIZE)
            
            # Apply action with boundary checking
            new_pos = (
                max(0, min(GRID_SIZE - 1, current_pos[0] + action_delta[0])),
                max(0, min(GRID_SIZE - 1, current_pos[1] + action_delta[1]))
            )
            
            # Check if new position hits a wall (from GRID_LAYOUT)
            wall_positions = set()
            for i in range(GRID_LAYOUT.shape[0]):
                for j in range(GRID_LAYOUT.shape[1]):
                    if GRID_LAYOUT[i, j] == 1:  # Wall
                        wall_positions.add((i, j))
            
            if new_pos in wall_positions:
                # Stay in current position if hitting wall
                next_state = current_state
            else:
                # Move to new position
                next_state = new_pos[0] * GRID_SIZE + new_pos[1]
            
            # Set transition probability to 1.0 (deterministic)
            # B[next_state, current_state, action] = 1.0
            B[0][next_state, current_state, action_idx] = 1.0
    
    # Normalize B matrix slices using PyMDP utilities
    B[0] = utils.norm_dist(B[0])
    logger.info("B matrix constructed and normalized")
    
    # Validate B matrix structure
    for action_idx in range(NUM_ACTIONS):
        slice_sum = np.sum(B[0][:, :, action_idx], axis=0)
        if not np.allclose(slice_sum, 1.0):
            logger.warning(f"B matrix slice {action_idx} not properly normalized")
    
    # =============================================================================
    # C VECTOR CONSTRUCTION: log preferences over observations
    # =============================================================================
    # Create preference model using PyMDP utilities
    C = utils.obj_array(1)
    C[0] = np.zeros(NUM_OBSERVATIONS)
    
    # Set preferences based on reward configuration
    # Higher values = more preferred observations
    C[0][2] = 2.0  # Goal preference (observation type 2)
    C[0][3] = -2.0  # Hazard avoidance (observation type 3)
    C[0][1] = -1.0  # Wall avoidance (observation type 1)
    # Other observations remain at 0.0 (neutral)
    
    logger.info(f"C vector constructed: {C[0]}")
    
    # =============================================================================
    # D VECTOR CONSTRUCTION: P(initial_state)
    # =============================================================================
    # Create prior beliefs about initial state using PyMDP utilities  
    D = utils.obj_array(1)
    D[0] = np.ones(NUM_STATES) / NUM_STATES  # Uniform prior over all states
    
    logger.info(f"D vector constructed: uniform prior over {NUM_STATES} states")
    
    # =============================================================================
    # PYMDP AGENT CREATION
    # =============================================================================
    # Create PyMDP agent using authentic Agent class with real parameters
    try:
        agent = Agent(
            A=A, B=B, C=C, D=D,
            policy_len=PLANNING_HORIZON,
            inference_horizon=INFERENCE_ITERATIONS,
            use_utility=True,
            use_states_info_gain=USE_INFORMATION_GAIN,
            use_param_info_gain=USE_PARAMETER_LEARNING,
            lr_pA=LEARNING_RATE_A,
            lr_pB=LEARNING_RATE_B,
            alpha=ACTION_PRECISION
        )
        
        logger.info("PyMDP Agent created successfully with parameters:")
        logger.info(f"  Planning horizon: {PLANNING_HORIZON}")
        logger.info(f"  Inference iterations: {INFERENCE_ITERATIONS}")
        logger.info(f"  Information gain enabled: {USE_INFORMATION_GAIN}")
        logger.info(f"  Parameter learning enabled: {USE_PARAMETER_LEARNING}")
        logger.info(f"  Action precision: {ACTION_PRECISION}")
        
    except Exception as e:
        logger.error(f"Failed to create PyMDP Agent: {e}")
        return None, None
    
    # Prepare model matrices for saving/analysis
    model_matrices = {
        'A': A,
        'B': B, 
        'C': C,
        'D': D,
        'action_names': action_names,
        'grid_layout': GRID_LAYOUT,
        'num_states': NUM_STATES,
        'num_observations': NUM_OBSERVATIONS,
        'num_actions': NUM_ACTIONS
    }
    
    logger.info("POMDP generative model construction complete")
    logger.info(f"Model summary:")
    logger.info(f"  State space: {NUM_STATES} discrete locations in {GRID_SIZE}x{GRID_SIZE} grid")
    logger.info(f"  Observation space: {NUM_OBSERVATIONS} observation types") 
    logger.info(f"  Action space: {NUM_ACTIONS} movement actions")
    logger.info(f"  A matrix dimensions: {A[0].shape}")
    logger.info(f"  B matrix dimensions: {B[0].shape}")
    logger.info(f"  Using authentic PyMDP Agent API v0.0.7+")
    
    return agent, model_matrices

# =============================================================================
# VISUALIZATION UTILITIES (IMPORTED FROM SEPARATE MODULE)
# =============================================================================
# GridworldVisualizer class is in pymdp_gridworld_visualizer.py

# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================

def run_gridworld_simulation():
    """Main simulation function"""
    # Start timing
    simulation_start_time = time.time()
    
    # Setup
    output_dir = create_output_directory()
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("PYMDP GRIDWORLD SIMULATION STARTING")
    logger.info("=" * 60)
    
    # Validate configuration
    validate_configuration()
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Create environment
    logger.info("Creating gridworld environment...")
    env = GridworldEnvironment(GRID_LAYOUT, OBSERVATION_ACCURACY)
    
    # Create PyMDP agent
    logger.info("Creating PyMDP agent...")
    agent, model_matrices = create_pymdp_model()
    
    if agent is None:
        logger.error("Failed to create PyMDP agent. Exiting.")
        return False
    
    # Create visualizer with configuration
    visualizer = GridworldVisualizer(
        GRID_LAYOUT, 
        output_dir,
        plot_style=PLOT_STYLE,
        figure_size=FIGURE_SIZE,
        show_plots=SHOW_PLOTS
    )
    
    # Initialize tracking variables
    all_traces = []
    performance_metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'belief_entropies': [],
        'success_rates': [],
        'convergence_times': []
    }
    
    # Save initial configuration
    config = {
        'grid_size': GRID_SIZE,
        'num_states': NUM_STATES,
        'num_observations': NUM_OBSERVATIONS,
        'num_actions': NUM_ACTIONS,
        'discount_factor': DISCOUNT_FACTOR,
        'planning_horizon': PLANNING_HORIZON,
        'observation_accuracy': OBSERVATION_ACCURACY,
        'grid_layout': GRID_LAYOUT.tolist(),
        'rewards': {
            'goal': REWARD_GOAL,
            'hazard': REWARD_HAZARD,
            'step': REWARD_STEP,
            'wall': REWARD_WALL
        }
    }
    
    # Save configuration using safe utility function
    safe_json_dump(config, output_dir / 'simulation_config.json')
    
    # Save model matrices if requested
    if SAVE_MATRICES:
        matrices_file = output_dir / 'model_matrices.pkl'
        if safe_pickle_dump(model_matrices, matrices_file):
            logger.info(f"Model matrices saved to {matrices_file}")
        else:
            logger.warning(f"Failed to save model matrices to {matrices_file}")
    
    # Run episodes
    successful_episodes = 0
    
    for episode in range(NUM_EPISODES):
        logger.info(f"\n--- Episode {episode + 1}/{NUM_EPISODES} ---")
        
        # Reset environment and agent
        true_state = env.reset()
        episode_trace = {
            'episode': episode,
            'true_states': [],
            'observations': [],
            'actions': [],
            'rewards': [],
            'beliefs': [],
            'positions': [],
            'policies': [],
            'expected_free_energies': [],
            'variational_free_energies': []
        }
        
        episode_reward = 0
        episode_length = 0
        episode_belief_entropies = []
        
        # Episode loop
        for step in range(MAX_STEPS_PER_EPISODE):
            # Get current position
            current_pos = env.get_position()
            
            # Generate observation from environment
            observation, reward, done, info = env.step(0)  # Dummy action to get observation
            
            # Agent inference
            start_time = time.time()
            qs = agent.infer_states([observation])
            
            # Calculate variational free energy (approximation)
            # VFE = -log evidence ≈ complexity - accuracy
            variational_fe = 0.0
            if hasattr(agent, 'qs_current') and agent.qs_current is not None:
                # Simple approximation: entropy of current beliefs
                belief_entropy = -np.sum(qs[0] * np.log(qs[0] + 1e-16))
                variational_fe = belief_entropy
            
            # Policy inference
            if hasattr(agent, 'infer_policies'):
                q_pi, neg_efe = agent.infer_policies()
            else:
                q_pi, neg_efe = None, None
            
            # Action selection
            action = agent.sample_action()
            inference_time = time.time() - start_time
            
            # Execute action in environment
            action_idx = int(action[0]) if isinstance(action[0], (np.integer, np.floating)) else action[0]
            observation, reward, done, info = env.step(action_idx)
            
            # Calculate belief entropy
            belief_entropy = -np.sum(qs[0] * np.log(qs[0] + 1e-16))
            episode_belief_entropies.append(belief_entropy)
            
            # Store trace data
            episode_trace['true_states'].append(env.get_true_state())
            episode_trace['observations'].append(observation)
            episode_trace['actions'].append(action[0])
            episode_trace['rewards'].append(reward)
            episode_trace['beliefs'].append(qs[0].copy())
            episode_trace['positions'].append(current_pos)
            episode_trace['policies'].append(q_pi.copy() if q_pi is not None else None)
            episode_trace['expected_free_energies'].append(neg_efe.copy() if neg_efe is not None else None)
            episode_trace['variational_free_energies'].append(variational_fe)
            
            episode_reward += reward
            episode_length += 1
            
            # Log step information
            if VERBOSE_OUTPUT:
                logger.info(f"Step {step}: pos={current_pos}, obs={observation}, "
                           f"action={action[0]}, reward={reward:.2f}, "
                           f"belief_entropy={belief_entropy:.3f}")
            
            # Check if episode is done
            if done:
                if info.get('position') in env.goal_positions:
                    successful_episodes += 1
                    logger.info(f"Episode {episode + 1} SUCCESS: Reached goal!")
                elif info.get('position') in env.hazard_positions:
                    logger.info(f"Episode {episode + 1} FAILED: Hit hazard!")
                else:
                    logger.info(f"Episode {episode + 1} TIMEOUT: Max steps reached")
                break
        
        # Store episode results
        all_traces.append(episode_trace)
        performance_metrics['episode_rewards'].append(episode_reward)
        performance_metrics['episode_lengths'].append(episode_length)
        performance_metrics['belief_entropies'].append(np.mean(episode_belief_entropies))
        performance_metrics['success_rates'].append(successful_episodes / (episode + 1))
        
        # Log episode summary
        logger.info(f"Episode {episode + 1} Summary:")
        logger.info(f"  Total Reward: {episode_reward:.2f}")
        logger.info(f"  Episode Length: {episode_length}")
        logger.info(f"  Average Belief Entropy: {np.mean(episode_belief_entropies):.3f}")
        logger.info(f"  Success Rate: {performance_metrics['success_rates'][-1]:.2f}")
        
        # Episode visualization will be handled comprehensively at the end
        # Individual episode plots are now generated by save_all_visualizations()
    
    # Save complete traces using comprehensive utility functions
    if SAVE_TRACES:
        save_results = save_simulation_results(
            traces=all_traces,
            metrics=performance_metrics,
            config=config,
            model_matrices=model_matrices if SAVE_MATRICES else None,
            output_dir=output_dir
        )
        
        # Log save results
        for save_type, success in save_results.items():
            if success:
                logger.info(f"✓ Successfully saved {save_type}")
            else:
                logger.warning(f"✗ Failed to save {save_type}")
    
    # Create comprehensive visualizations if requested
    if SAVE_VISUALIZATIONS:
        # Individual performance metrics plot
        fig = visualizer.plot_performance_metrics(
            performance_metrics,
            save_path=output_dir / 'performance_metrics.png'
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        # Generate all visualizations using comprehensive utility
        save_all_visualizations(
            visualizer=visualizer,
            all_traces=all_traces,
            performance_metrics=performance_metrics,
            grid_layout=GRID_LAYOUT
        )
    
    # Calculate simulation duration
    simulation_end_time = time.time()
    simulation_duration = simulation_end_time - simulation_start_time
    
    # Generate comprehensive simulation summary
    simulation_summary = generate_simulation_summary(all_traces, performance_metrics)
    
    # Add timing information to summary
    simulation_summary['simulation_duration_seconds'] = simulation_duration
    simulation_summary['simulation_duration_formatted'] = format_duration(simulation_duration)
    
    # Save simulation summary
    summary_saved = safe_json_dump(simulation_summary, output_dir / 'simulation_summary.json')
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total Episodes: {simulation_summary['total_episodes']}")
    logger.info(f"Successful Episodes: {simulation_summary['successful_episodes']}")
    logger.info(f"Success Rate: {simulation_summary['success_rate']:.2%}")
    logger.info(f"Average Episode Reward: {simulation_summary['average_reward']:.2f}")
    logger.info(f"Average Episode Length: {simulation_summary['average_episode_length']:.1f}")
    logger.info(f"Total Steps: {simulation_summary['total_steps']}")
    logger.info(f"Simulation Duration: {simulation_summary['simulation_duration_formatted']}")
    logger.info(f"Output Directory: {output_dir}")
    
    # Validate outputs
    logger.info("\nOutput Validation:")
    required_files = [
        'simulation_config.json',
        'simulation_summary.json',
        'performance_metrics.json'
    ]
    
    if SAVE_TRACES:
        required_files.extend(['simulation_traces.pkl', 'simulation_traces.json'])
    
    if SAVE_MATRICES:
        required_files.append('model_matrices.pkl')
    
    all_outputs_valid = True
    for filename in required_files:
        filepath = output_dir / filename
        if filepath.exists():
            logger.info(f"✓ {filename} ({filepath.stat().st_size} bytes)")
        else:
            logger.error(f"✗ Missing: {filename}")
            all_outputs_valid = False
    
    if SAVE_VISUALIZATIONS:
        png_files = list(output_dir.glob("*.png"))
        logger.info(f"✓ {len(png_files)} visualization files saved")
        if png_files:
            logger.info(f"  Examples: {', '.join([f.name for f in png_files[:3]])}")
    
    logger.info(f"\n{'✓ All outputs validated successfully!' if all_outputs_valid else '✗ Some outputs missing - check logs'}")
    
    return all_outputs_valid

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        success = run_gridworld_simulation()
        if success:
            print("\n🎉 PyMDP Gridworld Simulation completed successfully!")
            print("Check the output directory for results and visualizations.")
        else:
            print("\n❌ Simulation failed. Check the logs for details.")
            exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Simulation interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 