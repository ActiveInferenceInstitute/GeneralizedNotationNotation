#!/usr/bin/env python3
"""
PyMDP Simulation Class

A configurable PyMDP simulation that can be parameterized by GNN specifications.
This class integrates with the GNN processing pipeline to run Active Inference
simulations based on parsed GNN POMDP models.

Features:
- GNN-driven configuration from parsed state spaces
- Authentic PyMDP integration with real Agent class
- Comprehensive output and visualization
- Pipeline-compatible structure
- Configurable from actinf_pomdp_agent.md specifications

Author: GNN PyMDP Integration
Date: 2024
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

from .pymdp_visualizer import PyMDPVisualizer
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

# Import real PyMDP components - will be available when executed
try:
    from pymdp import utils
    from pymdp.agent import Agent
    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False
    logging.warning("PyMDP not available - simulation will use mock mode")


class PyMDPSimulation:
    """
    PyMDP Active Inference simulation configured from GNN specifications.
    
    This class creates and runs authentic PyMDP simulations using parameters
    extracted from GNN POMDP specifications. It supports both GNN-configured
    and default parameter modes.
    """

    def __init__(self, gnn_config: Optional[Dict[str, Any]] = None):
        """
        Initialize PyMDP simulation with optional GNN configuration.
        
        Args:
            gnn_config: Dictionary containing parsed GNN POMDP parameters
                       Expected structure:
                       {
                           'states': List[str],  # e.g., ['location_0', 'location_1', ...]
                           'actions': List[str], # e.g., ['move_up', 'move_down', ...]
                           'observations': List[str], # e.g., ['obs_location_0', ...]
                           'model_name': str,
                           'parameters': Dict[str, Any]
                       }
        """
        self.gnn_config = gnn_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize core parameters from GNN or defaults
        self._initialize_parameters()
        
        # Initialize PyMDP components
        self.agent = None
        self.model_matrices = {}
        self.simulation_trace = []
        self.results = {}
        
        # Initialize visualizer
        self.visualizer = PyMDPVisualizer()

    def _initialize_parameters(self):
        """Initialize simulation parameters from GNN config or defaults."""
        if self.gnn_config:
            # Extract from GNN configuration
            self.states = self.gnn_config.get('states', [])
            self.actions = self.gnn_config.get('actions', [])  
            self.observations = self.gnn_config.get('observations', [])
            self.model_name = self.gnn_config.get('model_name', 'GNN_POMDP')
            
            # Convert to numerical parameters
            self.num_states = len(self.states)
            self.num_actions = len(self.actions)
            self.num_observations = len(self.observations)
            
            self.logger.info(f"Initialized from GNN config: {self.num_states} states, "
                           f"{self.num_actions} actions, {self.num_observations} observations")
        else:
            # Default gridworld configuration
            self.num_states = 4
            self.num_actions = 5  
            self.num_observations = 4
            self.states = [f"location_{i}" for i in range(self.num_states)]
            self.actions = ["move_up", "move_down", "move_left", "move_right", "stay"]
            self.observations = [f"obs_location_{i}" for i in range(self.num_observations)]
            self.model_name = "Default_Gridworld"
            
            self.logger.info("Initialized with default gridworld configuration")

        # Simulation parameters (can be overridden by GNN config)
        gnn_params = self.gnn_config.get('parameters', {})
        self.num_timesteps = gnn_params.get('num_timesteps', 20)
        self.learning_rate = gnn_params.get('learning_rate', 0.5)
        self.alpha = gnn_params.get('alpha', 16.0)  # Precision parameter
        self.gamma = gnn_params.get('gamma', 16.0)  # Precision parameter

    def create_pymdp_model_from_gnn(self):
        """
        Create PyMDP model matrices using extracted GNN matrices from InitialParameterization.
        
        This method specifically uses the A, B, C, D, E matrices extracted from the GNN file's
        InitialParameterization section for authentic simulation.
        
        Returns:
            tuple: (agent, model_matrices) with PyMDP Agent and matrix dict
        """
        if not PYMDP_AVAILABLE:
            self.logger.error("PyMDP not available - cannot create model")
            return None, {}

        try:
            # Check if we have extracted GNN matrices
            if not hasattr(self, 'gnn_matrices') or not self.gnn_matrices:
                self.logger.warning("No GNN matrices available, using default creation method")
                return self.create_pymdp_model()
            
            self.logger.info(f"Creating PyMDP model from GNN matrices: {list(self.gnn_matrices.keys())}")
            
            # A matrix: P(observation | hidden_state) [NUM_OBS, NUM_STATES]
            A = utils.obj_array(1)
            if 'A' in self.gnn_matrices:
                A[0] = self._process_gnn_A_matrix(self.gnn_matrices['A'])
                self.logger.info(f"Using GNN A matrix with shape: {A[0].shape}")
            else:
                A[0] = self._create_observation_model()
                self.logger.info("Using default A matrix")
            
            # B matrix: P(next_state | current_state, action) [NUM_STATES, NUM_STATES, NUM_ACTIONS]  
            B = utils.obj_array(1)
            if 'B' in self.gnn_matrices:
                B[0] = self._process_gnn_B_matrix(self.gnn_matrices['B'])
                self.logger.info(f"Using GNN B matrix with shape: {B[0].shape}")
            else:
                B[0] = self._create_transition_model()
                self.logger.info("Using default B matrix")
            
            # C vector: log preferences over observations [NUM_OBS]
            C = utils.obj_array(1)
            if 'C' in self.gnn_matrices:
                C[0] = self._process_gnn_C_vector(self.gnn_matrices['C'])
                self.logger.info(f"Using GNN C vector with shape: {C[0].shape}")
            else:
                C[0] = self._create_preference_model()
                self.logger.info("Using default C vector")
            
            # D vector: prior beliefs over initial states [NUM_STATES]
            D = utils.obj_array(1)
            if 'D' in self.gnn_matrices:
                D[0] = self._process_gnn_D_vector(self.gnn_matrices['D'])
                self.logger.info(f"Using GNN D vector with shape: {D[0].shape}")
            else:
                D[0] = self._create_prior_beliefs()
                self.logger.info("Using default D vector")

            # Store matrices for analysis
            self.model_matrices = {
                'A': A[0],
                'B': B[0], 
                'C': C[0],
                'D': D[0]
            }
            
            # Log matrix properties
            self.logger.info(f"Final model matrices:")
            self.logger.info(f"  A (likelihood): {A[0].shape}, sum={A[0].sum():.3f}")
            self.logger.info(f"  B (transition): {B[0].shape}, sum={B[0].sum():.3f}")
            self.logger.info(f"  C (preferences): {C[0].shape}, min={C[0].min():.3f}, max={C[0].max():.3f}")
            self.logger.info(f"  D (prior): {D[0].shape}, sum={D[0].sum():.3f}")

            # Create PyMDP Agent with extracted parameters
            self.agent = Agent(
                A=A, B=B, C=C, D=D,
                lr_pB=self.learning_rate,
                policy_len=3,  # Plan 3 steps ahead
                control_fac_idx=[0],  # Control factor 0 (states)
                policies=self._generate_policies()
            )

            self.logger.info(f"Created PyMDP model from GNN matrices: {self.num_states}S, {self.num_actions}A, {self.num_observations}O")
            return self.agent, self.model_matrices

        except Exception as e:
            self.logger.error(f"Error creating PyMDP model from GNN matrices: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to default model creation
            self.logger.info("Falling back to default model creation")
            return self.create_pymdp_model()

    def _process_gnn_A_matrix(self, gnn_A) -> np.ndarray:
        """Process GNN A matrix into PyMDP format."""
        try:
            # Convert GNN matrix to numpy array
            if isinstance(gnn_A, list):
                A_matrix = np.array(gnn_A)
            elif isinstance(gnn_A, np.ndarray):
                A_matrix = gnn_A
            else:
                self.logger.warning(f"Unexpected A matrix type: {type(gnn_A)}")
                return self._create_observation_model()
            
            # Ensure correct shape [observations, states]
            if A_matrix.shape != (self.num_observations, self.num_states):
                self.logger.warning(f"A matrix shape mismatch: got {A_matrix.shape}, expected ({self.num_observations}, {self.num_states})")
                # Try to reshape or use default
                if A_matrix.size == self.num_observations * self.num_states:
                    A_matrix = A_matrix.reshape(self.num_observations, self.num_states)
                else:
                    return self._create_observation_model()
            
            # Normalize probabilities
            A_matrix = utils.norm_dist(A_matrix)
            
            self.logger.info(f"Processed GNN A matrix: shape={A_matrix.shape}, min={A_matrix.min():.3f}, max={A_matrix.max():.3f}")
            return A_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to process GNN A matrix: {e}")
            return self._create_observation_model()

    def _process_gnn_B_matrix(self, gnn_B) -> np.ndarray:
        """Process GNN B matrix into PyMDP format."""
        try:
            # Convert GNN matrix to numpy array
            if isinstance(gnn_B, list):
                B_matrix = np.array(gnn_B)
            elif isinstance(gnn_B, np.ndarray):
                B_matrix = gnn_B
            else:
                self.logger.warning(f"Unexpected B matrix type: {type(gnn_B)}")
                return self._create_transition_model()
            
            # Ensure correct shape [next_states, current_states, actions]
            expected_shape = (self.num_states, self.num_states, self.num_actions)
            if B_matrix.shape != expected_shape:
                self.logger.warning(f"B matrix shape mismatch: got {B_matrix.shape}, expected {expected_shape}")
                # Try to handle different organizations
                if B_matrix.ndim == 3 and B_matrix.shape[2] == self.num_actions:
                    # Assume [states, states, actions] format
                    if B_matrix.shape[:2] == (self.num_states, self.num_states):
                        pass  # Already correct
                    else:
                        return self._create_transition_model()
                else:
                    return self._create_transition_model()
            
            # Normalize transition probabilities for each action
            for action in range(self.num_actions):
                B_matrix[:, :, action] = utils.norm_dist(B_matrix[:, :, action])
            
            self.logger.info(f"Processed GNN B matrix: shape={B_matrix.shape}, min={B_matrix.min():.3f}, max={B_matrix.max():.3f}")
            return B_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to process GNN B matrix: {e}")
            return self._create_transition_model()

    def _process_gnn_C_vector(self, gnn_C) -> np.ndarray:
        """Process GNN C vector into PyMDP format."""
        try:
            # Convert GNN vector to numpy array
            if isinstance(gnn_C, list):
                C_vector = np.array(gnn_C)
            elif isinstance(gnn_C, np.ndarray):
                C_vector = gnn_C
            else:
                self.logger.warning(f"Unexpected C vector type: {type(gnn_C)}")
                return self._create_preference_model()
            
            # Handle different possible shapes
            if C_vector.ndim > 1:
                C_vector = C_vector.flatten()
            
            # Ensure correct length
            if len(C_vector) != self.num_observations:
                self.logger.warning(f"C vector length mismatch: got {len(C_vector)}, expected {self.num_observations}")
                if len(C_vector) > self.num_observations:
                    C_vector = C_vector[:self.num_observations]
                else:
                    # Pad with zeros
                    padded_C = np.zeros(self.num_observations)
                    padded_C[:len(C_vector)] = C_vector
                    C_vector = padded_C
            
            self.logger.info(f"Processed GNN C vector: shape={C_vector.shape}, min={C_vector.min():.3f}, max={C_vector.max():.3f}")
            return C_vector
            
        except Exception as e:
            self.logger.error(f"Failed to process GNN C vector: {e}")
            return self._create_preference_model()

    def _process_gnn_D_vector(self, gnn_D) -> np.ndarray:
        """Process GNN D vector into PyMDP format."""
        try:
            # Convert GNN vector to numpy array
            if isinstance(gnn_D, list):
                D_vector = np.array(gnn_D)
            elif isinstance(gnn_D, np.ndarray):
                D_vector = gnn_D
            else:
                self.logger.warning(f"Unexpected D vector type: {type(gnn_D)}")
                return self._create_prior_beliefs()
            
            # Handle different possible shapes
            if D_vector.ndim > 1:
                D_vector = D_vector.flatten()
            
            # Ensure correct length
            if len(D_vector) != self.num_states:
                self.logger.warning(f"D vector length mismatch: got {len(D_vector)}, expected {self.num_states}")
                if len(D_vector) > self.num_states:
                    D_vector = D_vector[:self.num_states]
                else:
                    # Pad with uniform distribution
                    padded_D = np.ones(self.num_states) / self.num_states
                    padded_D[:len(D_vector)] = D_vector[:len(D_vector)]
                    D_vector = padded_D
            
            # Normalize to ensure it's a proper probability distribution
            D_vector = utils.norm_dist(D_vector)
            
            self.logger.info(f"Processed GNN D vector: shape={D_vector.shape}, sum={D_vector.sum():.3f}")
            return D_vector
            
        except Exception as e:
            self.logger.error(f"Failed to process GNN D vector: {e}")
            return self._create_prior_beliefs()

    def create_pymdp_model(self):
        """
        Create PyMDP model matrices using authentic PyMDP methods.
        
        This function constructs the complete POMDP generative model (A, B, C, D)
        using real PyMDP utilities, configured from GNN specifications.
        
        Returns:
            tuple: (agent, model_matrices) with PyMDP Agent and matrix dict
        """
        if not PYMDP_AVAILABLE:
            self.logger.error("PyMDP not available - cannot create model")
            return None, {}

        try:
            # A matrix: P(observation | hidden_state) [NUM_OBS, NUM_STATES]
            A = utils.obj_array(1)
            A[0] = self._create_observation_model()
            
            # B matrix: P(next_state | current_state, action) [NUM_STATES, NUM_STATES, NUM_ACTIONS]  
            B = utils.obj_array(1)
            B[0] = self._create_transition_model()
            
            # C vector: log preferences over observations [NUM_OBS]
            C = utils.obj_array(1)
            C[0] = self._create_preference_model()
            
            # D vector: prior beliefs over initial states [NUM_STATES]
            D = utils.obj_array(1) 
            D[0] = self._create_prior_beliefs()

            # Store matrices for analysis
            self.model_matrices = {
                'A': A[0],
                'B': B[0], 
                'C': C[0],
                'D': D[0]
            }

            # Create PyMDP Agent with authentic parameters
            self.agent = Agent(
                A=A, B=B, C=C, D=D,
                lr_pB=self.learning_rate,
                policy_len=3,  # Plan 3 steps ahead
                control_fac_idx=[0],  # Control factor 0 (states)
                policies=self._generate_policies()
            )

            self.logger.info(f"Created PyMDP model: {self.num_states}S, {self.num_actions}A, {self.num_observations}O")
            return self.agent, self.model_matrices

        except Exception as e:
            self.logger.error(f"Error creating PyMDP model: {e}")
            return None, {}

    def _create_observation_model(self) -> np.ndarray:
        """Create observation likelihood matrix A."""
        # Create noisy identity mapping with configurable noise
        A_matrix = np.eye(self.num_observations, self.num_states) * 0.9
        
        # Add observation noise
        noise_level = 0.1 / (self.num_observations - 1)
        for i in range(self.num_observations):
            for j in range(self.num_states):
                if i != j:
                    A_matrix[i, j] = noise_level
                    
        return utils.norm_dist(A_matrix)

    def _create_transition_model(self) -> np.ndarray:
        """Create transition dynamics matrix B."""
        # B matrix: [next_state, current_state, action]
        B_matrix = np.zeros((self.num_states, self.num_states, self.num_actions))
        
        # Configure based on GNN state space or default gridworld
        if self.gnn_config and 'transition_structure' in self.gnn_config:
            # Use GNN-specified transitions
            transitions = self.gnn_config['transition_structure']
            for action_idx, action_name in enumerate(self.actions):
                if action_name in transitions:
                    for from_state, to_states in transitions[action_name].items():
                        from_idx = self.states.index(from_state)
                        for to_state, prob in to_states.items():
                            to_idx = self.states.index(to_state)
                            B_matrix[to_idx, from_idx, action_idx] = prob
        else:
            # Default gridworld transitions (2x2 grid)
            for action_idx, action_name in enumerate(self.actions):
                for state in range(self.num_states):
                    if action_name == "stay":
                        B_matrix[state, state, action_idx] = 1.0
                    elif action_name == "move_up":
                        next_state = max(0, state - 2) if state >= 2 else state
                        B_matrix[next_state, state, action_idx] = 1.0
                    elif action_name == "move_down": 
                        next_state = min(3, state + 2) if state < 2 else state
                        B_matrix[next_state, state, action_idx] = 1.0
                    elif action_name == "move_left":
                        next_state = state - 1 if state % 2 == 1 else state
                        B_matrix[next_state, state, action_idx] = 1.0
                    elif action_name == "move_right":
                        next_state = state + 1 if state % 2 == 0 else state
                        B_matrix[next_state, state, action_idx] = 1.0
                        
        # Normalize transition probabilities
        for action in range(self.num_actions):
            B_matrix[:, :, action] = utils.norm_dist(B_matrix[:, :, action])
            
        return B_matrix

    def _create_preference_model(self) -> np.ndarray:
        """Create preference vector C."""
        # Default preferences or GNN-specified
        gnn_params = self.gnn_config.get('parameters', {})
        if 'preferences' in gnn_params:
            preferences = np.array(gnn_params['preferences'])
        else:
            # Default: prefer last location (goal state)
            preferences = np.zeros(self.num_observations)
            preferences[-1] = 2.0  # Strong preference for goal
            
        return preferences

    def _create_prior_beliefs(self) -> np.ndarray:
        """Create prior belief vector D."""
        # Uniform prior or GNN-specified
        gnn_params = self.gnn_config.get('parameters', {})
        if 'prior_beliefs' in gnn_params:
            prior = np.array(gnn_params['prior_beliefs'])
        else:
            # Uniform prior with slight preference for first location
            prior = np.ones(self.num_states) / self.num_states
            prior[0] = 0.4  # Slightly higher prior for starting location
            
        return utils.norm_dist(prior)

    def _generate_policies(self) -> List[List[int]]:
        """Generate policy space for planning."""
        # Simple policy generation - can be enhanced based on GNN specs
        policies = []
        for a1 in range(self.num_actions):
            for a2 in range(self.num_actions):
                for a3 in range(self.num_actions):
                    policies.append([a1, a2, a3])
        return policies

    def run_simulation(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run the complete PyMDP simulation.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing simulation results and metrics
        """
        if not self.agent:
            self.logger.error("No agent available - create model first")
            return {}

        start_time = time.time()
        self.logger.info(f"Starting PyMDP simulation: {self.model_name}")
        
        # Initialize simulation state
        current_state = 0  # Start at first location
        self.simulation_trace = []
        
        try:
            for t in range(self.num_timesteps):
                # Generate observation from current state
                obs_probs = self.model_matrices['A'][:, current_state]
                observation = utils.sample(obs_probs)
                
                # Agent inference
                qs = self.agent.infer_states(observation)
                q_pi, G = self.agent.infer_policies()
                
                # Sample action
                action = self.agent.sample_action()
                
                # Update environment state
                next_state_probs = self.model_matrices['B'][:, current_state, action]
                next_state = utils.sample(next_state_probs)
                
                # Record step
                step_data = {
                    'timestep': t,
                    'current_state': int(current_state),
                    'observation': int(observation),
                    'action': int(action),
                    'next_state': int(next_state),
                    'beliefs': qs[0].copy(),
                    'policy_probs': q_pi.copy(),
                    'expected_free_energy': G.copy()
                }
                self.simulation_trace.append(step_data)
                
                # Update state
                current_state = next_state
                
                if t % 5 == 0:
                    self.logger.info(f"Timestep {t}: state={current_state}, obs={observation}, action={action}")

        except Exception as e:
            self.logger.error(f"Simulation error at timestep {t}: {e}")
            return {}

        # Calculate results
        duration = time.time() - start_time
        self.results = self._analyze_results(duration)
        
        # Save results if output directory provided
        if output_dir:
            self._save_results(output_dir)
        
        self.logger.info(f"Simulation completed in {format_duration(duration)}")
        return self.results

    def _analyze_results(self, duration: float) -> Dict[str, Any]:
        """Analyze simulation results and compute metrics."""
        if not self.simulation_trace:
            return {}

        # Basic metrics
        total_timesteps = len(self.simulation_trace)
        final_state = self.simulation_trace[-1]['next_state']
        states_visited = set(step['current_state'] for step in self.simulation_trace)
        
        # Belief dynamics analysis
        belief_entropies = []
        action_counts = np.zeros(self.num_actions)
        
        for step in self.simulation_trace:
            # Entropy of beliefs
            beliefs = step['beliefs']
            entropy = -np.sum(beliefs * np.log(beliefs + 1e-16))
            belief_entropies.append(entropy)
            
            # Action statistics
            action_counts[step['action']] += 1

        # Compute summary metrics
        results = {
            'model_name': self.model_name,
            'total_timesteps': total_timesteps,
            'duration_seconds': duration,
            'final_state': final_state,
            'states_visited': len(states_visited),
            'unique_states_ratio': len(states_visited) / self.num_states,
            'mean_belief_entropy': np.mean(belief_entropies),
            'action_distribution': (action_counts / total_timesteps).tolist(),
            'most_used_action': int(np.argmax(action_counts)),
            'exploration_efficiency': len(states_visited) / total_timesteps,
            'gnn_config_used': bool(self.gnn_config),
            'configuration': {
                'num_states': self.num_states,
                'num_actions': self.num_actions, 
                'num_observations': self.num_observations,
                'learning_rate': self.learning_rate,
                'alpha': self.alpha,
                'gamma': self.gamma
            }
        }
        
        return results

    def _save_results(self, output_dir: Path):
        """Save comprehensive simulation results."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            results_file = output_dir / f"pymdp_results_{self.model_name}.json"
            safe_json_dump(self.results, results_file)
            
            # Save detailed trace
            trace_file = output_dir / f"pymdp_trace_{self.model_name}.json" 
            cleaned_trace = clean_trace_for_serialization(self.simulation_trace)
            safe_json_dump(cleaned_trace, trace_file)
            
            # Save model matrices
            matrices_file = output_dir / f"pymdp_matrices_{self.model_name}.pkl"
            safe_pickle_dump(self.model_matrices, matrices_file)
            
            # Generate visualizations
            self.visualizer.save_all_visualizations(
                self.simulation_trace, 
                self.model_matrices,
                output_dir,
                prefix=f"{self.model_name}_"
            )
            
            self.logger.info(f"Results saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get concise simulation summary."""
        if not self.results:
            return {"status": "no_results"}
            
        return {
            "model_name": self.results.get('model_name', 'Unknown'),
            "timesteps": self.results.get('total_timesteps', 0),
            "duration": self.results.get('duration_seconds', 0),
            "final_state": self.results.get('final_state', -1),
            "states_explored": self.results.get('states_visited', 0),
            "gnn_configured": self.results.get('gnn_config_used', False),
            "success": True
        }


def create_pymdp_simulation_from_gnn(gnn_config: Dict[str, Any]) -> PyMDPSimulation:
    """
    Factory function to create PyMDP simulation from GNN configuration.
    
    Args:
        gnn_config: Parsed GNN POMDP specification
        
    Returns:
        Configured PyMDPSimulation instance
    """
    return PyMDPSimulation(gnn_config=gnn_config)


def run_pymdp_simulation_from_gnn(gnn_config: Dict[str, Any], 
                                  output_dir: Path) -> Dict[str, Any]:
    """
    Complete function to run PyMDP simulation from GNN config.
    
    Args:
        gnn_config: Parsed GNN POMDP specification
        output_dir: Directory to save results
        
    Returns:
        Simulation results dictionary
    """
    simulation = create_pymdp_simulation_from_gnn(gnn_config)
    simulation.create_pymdp_model()
    return simulation.run_simulation(output_dir) 