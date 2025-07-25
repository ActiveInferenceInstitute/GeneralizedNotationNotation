#!/usr/bin/env python3
"""
PyMDP Simulation Class

A configurable PyMDP simulation that can be parameterized by GNN specifications.
This class integrates with the GNN processing pipeline to run Active Inference
simulations based on parsed GNN POMDP models.

Features:
- GNN-driven configuration
- Authentic PyMDP integration
- Comprehensive output and visualization
- Pipeline-compatible structure
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
    save_simulation_results,
    generate_simulation_summary,
    create_output_directory_with_timestamp,
    format_duration
)

class PyMDPSimulation:
    """
    Configurable PyMDP simulation class that can be parameterized by GNN specifications.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PyMDP simulation with configuration.
        
        Args:
            config: Configuration dictionary containing simulation parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract core POMDP dimensions from config
        self.num_states = config.get('num_hidden_states', 3)
        self.num_observations = config.get('num_observations', 3) 
        self.num_actions = config.get('num_actions', 3)
        self.model_name = config.get('model_name', 'GNN_POMDP_Agent')
        
        # Simulation parameters
        self.num_episodes = config.get('num_episodes', 10)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 20)
        self.planning_horizon = config.get('planning_horizon', 5)
        self.inference_iterations = config.get('inference_iterations', 16)
        self.action_precision = config.get('action_precision', 4.0)
        
        # Learning parameters
        self.use_parameter_learning = config.get('use_parameter_learning', True)
        self.use_information_gain = config.get('use_information_gain', True)
        self.learning_rate_A = config.get('learning_rate_A', 0.05)
        self.learning_rate_B = config.get('learning_rate_B', 0.05)
        
        # Output configuration
        self.save_traces = config.get('save_traces', True)
        self.save_matrices = config.get('save_matrices', True)
        self.save_visualizations = config.get('save_visualizations', True)
        self.verbose_output = config.get('verbose_output', True)
        self.random_seed = config.get('random_seed', 42)
        
        # Initialize components
        self.agent = None
        self.model_matrices = None
        self.environment = None
        self.visualizer = None
        
        self.logger.info(f"Initialized PyMDP simulation: {self.model_name}")
        self.logger.info(f"  States: {self.num_states}, Observations: {self.num_observations}, Actions: {self.num_actions}")
    
    def setup_logging(self, output_dir: Path):
        """Setup logging for the simulation."""
        log_file = output_dir / 'pymdp_simulation.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO if self.verbose_output else logging.WARNING)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.info("PyMDP simulation logging initialized")
    
    def create_pymdp_model(self):
        """
        Create PyMDP model matrices using authentic PyMDP methods.
        Can be configured from GNN initial parameterization.
        """
        try:
            # Import real PyMDP modules
            from pymdp import utils
            from pymdp.agent import Agent
            self.logger.info("Successfully imported PyMDP modules - using authentic PyMDP library")
        except ImportError as e:
            self.logger.error(f"PyMDP not available: {e}")
            self.logger.error("Please install PyMDP: pip install inferactively-pymdp")
            return None, None
        
        self.logger.info("Constructing POMDP generative model matrices...")
        
        # Create A matrix: P(observation | hidden_state)
        A = utils.obj_array(1)
        A[0] = self._create_A_matrix()
        A[0] = utils.norm_dist(A[0])
        
        # Create B matrix: P(next_state | current_state, action)
        B = utils.obj_array(1)
        B[0] = self._create_B_matrix()
        B[0] = utils.norm_dist(B[0])
        
        # Create C vector: log preferences over observations
        C = utils.obj_array(1)
        C[0] = self._create_C_vector()
        
        # Create D vector: P(initial_state)
        D = utils.obj_array(1)
        D[0] = self._create_D_vector()
        
        self.logger.info(f"Model matrices created:")
        self.logger.info(f"  A matrix: {A[0].shape}")
        self.logger.info(f"  B matrix: {B[0].shape}")
        self.logger.info(f"  C vector: {C[0].shape}")
        self.logger.info(f"  D vector: {D[0].shape}")
        
        # Create PyMDP agent
        try:
            agent = Agent(
                A=A, B=B, C=C, D=D,
                policy_len=self.planning_horizon,
                inference_horizon=self.inference_iterations,
                use_utility=True,
                use_states_info_gain=self.use_information_gain,
                use_param_info_gain=self.use_parameter_learning,
                lr_pA=self.learning_rate_A,
                lr_pB=self.learning_rate_B,
                alpha=self.action_precision
            )
            
            self.logger.info("PyMDP Agent created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create PyMDP Agent: {e}")
            return None, None
        
        # Store model matrices for analysis
        model_matrices = {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'num_states': self.num_states,
            'num_observations': self.num_observations,
            'num_actions': self.num_actions,
            'model_name': self.model_name
        }
        
        return agent, model_matrices
    
    def _create_A_matrix(self) -> np.ndarray:
        """Create observation likelihood matrix."""
        # Check if initial A matrix provided in config
        if 'initial_A_matrix' in self.config and self.config['initial_A_matrix']:
            return self._parse_matrix_from_gnn(self.config['initial_A_matrix'], 
                                             (self.num_observations, self.num_states))
        
        # Default: identity-like mapping with some noise
        A = np.eye(min(self.num_observations, self.num_states), self.num_states)
        if self.num_observations > self.num_states:
            # Pad with small values
            padding = np.ones((self.num_observations - self.num_states, self.num_states)) * 0.1
            A = np.vstack([A, padding])
        
        # Add observation noise
        noise = np.random.uniform(0.01, 0.1, A.shape)
        A = A * 0.9 + noise
        
        return A
    
    def _create_B_matrix(self) -> np.ndarray:
        """Create transition dynamics matrix."""
        # Check if initial B matrix provided in config
        if 'initial_B_matrix' in self.config and self.config['initial_B_matrix']:
            return self._parse_3d_matrix_from_gnn(self.config['initial_B_matrix'], 
                                                (self.num_states, self.num_states, self.num_actions))
        
        # Default: each action causes deterministic state transitions
        B = np.zeros((self.num_states, self.num_states, self.num_actions))
        
        for action in range(self.num_actions):
            for state in range(self.num_states):
                # Simple cyclic transitions for each action
                next_state = (state + action + 1) % self.num_states
                B[next_state, state, action] = 1.0
        
        return B
    
    def _create_C_vector(self) -> np.ndarray:
        """Create preference vector."""
        # Check if initial C vector provided in config
        if 'initial_C_vector' in self.config and self.config['initial_C_vector']:
            return self._parse_vector_from_gnn(self.config['initial_C_vector'], self.num_observations)
        
        # Default: prefer later observation indices (simple goal-seeking)
        C = np.linspace(-1.0, 2.0, self.num_observations)
        return C
    
    def _create_D_vector(self) -> np.ndarray:
        """Create prior state distribution."""
        # Check if initial D vector provided in config
        if 'initial_D_vector' in self.config and self.config['initial_D_vector']:
            return self._parse_vector_from_gnn(self.config['initial_D_vector'], self.num_states)
        
        # Default: uniform prior
        return np.ones(self.num_states) / self.num_states
    
    def _parse_matrix_from_gnn(self, matrix_data: Any, expected_shape: Tuple[int, int]) -> np.ndarray:
        """Parse matrix from GNN initial parameterization."""
        try:
            if isinstance(matrix_data, str):
                # Parse string representation
                matrix_data = eval(matrix_data)  # Simple eval for now - could be made more robust
            
            if isinstance(matrix_data, (list, tuple)):
                matrix = np.array(matrix_data, dtype=float)
                if matrix.shape != expected_shape:
                    self.logger.warning(f"Matrix shape {matrix.shape} doesn't match expected {expected_shape}, reshaping")
                    matrix = matrix.reshape(expected_shape)
                return matrix
            
        except Exception as e:
            self.logger.warning(f"Failed to parse matrix from GNN data: {e}, using default")
        
        # Fallback to default
        return np.random.uniform(0.1, 1.0, expected_shape)
    
    def _parse_3d_matrix_from_gnn(self, matrix_data: Any, expected_shape: Tuple[int, int, int]) -> np.ndarray:
        """Parse 3D matrix (like B matrix) from GNN initial parameterization."""
        try:
            if isinstance(matrix_data, str):
                matrix_data = eval(matrix_data)
            
            if isinstance(matrix_data, (list, tuple)):
                matrix = np.array(matrix_data, dtype=float)
                if matrix.shape != expected_shape:
                    self.logger.warning(f"3D Matrix shape {matrix.shape} doesn't match expected {expected_shape}")
                    # Try to reshape or create default
                    if matrix.size == np.prod(expected_shape):
                        matrix = matrix.reshape(expected_shape)
                    else:
                        raise ValueError("Cannot reshape")
                return matrix
            
        except Exception as e:
            self.logger.warning(f"Failed to parse 3D matrix from GNN data: {e}, using default")
        
        # Fallback to default
        return np.random.uniform(0.1, 1.0, expected_shape)
    
    def _parse_vector_from_gnn(self, vector_data: Any, expected_length: int) -> np.ndarray:
        """Parse vector from GNN initial parameterization."""
        try:
            if isinstance(vector_data, str):
                vector_data = eval(vector_data)
            
            if isinstance(vector_data, (list, tuple)):
                vector = np.array(vector_data, dtype=float)
                if len(vector) != expected_length:
                    self.logger.warning(f"Vector length {len(vector)} doesn't match expected {expected_length}")
                    vector = vector[:expected_length]  # Truncate or pad as needed
                    if len(vector) < expected_length:
                        vector = np.pad(vector, (0, expected_length - len(vector)), 'constant')
                return vector
            
        except Exception as e:
            self.logger.warning(f"Failed to parse vector from GNN data: {e}, using default")
        
        # Fallback to default
        return np.random.uniform(-1.0, 1.0, expected_length)
    
    def create_simple_environment(self):
        """Create a simple discrete environment for the POMDP."""
        class SimpleDiscreteEnvironment:
            def __init__(self, num_states, num_observations, num_actions):
                self.num_states = num_states
                self.num_observations = num_observations
                self.num_actions = num_actions
                self.current_state = 0
                self.step_count = 0
                
            def reset(self):
                self.current_state = np.random.choice(self.num_states)
                self.step_count = 0
                return self.current_state
            
            def step(self, action):
                # Simple state transition
                self.current_state = (self.current_state + action) % self.num_states
                
                # Generate observation (with some noise)
                if np.random.random() < 0.8:  # 80% accuracy
                    observation = self.current_state % self.num_observations
                else:
                    observation = np.random.choice(self.num_observations)
                
                # Simple reward
                reward = 1.0 if self.current_state == self.num_states - 1 else -0.1
                
                self.step_count += 1
                done = self.step_count >= 20  # Episode ends after 20 steps
                
                return observation, reward, done, {'state': self.current_state}
        
        return SimpleDiscreteEnvironment(self.num_states, self.num_observations, self.num_actions)
    
    def run(self, output_dir: Path) -> Tuple[bool, Dict[str, Any]]:
        """
        Run the complete PyMDP simulation.
        
        Args:
            output_dir: Directory for output files
            
        Returns:
            Tuple of (success, results_dict)
        """
        start_time = time.time()
        
        # Create timestamped output directory
        timestamped_output_dir = create_output_directory_with_timestamp(output_dir, f"pymdp_{self.model_name}")
        self.setup_logging(timestamped_output_dir)
        
        self.logger.info("=" * 60)
        self.logger.info("PYMDP SIMULATION STARTING")
        self.logger.info("=" * 60)
        
        try:
            # Set random seed
            np.random.seed(self.random_seed)
            
            # Create PyMDP model
            self.agent, self.model_matrices = self.create_pymdp_model()
            if self.agent is None:
                return False, {"error": "Failed to create PyMDP model"}
            
            # Create environment
            self.environment = self.create_simple_environment()
            
            # Create visualizer
            if self.save_visualizations:
                self.visualizer = PyMDPVisualizer(timestamped_output_dir)
            
            # Save configuration
            config_file = timestamped_output_dir / 'simulation_config.json'
            safe_json_dump(self.config, config_file)
            
            # Save model matrices
            if self.save_matrices:
                matrices_file = timestamped_output_dir / 'model_matrices.pkl'
                safe_pickle_dump(self.model_matrices, matrices_file)
            
            # Run simulation episodes
            all_traces, performance_metrics = self._run_episodes()
            
            # Save results
            if self.save_traces:
                save_results = save_simulation_results(
                    traces=all_traces,
                    metrics=performance_metrics,
                    config=self.config,
                    model_matrices=self.model_matrices if self.save_matrices else None,
                    output_dir=timestamped_output_dir
                )
                
                for save_type, success in save_results.items():
                    if success:
                        self.logger.info(f"✓ Successfully saved {save_type}")
                    else:
                        self.logger.warning(f"✗ Failed to save {save_type}")
            
            # Generate visualizations
            if self.save_visualizations and self.visualizer:
                self.visualizer.create_comprehensive_visualizations(all_traces, performance_metrics)
            
            # Calculate final metrics
            simulation_duration = time.time() - start_time
            summary = generate_simulation_summary(all_traces, performance_metrics)
            summary['simulation_duration_seconds'] = simulation_duration
            summary['simulation_duration_formatted'] = format_duration(simulation_duration)
            summary['output_directory'] = str(timestamped_output_dir)
            
            # Save summary
            summary_file = timestamped_output_dir / 'simulation_summary.json'
            safe_json_dump(summary, summary_file)
            
            self.logger.info("SIMULATION COMPLETE")
            self.logger.info(f"Total Episodes: {summary['total_episodes']}")
            self.logger.info(f"Success Rate: {summary.get('success_rate', 0):.2%}")
            self.logger.info(f"Duration: {summary['simulation_duration_formatted']}")
            self.logger.info(f"Output: {timestamped_output_dir}")
            
            return True, summary
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            return False, {"error": str(e)}
    
    def _run_episodes(self) -> Tuple[List[Dict], Dict[str, List]]:
        """Run simulation episodes and collect data."""
        all_traces = []
        performance_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'belief_entropies': [],
            'success_rates': []
        }
        
        successful_episodes = 0
        
        for episode in range(self.num_episodes):
            self.logger.info(f"Running episode {episode + 1}/{self.num_episodes}")
            
            # Reset environment
            true_state = self.environment.reset()
            
            # Episode trace
            episode_trace = {
                'episode': episode,
                'true_states': [],
                'observations': [],
                'actions': [],
                'rewards': [],
                'beliefs': []
            }
            
            episode_reward = 0
            episode_length = 0
            episode_entropies = []
            
            # Run episode
            for step in range(self.max_steps_per_episode):
                # Get observation from environment (dummy step first)
                obs, reward, done, info = self.environment.step(0)
                
                # Agent inference
                qs = self.agent.infer_states([obs])
                q_pi, _ = self.agent.infer_policies()
                action = self.agent.sample_action()
                
                # Execute action
                action_idx = int(action[0]) if hasattr(action[0], '__iter__') else int(action[0])
                obs, reward, done, info = self.environment.step(action_idx)
                
                # Calculate belief entropy
                belief_entropy = -np.sum(qs[0] * np.log(qs[0] + 1e-16))
                episode_entropies.append(belief_entropy)
                
                # Store trace
                episode_trace['true_states'].append(info.get('state', 0))
                episode_trace['observations'].append(obs)
                episode_trace['actions'].append(action_idx)
                episode_trace['rewards'].append(reward)
                episode_trace['beliefs'].append(qs[0].copy())
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Check success (simple: positive total reward)
            if episode_reward > 0:
                successful_episodes += 1
            
            # Store episode results
            all_traces.append(episode_trace)
            performance_metrics['episode_rewards'].append(episode_reward)
            performance_metrics['episode_lengths'].append(episode_length)
            performance_metrics['belief_entropies'].append(np.mean(episode_entropies))
            performance_metrics['success_rates'].append(successful_episodes / (episode + 1))
            
            self.logger.info(f"Episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}")
        
        return all_traces, performance_metrics 