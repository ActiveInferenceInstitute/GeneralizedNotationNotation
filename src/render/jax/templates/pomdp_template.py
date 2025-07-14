"""
JAX POMDP Template

Comprehensive POMDP implementation template with JIT compilation, belief updates,
value iteration, and alpha vector backup. Optimized for performance and numerical stability.

@Web: https://github.com/google/jax
@Web: https://optax.readthedocs.io
@Web: https://pfjax.readthedocs.io
@Web: https://juliapomdp.github.io/POMDPs.jl/latest/def_pomdp/
"""

POMDP_TEMPLATE = '''
"""
JAX POMDP Solver Generated from GNN Specification: {model_name}

This file contains a complete POMDP implementation with JIT compilation,
belief updates, value iteration, and alpha vector backup.

Generated on: {timestamp}
Source: {source_file}
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from functools import partial
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import json
from pathlib import Path

# Configure JAX for optimal performance
jax.config.update('jax_enable_x64', True)  # Use double precision for numerical stability
jax.config.update('jax_debug_nans', False)  # Disable NaN checking in production

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JAXPOMDPSolver:
    """
    JAX-optimized POMDP solver with belief updates, value iteration, and alpha vector backup.
    
    Features:
    - JIT-compiled belief updates for maximum performance
    - Alpha vector backup with vectorization
    - Value iteration with convergence checking
    - Numerical stability in all operations
    - Device-agnostic execution (CPU/GPU/TPU)
    """
    
    def __init__(self, 
                 A: jnp.ndarray,  # Observation model P(o|s)
                 B: List[jnp.ndarray],  # Transition models P(s'|s,u) for each action
                 C: jnp.ndarray,  # Preferences over observations
                 D: jnp.ndarray,  # Prior over initial states
                 discount: float = 0.95,
                 epsilon: float = 1e-6,
                 max_iterations: int = 1000):
        """
        Initialize POMDP solver with model parameters.
        
        Args:
            A: Observation model matrix [n_observations, n_states]
            B: List of transition matrices [n_actions][n_states, n_states]
            C: Preference vector [n_observations]
            D: Initial state prior [n_states]
            discount: Discount factor for future rewards
            epsilon: Convergence threshold for value iteration
            max_iterations: Maximum iterations for value iteration
        """
        self.A = jnp.array(A, dtype=jnp.float64)
        self.B = [jnp.array(B_i, dtype=jnp.float64) for B_i in B]
        self.C = jnp.array(C, dtype=jnp.float64)
        self.D = jnp.array(D, dtype=jnp.float64)
        self.discount = discount
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        
        # Extract dimensions
        self.n_states = self.A.shape[1]
        self.n_observations = self.A.shape[0]
        self.n_actions = len(self.B)
        
        # Validate dimensions
        assert self.A.shape == (self.n_observations, self.n_states), f"A matrix shape mismatch: {self.A.shape}"
        for i, B_i in enumerate(self.B):
            assert B_i.shape == (self.n_states, self.n_states), f"B[{i}] matrix shape mismatch: {B_i.shape}"
        assert self.C.shape == (self.n_observations,), f"C vector shape mismatch: {self.C.shape}"
        assert self.D.shape == (self.n_states,), f"D vector shape mismatch: {self.D.shape}"
        
        logger.info(f"POMDP initialized: {self.n_states} states, {self.n_observations} observations, {self.n_actions} actions")
    
    @partial(jax.jit, static_argnums=(0,))
    def belief_update(self, belief: jnp.ndarray, action: int, observation: int) -> jnp.ndarray:
        """
        JIT-compiled belief update: b'(s') = P(s'|b,a,o) = P(o|s') * sum_s P(s'|s,a) * b(s) / P(o|b,a)
        
        Args:
            belief: Current belief state [n_states]
            action: Action taken
            observation: Observation received
            
        Returns:
            Updated belief state [n_states]
        """
        # P(s'|s,a) * b(s) for all s, s'
        belief_prediction = self.B[action] @ belief
        
        # P(o|s') * P(s'|s,a) * b(s)
        numerator = self.A[observation, :] * belief_prediction
        
        # Normalize: P(o|b,a) = sum_s' P(o|s') * P(s'|s,a) * b(s)
        denominator = jnp.sum(numerator)
        
        # Avoid division by zero
        denominator = jnp.where(denominator > 1e-10, denominator, 1e-10)
        
        return numerator / denominator
    
    @partial(jax.jit, static_argnums=(0,))
    def alpha_vector_backup(self, belief_points: jnp.ndarray, alpha_vectors: jnp.ndarray) -> jnp.ndarray:
        """
        JIT-compiled alpha vector backup for point-based value iteration.
        
        Args:
            belief_points: Belief points to evaluate [n_points, n_states]
            alpha_vectors: Current alpha vectors [n_vectors, n_states]
            
        Returns:
            New alpha vector [n_states]
        """
        def backup_single_action(action):
            # For each belief point, compute expected value
            def compute_value_for_belief(belief):
                # Expected immediate reward: sum_s b(s) * sum_o P(o|s) * C(o)
                immediate_reward = jnp.sum(belief * jnp.sum(self.A * self.C, axis=0))
                
                # Expected future value: sum_o P(o|b,a) * max_alpha sum_s' b'(s') * alpha(s')
                future_values = []
                
                for obs in range(self.n_observations):
                    # Update belief for this observation
                    updated_belief = self.belief_update(belief, action, obs)
                    
                    # Find maximum value over alpha vectors
                    alpha_values = alpha_vectors @ updated_belief
                    max_value = jnp.max(alpha_values)
                    
                    # P(o|b,a) = sum_s' P(o|s') * sum_s P(s'|s,a) * b(s)
                    obs_prob = jnp.sum(self.A[obs, :] * (self.B[action] @ belief))
                    future_values.append(obs_prob * max_value)
                
                total_future_value = jnp.sum(jnp.array(future_values))
                return immediate_reward + self.discount * total_future_value
            
            # Vectorize over belief points
            values = jax.vmap(compute_value_for_belief)(belief_points)
            
            # Return action that maximizes value
            return jnp.max(values)
        
        # Compute value for each action
        action_values = jax.vmap(backup_single_action)(jnp.arange(self.n_actions))
        
        # Find best action
        best_action = jnp.argmax(action_values)
        
        # Compute alpha vector for best action
        def compute_alpha_vector():
            alpha = jnp.zeros(self.n_states)
            
            for s in range(self.n_states):
                # Immediate reward for state s
                immediate_reward = jnp.sum(self.A[:, s] * self.C)
                
                # Future value for state s
                future_value = 0.0
                for obs in range(self.n_observations):
                    # P(o|s)
                    obs_prob = self.A[obs, s]
                    
                    # P(s'|s,a) for best action
                    state_transition = self.B[best_action][:, s]
                    
                    # max_alpha sum_s' P(s'|s,a) * alpha(s')
                    alpha_values = alpha_vectors @ state_transition
                    max_value = jnp.max(alpha_values)
                    
                    future_value += obs_prob * max_value
                
                alpha = alpha.at[s].set(immediate_reward + self.discount * future_value)
            
            return alpha
        
        return compute_alpha_vector()
    
    def value_iteration(self, belief_points: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, List[int]]:
        """
        Point-based value iteration with convergence checking.
        
        Args:
            belief_points: Belief points to evaluate (if None, use uniform grid)
            
        Returns:
            Tuple of (alpha_vectors, policy)
        """
        if belief_points is None:
            # Generate uniform belief points
            n_points = min(100, 2**self.n_states)  # Limit number of points
            belief_points = self._generate_uniform_belief_points(n_points)
        
        belief_points = jnp.array(belief_points, dtype=jnp.float64)
        n_points = belief_points.shape[0]
        
        # Initialize alpha vectors with immediate rewards
        alpha_vectors = jnp.zeros((self.n_actions, self.n_states), dtype=jnp.float64)
        for a in range(self.n_actions):
            for s in range(self.n_states):
                alpha_vectors = alpha_vectors.at[a, s].set(jnp.sum(self.A[:, s] * self.C))
        
        logger.info(f"Starting value iteration with {n_points} belief points")
        
        # Value iteration loop
        for iteration in range(self.max_iterations):
            old_alpha_vectors = alpha_vectors.copy()
            
            # Backup for each belief point
            new_alpha_vectors = []
            for i in range(n_points):
                new_alpha = self.alpha_vector_backup(belief_points[i:i+1], alpha_vectors)
                new_alpha_vectors.append(new_alpha)
            
            # Add new alpha vectors
            alpha_vectors = jnp.vstack([alpha_vectors] + new_alpha_vectors)
            
            # Check convergence
            max_change = jnp.max(jnp.abs(alpha_vectors - old_alpha_vectors))
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: max change = {max_change:.6f}")
            
            if max_change < self.epsilon:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
        
        # Extract policy
        policy = self._extract_policy(alpha_vectors, belief_points)
        
        return alpha_vectors, policy
    
    def _generate_uniform_belief_points(self, n_points: int) -> jnp.ndarray:
        """Generate uniform belief points for point-based value iteration."""
        import numpy as np
        
        # Generate random belief points
        points = []
        for _ in range(n_points):
            # Generate random values and normalize
            belief = np.random.dirichlet(np.ones(self.n_states))
            points.append(belief)
        
        return jnp.array(points, dtype=jnp.float64)
    
    def _extract_policy(self, alpha_vectors: jnp.ndarray, belief_points: jnp.ndarray) -> List[int]:
        """Extract policy from alpha vectors."""
        def get_best_action(belief):
            values = alpha_vectors @ belief
            return jnp.argmax(values)
        
        policy = jax.vmap(get_best_action)(belief_points)
        return policy.tolist()
    
    def solve(self) -> Dict[str, Any]:
        """
        Solve the POMDP and return results.
        
        Returns:
            Dictionary containing solution and metadata
        """
        start_time = time.time()
        
        # Run value iteration
        alpha_vectors, policy = self.value_iteration()
        
        # Compute initial belief value
        initial_value = jnp.max(alpha_vectors @ self.D)
        
        # Performance metrics
        solve_time = time.time() - start_time
        
        results = {
            "alpha_vectors": alpha_vectors.tolist(),
            "policy": policy,
            "initial_belief": self.D.tolist(),
            "initial_value": float(initial_value),
            "solve_time": solve_time,
            "n_states": self.n_states,
            "n_observations": self.n_observations,
            "n_actions": self.n_actions,
            "discount": self.discount,
            "epsilon": self.epsilon
        }
        
        logger.info(f"POMDP solved in {solve_time:.3f}s, initial value: {initial_value:.6f}")
        
        return results

def main():
    """Main function to run the POMDP solver."""
    
    # Model parameters from GNN specification
    {model_parameters}
    
    # Create and solve POMDP
    solver = JAXPOMDPSolver(A=A, B=B, C=C, D=D, discount={discount})
    results = solver.solve()
    
    # Save results
    output_dir = Path("output/execution_results/jax")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "{model_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"POMDP solved successfully!")
    print(f"Initial value: {results['initial_value']:.6f}")
    print(f"Solve time: {results['solve_time']:.3f}s")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
''' 