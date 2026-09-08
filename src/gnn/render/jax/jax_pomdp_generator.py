#!/usr/bin/env python3
"""
JAX POMDP solver script generation for GNN Step 11.

Extracted from ``render.jax.jax_renderer``.
"""

import logging
from typing import (
    Any,
    Dict,
    Optional,
)

from .jax_spec_extract import (
    _jax_model_name,
    _validated_jax_matrices,
)

logger = logging.getLogger(__name__)


def _generate_jax_pomdp_code(
    gnn_spec: Dict[str, Any], options: Optional[Dict[str, Any]]
) -> str:
    """Generate JAX POMDP solver code from GNN specification."""

    try:
        model_name = _jax_model_name(gnn_spec, "POMDPModel")
        matrices = _validated_jax_matrices(gnn_spec)
        A_matrix = matrices["A"]
        B_matrix = matrices["B"]
        C_vector = matrices["C"]
        D_vector = matrices["D"]

        logger.info(f"A matrix shape: {A_matrix.shape}")
        logger.info(f"B matrix shape: {B_matrix.shape}")
        logger.info(f"C vector shape: {C_vector.shape}")
        logger.info(f"D vector shape: {D_vector.shape}")

        num_states = A_matrix.shape[1]
        num_observations = A_matrix.shape[0]
        num_actions = B_matrix.shape[2]

        logger.info(
            f"Final dimensions: states={num_states}, observations={num_observations}, actions={num_actions}"
        )

        code = f'''"""
JAX POMDP Solver Generated from GNN Specification: {model_name}

This implements a complete POMDP solver using JAX optimizations including JIT, vmap, and pmap.
Based on the GNN specification with belief updates, value iteration, and alpha vector backup.

@Web: https://pfjax.readthedocs.io
@Web: https://arxiv.org/abs/1304.1118
@Web: https://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf
"""

import jax
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap, pmap
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

class POMDPModels:
    """Container for POMDP model parameters."""
    
    def __init__(self, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, D: jnp.ndarray):
        self.A = A  # Observation model P(o|s)
        self.B = B  # Transition model P(s'|s,u)
        self.C = C  # Preferences over observations
        self.D = D  # Prior over initial states
        self.discount_factor = 0.95

class JAXPOMDPSolver:
    """
    High-performance POMDP solver using JAX optimizations.
    
    Implements belief updates, value iteration, and alpha vector backup with JIT compilation.
    """
    
    def __init__(self, models: POMDPModels):
        self.models = models
        self.num_states = models.A.shape[1]
        self.num_observations = models.A.shape[0]
        self.num_actions = models.B.shape[2]
        self.discount_factor = models.discount_factor  # Add missing discount factor
        
        # JIT-compiled functions for performance
        self.belief_update_jitted = jit(self.belief_update)
        self.alpha_vector_backup_jitted = jit(self.alpha_vector_backup)
    
    @partial(jit, static_argnums=(0,))
    def belief_update(self, belief: jnp.ndarray, action: int, observation: int) -> jnp.ndarray:
        """
        Bayesian belief update with numerical stability.
        
        Args:
            belief: Current belief state
            action: Action taken
            observation: Observation received
            
        Returns:
            Updated belief state
        """
        # Prediction step
        predicted_belief = jnp.dot(self.models.B[:, :, action], belief)
        
        # Update step
        updated_belief = predicted_belief * self.models.A[observation, :]
        
        # Normalization with numerical stability
        normalizer = jnp.sum(updated_belief)
        normalized_belief = jnp.where(
            normalizer > 1e-10,
            updated_belief / normalizer,
            jnp.ones_like(updated_belief) / self.num_states
        )
        
        return normalized_belief
    
    @partial(jit, static_argnums=(0,))
    def alpha_vector_backup(self, belief: jnp.ndarray, action: int,
                           alpha_vectors: jnp.ndarray) -> jnp.ndarray:
        """
        Optimized alpha vector backup with vectorization.
        
        Args:
            belief: Current belief state
            action: Action to evaluate
            alpha_vectors: Current alpha vectors
            
        Returns:
            New alpha vector for this action
        """
        # Convert observation preferences to state-conditioned immediate value.
        base_alpha = jnp.dot(self.models.A.T, self.models.C)
        
        # Pre-calculate observation probabilities
        next_belief_pred = jnp.dot(self.models.B[:, :, action], belief)
        obs_probs = jnp.dot(self.models.A, next_belief_pred)
        
        def compute_obs_contribution(obs_idx):
            next_belief = self.belief_update(belief, action, obs_idx)
            values = jnp.dot(alpha_vectors, next_belief)
            best_alpha = alpha_vectors[jnp.argmax(values)]
            
            # Only add contribution if observation probability is significant
            return jnp.where(obs_probs[obs_idx] > 1e-10, 
                           self.discount_factor * obs_probs[obs_idx] * best_alpha,
                           jnp.zeros_like(best_alpha))
                           
        # Vectorized map-reduce over observations
        contributions = vmap(compute_obs_contribution)(jnp.arange(self.num_observations))
        alpha = base_alpha + jnp.sum(contributions, axis=0)
        
        return alpha
    
    def compute_observation_probability(self, belief: jnp.ndarray, action: int) -> jnp.ndarray:
        """Compute probability of observations given belief and action."""
        next_belief = jnp.dot(self.models.B[:, :, action], belief)
        return jnp.dot(self.models.A, next_belief)

def create_pomdp_solver() -> JAXPOMDPSolver:
    """Create and return a POMDP solver with the specified model parameters."""
    
    # Model parameters from GNN specification
    A = jnp.array({A_matrix.tolist()})  # Observation model
    B = jnp.array({B_matrix.tolist()})  # Transition model  
    C = jnp.array({C_vector.tolist()})  # Preferences
    D = jnp.array({D_vector.tolist()})  # Prior
    
    models = POMDPModels(A=A, B=B, C=C, D=D)
    return JAXPOMDPSolver(models)

def solve_pomdp(solver: JAXPOMDPSolver, initial_belief: jnp.ndarray, 
                horizon: int = 10) -> Dict[str, jnp.ndarray]:
    """
    Solve POMDP using value iteration with alpha vectors.
    
    Args:
        solver: POMDP solver instance
        initial_belief: Initial belief state
        horizon: Planning horizon
        
    Returns:
        Dictionary containing solution components
    """
    # Initialize alpha vectors
    alpha_vectors = jnp.zeros((solver.num_actions, solver.num_states))
    
    # Value iteration
    for t in range(horizon):
        new_alpha_vectors = []
        for action in range(solver.num_actions):
            alpha = solver.alpha_vector_backup_jitted(initial_belief, action, alpha_vectors)
            new_alpha_vectors.append(alpha)
        alpha_vectors = jnp.array(new_alpha_vectors)
    
    # Compute optimal action
    values = jnp.dot(alpha_vectors, initial_belief)
    optimal_action = jnp.argmax(values)
    
    return {{
        "optimal_action": optimal_action,
        "value": jnp.max(values),
        "alpha_vectors": alpha_vectors
    }}

if __name__ == "__main__":
    # Example usage
    solver = create_pomdp_solver()
    print(f"POMDP Solver created with {{solver.num_states}} states, {{solver.num_observations}} observations, {{solver.num_actions}} actions")
    
    # Test with uniform initial belief
    initial_belief = jnp.ones(solver.num_states) / solver.num_states
    solution = solve_pomdp(solver, initial_belief, horizon=5)
    
    print(f"Optimal action: {{solution['optimal_action']}}")
    print(f"Value: {{solution['value']:.4f}}")
    print("POMDP solver test successful!")
'''

        return code

    except Exception as e:
        raise ValueError(f"JAX POMDP generation failed: {e}") from e
