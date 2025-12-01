"""
JAX Model Generated from GNN Specification: GNNModel

This model implements the GNN specification using pure JAX for high-performance computation.
Generated automatically by the GNN Processing Pipeline.

This is a standalone JAX implementation that only requires jax and numpy.
No external dependencies like Flax or Optax are required.

@Web: https://github.com/google/jax
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Dict, Any, Tuple
import numpy as np


# Model configuration
NUM_STATES = 3
NUM_OBSERVATIONS = 3
NUM_ACTIONS = 1


def create_params() -> Dict[str, jnp.ndarray]:
    """Create model parameters from GNN specification."""
    return {
        'A_matrix': jnp.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]),  # Observation model P(o|s)
        'B_matrix': jnp.array([[[1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [1.0]]]),  # Transition model P(s'|s,a)
        'C_vector': jnp.array([0.1, 0.1, 1.0]),  # Preferences over observations
        'D_vector': jnp.array([0.33333, 0.33333, 0.33333]),  # Prior over initial states
    }


@jit
def belief_update(params: Dict[str, jnp.ndarray], belief: jnp.ndarray, 
                  observation: jnp.ndarray) -> jnp.ndarray:
    """
    Bayesian belief update given an observation.
    
    P(s|o) ∝ P(o|s) * P(s) using the A matrix (likelihood)
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        observation: Observation vector [num_observations]
        
    Returns:
        Updated belief state [num_states]
    """
    A_matrix = params['A_matrix']
    
    # Compute likelihood P(o|s) for each state
    # For each state s, compute sum over observations: A[o,s] * observation[o]
    likelihood = jnp.dot(observation, A_matrix)  # [num_states]
    
    # Bayesian update: P(s|o) ∝ P(o|s) * P(s)
    updated_belief = belief * likelihood
    
    # Normalize with numerical stability
    normalizer = jnp.sum(updated_belief) + 1e-8
    updated_belief = updated_belief / normalizer
    
    return updated_belief


@jit
def compute_expected_free_energy(params: Dict[str, jnp.ndarray], belief: jnp.ndarray, 
                                  action: int) -> float:
    """
    Compute expected free energy (EFE) for a given action.
    
    EFE = -E[log P(o)] + KL[Q(s')||P(s')]
        = Entropy of predicted observations + KL divergence
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        action: Action index
        
    Returns:
        Expected free energy value
    """
    A_matrix = params['A_matrix']
    B_matrix = params['B_matrix']
    C_vector = params['C_vector']
    D_vector = params['D_vector']
    
    # Predict next state distribution using B matrix (transitions)
    # B_matrix[:, :, action] is [num_states, num_states]
    next_belief = jnp.dot(B_matrix[:, :, action], belief)
    next_belief = next_belief / (jnp.sum(next_belief) + 1e-8)  # Normalize
    
    # Predict observation distribution using A matrix
    predicted_obs = jnp.dot(A_matrix, next_belief)
    predicted_obs = predicted_obs / (jnp.sum(predicted_obs) + 1e-8)  # Normalize
    
    # Compute epistemic value (expected information gain)
    # This is the negative entropy of predicted observations
    obs_entropy = -jnp.sum(predicted_obs * jnp.log(predicted_obs + 1e-8))
    
    # Compute pragmatic value (expected utility based on preferences)
    # Higher values in C_vector mean more preferred observations
    pragmatic_value = jnp.dot(predicted_obs, C_vector)
    
    # Compute KL divergence between predicted state and prior
    kl_divergence = jnp.sum(
        jnp.where(next_belief > 1e-8,
                  next_belief * jnp.log((next_belief + 1e-8) / (D_vector + 1e-8)),
                  0.0)
    )
    
    # Expected free energy = uncertainty - utility + complexity
    efe = obs_entropy - pragmatic_value + 0.1 * kl_divergence
    
    return efe


@jit
def select_action(params: Dict[str, jnp.ndarray], belief: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    """
    Select action with minimum expected free energy.
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        
    Returns:
        Tuple of (selected_action_index, efe_values_for_all_actions)
    """
    # Compute EFE for all actions
    efe_values = jnp.array([
        compute_expected_free_energy(params, belief, a) 
        for a in range(NUM_ACTIONS)
    ])
    
    # Select action with minimum EFE
    selected_action = jnp.argmin(efe_values)
    
    return selected_action, efe_values


@jit
def state_transition(params: Dict[str, jnp.ndarray], belief: jnp.ndarray, 
                     action: int) -> jnp.ndarray:
    """
    Predict next state distribution given current belief and action.
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        action: Action index
        
    Returns:
        Predicted next state distribution [num_states]
    """
    B_matrix = params['B_matrix']
    
    # Apply transition model
    next_belief = jnp.dot(B_matrix[:, :, action], belief)
    
    # Normalize
    next_belief = next_belief / (jnp.sum(next_belief) + 1e-8)
    
    return next_belief


def simulate_step(params: Dict[str, jnp.ndarray], belief: jnp.ndarray, 
                  observation: jnp.ndarray) -> Dict[str, Any]:
    """
    Perform one step of Active Inference simulation.
    
    This includes:
    1. Belief update given observation
    2. Action selection based on expected free energy
    3. State prediction for selected action
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        observation: Observation vector [num_observations]
        
    Returns:
        Dictionary with simulation results
    """
    # 1. Update belief based on observation
    updated_belief = belief_update(params, belief, observation)
    
    # 2. Select action
    selected_action, efe_values = select_action(params, updated_belief)
    
    # 3. Predict next state
    predicted_next_state = state_transition(params, updated_belief, selected_action)
    
    return {
        'belief': updated_belief,
        'action': selected_action,
        'expected_free_energy': efe_values[selected_action],
        'all_efe_values': efe_values,
        'predicted_next_state': predicted_next_state,
    }


def run_simulation(params: Dict[str, jnp.ndarray], observations: jnp.ndarray, 
                   initial_belief: jnp.ndarray = None) -> Dict[str, Any]:
    """
    Run a full Active Inference simulation over a sequence of observations.
    
    Args:
        params: Model parameters dictionary
        observations: Sequence of observations [num_steps, num_observations]
        initial_belief: Optional initial belief (default: uniform)
        
    Returns:
        Dictionary with simulation trajectory
    """
    num_steps = observations.shape[0]
    
    # Initialize belief
    if initial_belief is None:
        belief = params['D_vector'].copy()
    else:
        belief = initial_belief
    
    # Storage for trajectory
    beliefs = []
    actions = []
    efes = []
    
    # Run simulation
    for t in range(num_steps):
        result = simulate_step(params, belief, observations[t])
        
        beliefs.append(result['belief'])
        actions.append(result['action'])
        efes.append(result['expected_free_energy'])
        
        # Update belief for next step (using predicted next state)
        belief = result['predicted_next_state']
    
    return {
        'beliefs': jnp.stack(beliefs),
        'actions': jnp.array(actions),
        'expected_free_energies': jnp.array(efes),
        'final_belief': belief,
    }


def get_model_summary() -> str:
    """Get a summary of the model architecture."""
    return f"""
GNNModel Model Summary (Pure JAX Implementation):
- Number of states: {NUM_STATES}
- Number of observations: {NUM_OBSERVATIONS}
- Number of actions: {NUM_ACTIONS}
- Parameters: A matrix, B matrix, C vector, D vector

Key Functions:
- create_params(): Create model parameters
- belief_update(): Bayesian belief update
- compute_expected_free_energy(): Compute EFE for action
- select_action(): Select action with minimum EFE
- simulate_step(): One step of Active Inference
- run_simulation(): Full simulation over observations
"""


if __name__ == "__main__":
    print("=" * 60)
    print("JAX Active Inference Model: GNNModel")
    print("=" * 60)
    
    # Create parameters
    params = create_params()
    print("\n✅ Model parameters created")
    print(f"   A matrix shape: {params['A_matrix'].shape}")
    print(f"   B matrix shape: {params['B_matrix'].shape}")
    print(f"   C vector shape: {params['C_vector'].shape}")
    print(f"   D vector shape: {params['D_vector'].shape}")
    
    # Print model summary
    print(get_model_summary())
    
    # Test with uniform initial belief
    print("\n🧪 Running test simulation...")
    initial_belief = params['D_vector']
    
    # Create a test observation (one-hot for first observation)
    test_obs = jnp.zeros(NUM_OBSERVATIONS)
    test_obs = test_obs.at[0].set(1.0)
    
    # Run one simulation step
    result = simulate_step(params, initial_belief, test_obs)
    
    print(f"\n📊 Simulation Results:")
    print(f"   Initial belief: {initial_belief}")
    print(f"   Updated belief: {result['belief']}")
    print(f"   Selected action: {result['action']}")
    print(f"   Expected free energy: {result['expected_free_energy']:.4f}")
    print(f"   EFE for all actions: {result['all_efe_values']}")
    
    # Run multi-step simulation
    print("\n🔄 Running 5-step simulation...")
    observations = jnp.eye(NUM_OBSERVATIONS)[:min(5, NUM_OBSERVATIONS)]  # Use identity as test observations
    if observations.shape[0] < 5:
        # Pad with repeated observations if needed
        observations = jnp.concatenate([observations] * (5 // observations.shape[0] + 1))[:5]
    
    trajectory = run_simulation(params, observations)
    
    print(f"   Actions taken: {trajectory['actions']}")
    print(f"   Final belief: {trajectory['final_belief']}")
    print(f"   Average EFE: {jnp.mean(trajectory['expected_free_energies']):.4f}")
    
    print("\n✅ JAX Active Inference model test successful!")
