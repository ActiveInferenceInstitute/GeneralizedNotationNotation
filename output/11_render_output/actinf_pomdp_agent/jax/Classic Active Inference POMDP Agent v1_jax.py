"""
JAX Model Generated from GNN Specification: GNNModel

This model implements the GNN specification using JAX for high-performance computation.
Generated automatically by the GNN Processing Pipeline.

@Web: https://github.com/google/jax
@Web: https://flax.readthedocs.io
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Optional, Tuple
import numpy as np

class GNNModelModel(nn.Module):
    """
    JAX implementation of GNNModel using Flax.
    
    This model implements the GNN specification with JAX optimizations.
    """
    
    def setup(self):
        """Initialize model parameters from GNN specification."""
        # Extract dimensions from matrices
        self.num_states = 3
        self.num_observations = 3
        self.num_actions = 1
        
        # Initialize matrices as learnable parameters
        self.A_matrix = self.param('A_matrix', 
            lambda key, shape: jnp.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]), 
            (self.num_observations, self.num_states))
        
        self.B_matrix = self.param('B_matrix',
            lambda key, shape: jnp.array([[[1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [1.0]]]),
            (self.num_states, self.num_states, self.num_actions))
        
        self.C_vector = self.param('C_vector',
            lambda key, shape: jnp.array([0.1, 0.1, 1.0]),
            (self.num_observations,))
        
        self.D_vector = self.param('D_vector',
            lambda key, shape: jnp.array([0.33333, 0.33333, 0.33333]),
            (self.num_states,))
    
    @nn.compact
    def __call__(self, inputs: Dict[str, jnp.ndarray], training: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Forward pass of the model based on GNN specification.
        
        This implements the core Active Inference computation including:
        - State inference from observations
        - Belief updates using the A matrix (likelihood)
        - Action selection based on expected free energy
        - State transitions using the B matrix
        
        Args:
            inputs: Dictionary containing input data with keys:
                - 'observations': Observation data [batch_size, num_observations]
                - 'actions': Optional action data [batch_size, num_actions]
                - 'beliefs': Optional initial beliefs [batch_size, num_states]
            training: Whether in training mode
            
        Returns:
            Dictionary containing model outputs:
                - 'beliefs': Updated belief states
                - 'actions': Selected actions
                - 'expected_free_energy': Expected free energy values
                - 'state_predictions': Predicted next states
        """
        # Extract input components
        observations = inputs.get('observations', jnp.zeros((1, self.num_observations)))
        actions = inputs.get('actions', jnp.zeros((1, self.num_actions)))
        initial_beliefs = inputs.get('beliefs', self.D_vector)
        
        batch_size = observations.shape[0]
        
        # Ensure beliefs have correct shape
        if initial_beliefs.ndim == 1:
            initial_beliefs = jnp.expand_dims(initial_beliefs, 0)
        
        # 1. State Inference (Bayesian belief update)
        # P(s|o) ∝ P(o|s) * P(s) using the A matrix (likelihood)
        def belief_update(belief, obs):
            # Compute likelihood P(o|s) using A matrix
            likelihood = jnp.dot(self.A_matrix, belief)  # [num_observations, num_states] @ [num_states] = [num_observations]
            
            # Apply observation to get P(o|s) for this specific observation
            obs_likelihood = jnp.where(obs > 0, likelihood, 1.0)  # Handle sparse observations
            
            # Bayesian update: P(s|o) ∝ P(o|s) * P(s)
            updated_belief = belief * obs_likelihood
            
            # Normalize
            updated_belief = updated_belief / (jnp.sum(updated_belief) + 1e-8)
            return updated_belief
        
        # Update beliefs for each observation
        beliefs = jax.vmap(belief_update)(initial_beliefs, observations)
        
        # 2. Action Selection (Expected Free Energy)
        # Compute expected free energy for each action
        def compute_efe(belief, action):
            # Expected free energy = -log P(o) + KL[q(s)||p(s)]
            
            # Predict next state using B matrix (transitions)
            next_belief = jnp.dot(self.B_matrix[:, :, action], belief)
            
            # Predict observations using A matrix
            predicted_obs = jnp.dot(self.A_matrix, next_belief)
            
            # Compute entropy of predicted observations
            obs_entropy = -jnp.sum(predicted_obs * jnp.log(predicted_obs + 1e-8))
            
            # Compute KL divergence between current and prior beliefs
            kl_divergence = jnp.sum(belief * jnp.log((belief + 1e-8) / (self.D_vector + 1e-8)))
            
            # Expected free energy
            efe = obs_entropy + kl_divergence
            return efe
        
        # Compute EFE for all actions
        action_efes = jax.vmap(lambda belief: jax.vmap(lambda action: compute_efe(belief, action))(jnp.arange(self.num_actions)))(beliefs)
        
        # Select action with minimum expected free energy
        selected_actions = jnp.argmin(action_efes, axis=-1)
        
        # 3. State Prediction
        # Predict next states using selected actions and B matrix
        def predict_next_state(belief, action):
            return jnp.dot(self.B_matrix[:, :, action], belief)
        
        state_predictions = jax.vmap(predict_next_state)(beliefs, selected_actions)
        
        # 4. Compute final expected free energy values
        final_efes = jnp.take_along_axis(action_efes, jnp.expand_dims(selected_actions, -1), axis=-1).squeeze(-1)
        
        return {
            "beliefs": beliefs,
            "actions": selected_actions,
            "expected_free_energy": final_efes,
            "state_predictions": state_predictions,
            "action_efes": action_efes,  # EFE for all actions
            "predicted_observations": jnp.dot(observations, self.A_matrix)  # Predicted observations
        }

def create_model() -> GNNModelModel:
    """Create and return a new instance of the model."""
    return GNNModelModel()

def get_model_summary(num_states: int, num_observations: int, num_actions: int, variables: Dict) -> str:
    """Get a summary of the model architecture."""
    return f"""
GNNModel Model Summary:
- Number of states: {num_states}
- Number of observations: {num_observations}
- Number of actions: {num_actions}
- Parameters: {sum(p.size for p in jax.tree_util.tree_leaves(variables))}
"""

if __name__ == "__main__":
    # Example usage
    model = create_model()
    
    # Initialize model variables
    key = jax.random.PRNGKey(0)
    variables = model.init(key, {"observations": jnp.zeros((1, 3))})
    
    print(get_model_summary(3, 3, 1, variables))
    
    # Test forward pass
    outputs = model.apply(variables, {"observations": jnp.zeros((1, 3))})
    print("Model test successful!")
    print(f"Output keys: {list(outputs.keys())}")
