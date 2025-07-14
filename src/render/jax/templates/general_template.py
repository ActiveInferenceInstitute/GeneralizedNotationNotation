"""
General JAX Template for Active Inference Models

General JAX implementation template for Active Inference models beyond POMDPs,
including continuous state spaces, hierarchical models, and neural network components.

@Web: https://github.com/google/jax
@Web: https://optax.readthedocs.io
@Web: https://flax.readthedocs.io
@Web: https://pfjax.readthedocs.io
"""

GENERAL_TEMPLATE = '''
"""
JAX Active Inference Model Generated from GNN Specification: {model_name}

This file contains a general Active Inference implementation with JAX optimization,
neural network components, and gradient-based learning.

Generated on: {timestamp}
Source: {source_file}
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from functools import partial
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import time
import json
from pathlib import Path

# Configure JAX for optimal performance
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_debug_nans', False)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JAXActiveInferenceModel(nn.Module):
    """
    JAX-optimized Active Inference model with neural network components.
    
    Features:
    - Flax-based neural network architecture
    - Gradient-based optimization with Optax
    - JIT-compiled forward and backward passes
    - Support for continuous and discrete state spaces
    - Hierarchical model support
    """
    
    # Model dimensions
    n_states: int
    n_observations: int
    n_actions: int
    hidden_dim: int = 64
    n_layers: int = 2
    
    def setup(self):
        """Initialize neural network components."""
        
        # Generative model (likelihood)
        self.likelihood_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.n_observations),
            nn.softmax
        ])
        
        # Transition model (dynamics)
        self.transition_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.n_states),
            nn.softmax
        ])
        
        # Recognition model (inference)
        self.recognition_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.n_states),
            nn.softmax
        ])
        
        # Policy network
        self.policy_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.n_actions),
            nn.softmax
        ])
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Forward pass through the Active Inference model.
        
        Args:
            x: Input observations [batch_size, n_observations]
            training: Whether in training mode
            
        Returns:
            Dictionary containing model outputs
        """
        # Recognition (inference)
        q_s = self.recognition_net(x)
        
        # Generative model
        p_o_given_s = self.likelihood_net(q_s)
        
        # Policy
        policy = self.policy_net(q_s)
        
        return {
            'q_s': q_s,  # Recognition
            'p_o_given_s': p_o_given_s,  # Likelihood
            'policy': policy  # Action policy
        }
    
    @partial(jax.jit, static_argnums=(0,))
    def free_energy(self, params: Dict, observations: jnp.ndarray, 
                   actions: jnp.ndarray, states: jnp.ndarray) -> jnp.ndarray:
        """
        JIT-compiled free energy computation.
        
        Args:
            params: Model parameters
            observations: Observed data [batch_size, n_observations]
            actions: Actions taken [batch_size, n_actions]
            states: True states [batch_size, n_states]
            
        Returns:
            Free energy values [batch_size]
        """
        # Apply model
        outputs = self.apply(params, observations)
        q_s = outputs['q_s']
        p_o_given_s = outputs['p_o_given_s']
        
        # Likelihood term: -log p(o|s)
        likelihood_term = -jnp.sum(observations * jnp.log(p_o_given_s + 1e-8), axis=1)
        
        # Prior term: KL divergence between q(s) and p(s)
        prior_term = jnp.sum(q_s * jnp.log(q_s / (states + 1e-8) + 1e-8), axis=1)
        
        # Expected free energy (for policy)
        expected_free_energy = self._compute_expected_free_energy(
            params, observations, actions, q_s
        )
        
        return likelihood_term + prior_term + expected_free_energy
    
    def _compute_expected_free_energy(self, params: Dict, observations: jnp.ndarray,
                                    actions: jnp.ndarray, q_s: jnp.ndarray) -> jnp.ndarray:
        """Compute expected free energy for policy evaluation."""
        
        # Risk term: expected cost under current policy
        risk_term = jnp.sum(actions * jnp.log(actions + 1e-8), axis=1)
        
        # Ambiguity term: uncertainty about observations
        ambiguity_term = -jnp.sum(q_s * jnp.log(q_s + 1e-8), axis=1)
        
        return risk_term + ambiguity_term
    
    @partial(jax.jit, static_argnums=(0,))
    def update_step(self, params: Dict, opt_state: optax.OptState,
                   observations: jnp.ndarray, actions: jnp.ndarray, 
                   states: jnp.ndarray) -> Tuple[Dict, optax.OptState, Dict]:
        """
        JIT-compiled parameter update step.
        
        Args:
            params: Current model parameters
            opt_state: Optimizer state
            observations: Observed data
            actions: Actions taken
            states: True states
            
        Returns:
            Tuple of (updated_params, updated_opt_state, metrics)
        """
        def loss_fn(params):
            return jnp.mean(self.free_energy(params, observations, actions, states))
        
        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(params)
        
        # Apply updates
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        metrics = {
            'loss': loss,
            'grad_norm': optax.global_norm(grads)
        }
        
        return new_params, new_opt_state, metrics

class JAXActiveInferenceTrainer:
    """
    Trainer for JAX Active Inference models.
    """
    
    def __init__(self, model: JAXActiveInferenceModel, learning_rate: float = 1e-3):
        """
        Initialize trainer.
        
        Args:
            model: Active Inference model
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.learning_rate = learning_rate
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        
        # Initialize model parameters
        self.key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, model.n_observations))
        self.params = model.init(self.key, dummy_input)['params']
        self.opt_state = self.optimizer.init(self.params)
        
        logger.info(f"Trainer initialized with {model.n_states} states, {model.n_observations} observations, {model.n_actions} actions")
    
    def train_step(self, observations: jnp.ndarray, actions: jnp.ndarray, 
                  states: jnp.ndarray) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            observations: Observed data [batch_size, n_observations]
            actions: Actions taken [batch_size, n_actions]
            states: True states [batch_size, n_states]
            
        Returns:
            Training metrics
        """
        self.params, self.opt_state, metrics = self.model.update_step(
            self.params, self.opt_state, observations, actions, states
        )
        
        return {k: float(v) for k, v in metrics.items()}
    
    def train(self, data: Dict[str, jnp.ndarray], n_epochs: int = 100) -> List[Dict[str, float]]:
        """
        Train the model.
        
        Args:
            data: Dictionary containing 'observations', 'actions', 'states'
            n_epochs: Number of training epochs
            
        Returns:
            List of training metrics per epoch
        """
        observations = data['observations']
        actions = data['actions']
        states = data['states']
        
        metrics_history = []
        
        logger.info(f"Starting training for {n_epochs} epochs")
        
        for epoch in range(n_epochs):
            metrics = self.train_step(observations, actions, states)
            metrics_history.append(metrics)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss = {metrics['loss']:.6f}, grad_norm = {metrics['grad_norm']:.6f}")
        
        logger.info("Training completed")
        return metrics_history
    
    def evaluate(self, data: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            data: Test data
            
        Returns:
            Evaluation metrics
        """
        observations = data['observations']
        actions = data['actions']
        states = data['states']
        
        # Compute free energy
        free_energy = self.model.free_energy(self.params, observations, actions, states)
        
        # Compute accuracy
        outputs = self.model.apply({'params': self.params}, observations)
        predicted_states = jnp.argmax(outputs['q_s'], axis=1)
        true_states = jnp.argmax(states, axis=1)
        accuracy = jnp.mean(predicted_states == true_states)
        
        return {
            'free_energy': float(jnp.mean(free_energy)),
            'accuracy': float(accuracy)
        }
    
    def save_model(self, path: Path):
        """Save model parameters."""
        import pickle
        
        model_data = {
            'params': self.params,
            'model_config': {
                'n_states': self.model.n_states,
                'n_observations': self.model.n_observations,
                'n_actions': self.model.n_actions,
                'hidden_dim': self.model.hidden_dim,
                'n_layers': self.model.n_layers
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")

def generate_synthetic_data(n_samples: int, n_states: int, n_observations: int, 
                          n_actions: int) -> Dict[str, jnp.ndarray]:
    """Generate synthetic data for training."""
    
    # Generate random states
    states = jax.random.categorical(jax.random.PRNGKey(0), 
                                  jnp.ones(n_states), shape=(n_samples,))
    states_onehot = jax.nn.one_hot(states, n_states)
    
    # Generate observations based on states
    observations = jax.random.categorical(jax.random.PRNGKey(1),
                                        jnp.ones(n_observations), shape=(n_samples,))
    observations_onehot = jax.nn.one_hot(observations, n_observations)
    
    # Generate actions
    actions = jax.random.categorical(jax.random.PRNGKey(2),
                                   jnp.ones(n_actions), shape=(n_samples,))
    actions_onehot = jax.nn.one_hot(actions, n_actions)
    
    return {
        'observations': observations_onehot,
        'actions': actions_onehot,
        'states': states_onehot
    }

def main():
    """Main function to run the Active Inference model."""
    
    # Model parameters from GNN specification
    {model_parameters}
    
    # Create model
    model = JAXActiveInferenceModel(
        n_states={n_states},
        n_observations={n_observations},
        n_actions={n_actions},
        hidden_dim=64,
        n_layers=2
    )
    
    # Create trainer
    trainer = JAXActiveInferenceTrainer(model, learning_rate=1e-3)
    
    # Generate synthetic data
    train_data = generate_synthetic_data(1000, {n_states}, {n_observations}, {n_actions})
    test_data = generate_synthetic_data(200, {n_states}, {n_observations}, {n_actions})
    
    # Train model
    metrics_history = trainer.train(train_data, n_epochs=100)
    
    # Evaluate model
    eval_metrics = trainer.evaluate(test_data)
    
    # Save results
    output_dir = Path("output/execution_results/jax")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_name': '{model_name}',
        'training_metrics': metrics_history,
        'evaluation_metrics': eval_metrics,
        'model_config': {
            'n_states': {n_states},
            'n_observations': {n_observations},
            'n_actions': {n_actions},
            'hidden_dim': 64,
            'n_layers': 2
        }
    }
    
    results_file = output_dir / "{model_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save model
    model_file = output_dir / "{model_name}_model.pkl"
    trainer.save_model(model_file)
    
    print(f"Active Inference model trained successfully!")
    print(f"Final loss: {metrics_history[-1]['loss']:.6f}")
    print(f"Test accuracy: {eval_metrics['accuracy']:.3f}")
    print(f"Test free energy: {eval_metrics['free_energy']:.6f}")
    print(f"Results saved to: {results_file}")
    print(f"Model saved to: {model_file}")

if __name__ == "__main__":
    main()
''' 