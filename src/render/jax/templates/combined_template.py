"""
Combined JAX Template for Active Inference Models

Combined JAX implementation template that includes both POMDP solving and general
Active Inference capabilities, suitable for hierarchical and multi-agent models.

@Web: https://github.com/google/jax
@Web: https://optax.readthedocs.io
@Web: https://flax.readthedocs.io
@Web: https://pfjax.readthedocs.io
"""

COMBINED_TEMPLATE = '''
"""
JAX Combined Active Inference Model Generated from GNN Specification: {model_name}

This file contains a combined Active Inference implementation with both POMDP solving
and neural network components, suitable for hierarchical and multi-agent models.

Generated on: {timestamp}
Source: {source_file}
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from functools import partial
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
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

class JAXCombinedActiveInferenceModel(nn.Module):
    """
    JAX-optimized combined Active Inference model with POMDP and neural network components.
    
    Features:
    - POMDP solving with JIT-compiled belief updates and value iteration
    - Flax-based neural network architecture for continuous components
    - Gradient-based optimization with Optax
    - Support for hierarchical and multi-agent models
    - Hybrid discrete-continuous state spaces
    """
    
    # Model dimensions
    n_states: int
    n_observations: int
    n_actions: int
    hidden_dim: int = 64
    n_layers: int = 2
    use_pomdp: bool = True
    use_neural: bool = True
    
    def setup(self):
        """Initialize model components."""
        
        if self.use_neural:
            # Neural network components for continuous/learnable parts
            self.likelihood_net = nn.Sequential([
                nn.Dense(self.hidden_dim),
                nn.relu,
                nn.Dense(self.hidden_dim),
                nn.relu,
                nn.Dense(self.n_observations),
                nn.softmax
            ])
            
            self.transition_net = nn.Sequential([
                nn.Dense(self.hidden_dim),
                nn.relu,
                nn.Dense(self.hidden_dim),
                nn.relu,
                nn.Dense(self.n_states),
                nn.softmax
            ])
            
            self.recognition_net = nn.Sequential([
                nn.Dense(self.hidden_dim),
                nn.relu,
                nn.Dense(self.hidden_dim),
                nn.relu,
                nn.Dense(self.n_states),
                nn.softmax
            ])
            
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
        Forward pass through the combined model.
        
        Args:
            x: Input observations [batch_size, n_observations]
            training: Whether in training mode
            
        Returns:
            Dictionary containing model outputs
        """
        outputs = {}
        
        if self.use_neural:
            # Neural network components
            q_s = self.recognition_net(x)
            p_o_given_s = self.likelihood_net(q_s)
            policy = self.policy_net(q_s)
            
            outputs.update({
                'q_s': q_s,
                'p_o_given_s': p_o_given_s,
                'policy': policy
            })
        
        return outputs

class JAXCombinedPOMDPSolver:
    """
    JAX-optimized POMDP solver component for the combined model.
    """
    
    def __init__(self, 
                 A: jnp.ndarray,  # Observation model P(o|s)
                 B: List[jnp.ndarray],  # Transition models P(s'|s,u) for each action
                 C: jnp.ndarray,  # Preferences over observations
                 D: jnp.ndarray,  # Prior over initial states
                 discount: float = 0.95,
                 epsilon: float = 1e-6,
                 max_iterations: int = 1000):
        """Initialize POMDP solver with model parameters."""
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
        
        logger.info(f"POMDP solver initialized: {self.n_states} states, {self.n_observations} observations, {self.n_actions} actions")
    
    @partial(jax.jit, static_argnums=(0,))
    def belief_update(self, belief: jnp.ndarray, action: int, observation: int) -> jnp.ndarray:
        """JIT-compiled belief update."""
        belief_prediction = self.B[action] @ belief
        numerator = self.A[observation, :] * belief_prediction
        denominator = jnp.sum(numerator)
        denominator = jnp.where(denominator > 1e-10, denominator, 1e-10)
        return numerator / denominator
    
    @partial(jax.jit, static_argnums=(0,))
    def alpha_vector_backup(self, belief_points: jnp.ndarray, alpha_vectors: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled alpha vector backup."""
        def backup_single_action(action):
            def compute_value_for_belief(belief):
                immediate_reward = jnp.sum(belief * jnp.sum(self.A * self.C, axis=0))
                future_values = []
                
                for obs in range(self.n_observations):
                    updated_belief = self.belief_update(belief, action, obs)
                    alpha_values = alpha_vectors @ updated_belief
                    max_value = jnp.max(alpha_values)
                    obs_prob = jnp.sum(self.A[obs, :] * (self.B[action] @ belief))
                    future_values.append(obs_prob * max_value)
                
                total_future_value = jnp.sum(jnp.array(future_values))
                return immediate_reward + self.discount * total_future_value
            
            values = jax.vmap(compute_value_for_belief)(belief_points)
            return jnp.max(values)
        
        action_values = jax.vmap(backup_single_action)(jnp.arange(self.n_actions))
        best_action = jnp.argmax(action_values)
        
        # Compute alpha vector for best action
        alpha = jnp.zeros(self.n_states)
        for s in range(self.n_states):
            immediate_reward = jnp.sum(self.A[:, s] * self.C)
            future_value = 0.0
            for obs in range(self.n_observations):
                obs_prob = self.A[obs, s]
                state_transition = self.B[best_action][:, s]
                alpha_values = alpha_vectors @ state_transition
                max_value = jnp.max(alpha_values)
                future_value += obs_prob * max_value
            alpha = alpha.at[s].set(immediate_reward + self.discount * future_value)
        
        return alpha
    
    def value_iteration(self, belief_points: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, List[int]]:
        """Point-based value iteration with convergence checking."""
        if belief_points is None:
            n_points = min(100, 2**self.n_states)
            belief_points = self._generate_uniform_belief_points(n_points)
        
        belief_points = jnp.array(belief_points, dtype=jnp.float64)
        n_points = belief_points.shape[0]
        
        # Initialize alpha vectors
        alpha_vectors = jnp.zeros((self.n_actions, self.n_states), dtype=jnp.float64)
        for a in range(self.n_actions):
            for s in range(self.n_states):
                alpha_vectors = alpha_vectors.at[a, s].set(jnp.sum(self.A[:, s] * self.C))
        
        logger.info(f"Starting value iteration with {n_points} belief points")
        
        # Value iteration loop
        for iteration in range(self.max_iterations):
            old_alpha_vectors = alpha_vectors.copy()
            
            new_alpha_vectors = []
            for i in range(n_points):
                new_alpha = self.alpha_vector_backup(belief_points[i:i+1], alpha_vectors)
                new_alpha_vectors.append(new_alpha)
            
            alpha_vectors = jnp.vstack([alpha_vectors] + new_alpha_vectors)
            
            max_change = jnp.max(jnp.abs(alpha_vectors - old_alpha_vectors))
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: max change = {max_change:.6f}")
            
            if max_change < self.epsilon:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
        
        policy = self._extract_policy(alpha_vectors, belief_points)
        return alpha_vectors, policy
    
    def _generate_uniform_belief_points(self, n_points: int) -> jnp.ndarray:
        """Generate uniform belief points."""
        import numpy as np
        points = []
        for _ in range(n_points):
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
        """Solve the POMDP and return results."""
        start_time = time.time()
        
        alpha_vectors, policy = self.value_iteration()
        initial_value = jnp.max(alpha_vectors @ self.D)
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

class JAXCombinedTrainer:
    """
    Trainer for combined JAX Active Inference models.
    """
    
    def __init__(self, model: JAXCombinedActiveInferenceModel, 
                 pomdp_solver: Optional[JAXCombinedPOMDPSolver] = None,
                 learning_rate: float = 1e-3):
        """Initialize trainer."""
        self.model = model
        self.pomdp_solver = pomdp_solver
        self.learning_rate = learning_rate
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        
        # Initialize model parameters
        self.key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, model.n_observations))
        self.params = model.init(self.key, dummy_input)['params']
        self.opt_state = self.optimizer.init(self.params)
        
        logger.info(f"Combined trainer initialized")
    
    @partial(jax.jit, static_argnums=(0,))
    def free_energy(self, params: Dict, observations: jnp.ndarray, 
                   actions: jnp.ndarray, states: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled free energy computation."""
        outputs = self.model.apply({'params': params}, observations)
        q_s = outputs['q_s']
        p_o_given_s = outputs['p_o_given_s']
        
        # Likelihood term
        likelihood_term = -jnp.sum(observations * jnp.log(p_o_given_s + 1e-8), axis=1)
        
        # Prior term
        prior_term = jnp.sum(q_s * jnp.log(q_s / (states + 1e-8) + 1e-8), axis=1)
        
        # Expected free energy
        risk_term = jnp.sum(actions * jnp.log(actions + 1e-8), axis=1)
        ambiguity_term = -jnp.sum(q_s * jnp.log(q_s + 1e-8), axis=1)
        expected_free_energy = risk_term + ambiguity_term
        
        return likelihood_term + prior_term + expected_free_energy
    
    @partial(jax.jit, static_argnums=(0,))
    def update_step(self, params: Dict, opt_state: optax.OptState,
                   observations: jnp.ndarray, actions: jnp.ndarray, 
                   states: jnp.ndarray) -> Tuple[Dict, optax.OptState, Dict]:
        """JIT-compiled parameter update step."""
        def loss_fn(params):
            return jnp.mean(self.free_energy(params, observations, actions, states))
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        metrics = {
            'loss': loss,
            'grad_norm': optax.global_norm(grads)
        }
        
        return new_params, new_opt_state, metrics
    
    def train_step(self, observations: jnp.ndarray, actions: jnp.ndarray, 
                  states: jnp.ndarray) -> Dict[str, float]:
        """Single training step."""
        self.params, self.opt_state, metrics = self.update_step(
            self.params, self.opt_state, observations, actions, states
        )
        return {k: float(v) for k, v in metrics.items()}
    
    def train(self, data: Dict[str, jnp.ndarray], n_epochs: int = 100) -> List[Dict[str, float]]:
        """Train the model."""
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
        """Evaluate the model."""
        observations = data['observations']
        actions = data['actions']
        states = data['states']
        
        free_energy = self.free_energy(self.params, observations, actions, states)
        outputs = self.model.apply({'params': self.params}, observations)
        predicted_states = jnp.argmax(outputs['q_s'], axis=1)
        true_states = jnp.argmax(states, axis=1)
        accuracy = jnp.mean(predicted_states == true_states)
        
        return {
            'free_energy': float(jnp.mean(free_energy)),
            'accuracy': float(accuracy)
        }
    
    def solve_pomdp(self) -> Optional[Dict[str, Any]]:
        """Solve POMDP if solver is available."""
        if self.pomdp_solver is not None:
            return self.pomdp_solver.solve()
        return None
    
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
                'n_layers': self.model.n_layers,
                'use_pomdp': self.model.use_pomdp,
                'use_neural': self.model.use_neural
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")

def generate_synthetic_data(n_samples: int, n_states: int, n_observations: int, 
                          n_actions: int) -> Dict[str, jnp.ndarray]:
    """Generate synthetic data for training."""
    
    states = jax.random.categorical(jax.random.PRNGKey(0), 
                                  jnp.ones(n_states), shape=(n_samples,))
    states_onehot = jax.nn.one_hot(states, n_states)
    
    observations = jax.random.categorical(jax.random.PRNGKey(1),
                                        jnp.ones(n_observations), shape=(n_samples,))
    observations_onehot = jax.nn.one_hot(observations, n_observations)
    
    actions = jax.random.categorical(jax.random.PRNGKey(2),
                                   jnp.ones(n_actions), shape=(n_samples,))
    actions_onehot = jax.nn.one_hot(actions, n_actions)
    
    return {
        'observations': observations_onehot,
        'actions': actions_onehot,
        'states': states_onehot
    }

def main():
    """Main function to run the combined Active Inference model."""
    
    # Model parameters from GNN specification
    {model_parameters}
    
    # Create POMDP solver if parameters are available
    pomdp_solver = None
    if {use_pomdp}:
        try:
            pomdp_solver = JAXCombinedPOMDPSolver(A=A, B=B, C=C, D=D, discount={discount})
            logger.info("POMDP solver created successfully")
        except Exception as e:
            logger.warning(f"Failed to create POMDP solver: {e}")
    
    # Create combined model
    model = JAXCombinedActiveInferenceModel(
        n_states={n_states},
        n_observations={n_observations},
        n_actions={n_actions},
        hidden_dim=64,
        n_layers=2,
        use_pomdp={use_pomdp},
        use_neural={use_neural}
    )
    
    # Create trainer
    trainer = JAXCombinedTrainer(model, pomdp_solver, learning_rate=1e-3)
    
    results = {
        'model_name': '{model_name}',
        'model_config': {
            'n_states': {n_states},
            'n_observations': {n_observations},
            'n_actions': {n_actions},
            'hidden_dim': 64,
            'n_layers': 2,
            'use_pomdp': {use_pomdp},
            'use_neural': {use_neural}
        }
    }
    
    # Train neural components if enabled
    if {use_neural}:
        train_data = generate_synthetic_data(1000, {n_states}, {n_observations}, {n_actions})
        test_data = generate_synthetic_data(200, {n_states}, {n_observations}, {n_actions})
        
        metrics_history = trainer.train(train_data, n_epochs=100)
        eval_metrics = trainer.evaluate(test_data)
        
        results.update({
            'training_metrics': metrics_history,
            'evaluation_metrics': eval_metrics
        })
        
        print(f"Neural training completed!")
        print(f"Final loss: {metrics_history[-1]['loss']:.6f}")
        print(f"Test accuracy: {eval_metrics['accuracy']:.3f}")
        print(f"Test free energy: {eval_metrics['free_energy']:.6f}")
    
    # Solve POMDP if enabled
    if {use_pomdp} and pomdp_solver is not None:
        pomdp_results = trainer.solve_pomdp()
        if pomdp_results:
            results['pomdp_results'] = pomdp_results
            print(f"POMDP solved successfully!")
            print(f"Initial value: {pomdp_results['initial_value']:.6f}")
            print(f"Solve time: {pomdp_results['solve_time']:.3f}s")
    
    # Save results
    output_dir = Path("output/execution_results/jax")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "{model_name}_combined_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save model
    model_file = output_dir / "{model_name}_combined_model.pkl"
    trainer.save_model(model_file)
    
    print(f"Combined Active Inference model completed successfully!")
    print(f"Results saved to: {results_file}")
    print(f"Model saved to: {model_file}")

if __name__ == "__main__":
    main()
''' 